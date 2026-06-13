"""Interactive libngspice shared-library command session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from monata.netlist import Circuit, render_ngspice
from monata.sim.backends.ngspice_common import ngspice_control_path as _control_path
from monata.sim.backends.ngspice_shared_commands import (
    command_part as _command_part,
    command_value as _command_value,
    join_command_outputs as _join_command_outputs,
    parse_key_value_output as _parse_key_value_output,
)
from monata.sim.backends.ngspice_shared_ffi import (
    NgspiceSharedError,
    NgspiceSharedLibraryError,
    load_library as _load_library,
    new_ffi as _new_ffi,
)


class NgspiceSharedCommandError(NgspiceSharedError):
    """Raised when a libngspice command returns an error."""


@dataclass(frozen=True)
class NgspiceCallbackEvent:
    """A recorded libngspice callback event."""

    kind: str
    ngspice_id: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class NgspiceInitVector:
    """Vector metadata reported by ngspice init-data callbacks."""

    number: int
    name: str
    is_real: bool


@dataclass(frozen=True)
class NgspiceInitData:
    """Plot metadata reported by ngspice init-data callbacks."""

    name: str
    title: str
    date: str
    analysis_type: str
    vectors: tuple[NgspiceInitVector, ...]


class NgspiceSharedCallbacks:
    """Optional handler protocol for libngspice shared callbacks."""

    def send_char(self, session: NgspiceSharedSession, message: str, ngspice_id: int) -> int | None:
        return 0

    def send_stat(self, session: NgspiceSharedSession, message: str, ngspice_id: int) -> int | None:
        return 0

    def controlled_exit(
        self,
        session: NgspiceSharedSession,
        status: int,
        immediate: bool,
        quit_exit: bool,
        ngspice_id: int,
    ) -> int | None:
        return status

    def background_thread_running(
        self,
        session: NgspiceSharedSession,
        is_running: bool,
        ngspice_id: int,
    ) -> int | None:
        return 0

    def send_data(
        self,
        session: NgspiceSharedSession,
        values: dict[str, float | complex],
        vector_count: int,
        ngspice_id: int,
    ) -> int | None:
        return 0

    def send_init_data(self, session: NgspiceSharedSession, data: NgspiceInitData, ngspice_id: int) -> int | None:
        return 0

    def get_vsrc_data(self, session: NgspiceSharedSession, time_value: float, node: str, ngspice_id: int) -> float | None:
        return None

    def get_isrc_data(self, session: NgspiceSharedSession, time_value: float, node: str, ngspice_id: int) -> float | None:
        return None


class NgspiceSharedSession:
    """Small libngspice command session.

    The session intentionally exposes a narrow command/vector surface. It is
    enough for interactive source/load/run/alter workflows and for the Monata
    backend runner, while keeping the subprocess backend as the safer default.
    """

    def __init__(
        self,
        library: str | None = None,
        *,
        callbacks: NgspiceSharedCallbacks | None = None,
        enable_data_callbacks: bool = False,
        enable_sync_callbacks: bool = False,
    ) -> None:
        self.library_name = library
        self._ffi = _new_ffi()
        self._lib = _load_library(self._ffi, library)
        self.callbacks = callbacks or NgspiceSharedCallbacks()
        self._stdout: list[str] = []
        self._stderr: list[str] = []
        self._stats: list[str] = []
        self._callback_events: list[NgspiceCallbackEvent] = []
        self._data_events: list[dict[str, float | complex]] = []
        self._init_data_events: list[NgspiceInitData] = []
        self._is_running = False
        self._lock = RLock()
        self._handle = self._ffi.new_handle(self)
        self._send_char_cb = self._ffi.callback("SendChar *", self._send_char)
        self._send_stat_cb = self._ffi.callback("SendStat *", self._send_stat)
        self._exit_cb = self._ffi.callback("ControlledExit *", self._controlled_exit)
        self._send_data_cb = (
            self._ffi.callback("SendData *", self._send_data) if enable_data_callbacks else self._ffi.NULL
        )
        self._send_init_data_cb = (
            self._ffi.callback("SendInitData *", self._send_init_data) if enable_data_callbacks else self._ffi.NULL
        )
        self._running_cb = self._ffi.callback("BGThreadRunning *", self._background_thread_running)
        rc = self._lib.ngSpice_Init(
            self._send_char_cb,
            self._send_stat_cb,
            self._exit_cb,
            self._send_data_cb,
            self._send_init_data_cb,
            self._running_cb,
            self._handle,
        )
        if rc:
            raise NgspiceSharedLibraryError(f"ngSpice_Init returned {rc}")
        self._sync_id = self._ffi.NULL
        self._get_vsrc_data_cb = self._ffi.NULL
        self._get_isrc_data_cb = self._ffi.NULL
        if enable_sync_callbacks:
            self._init_sync_callbacks()

    @classmethod
    def available(cls, library: str | None = None) -> bool:
        try:
            ffi = _new_ffi()
            _load_library(ffi, library)
        except NgspiceSharedLibraryError:
            return False
        return True

    def __enter__(self) -> NgspiceSharedSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.destroy("all")

    @property
    def stdout(self) -> str:
        return "\n".join(self._stdout)

    @property
    def stderr(self) -> str:
        return "\n".join(self._stderr)

    @property
    def stats(self) -> str:
        return "\n".join(self._stats)

    @property
    def callback_events(self) -> tuple[NgspiceCallbackEvent, ...]:
        return tuple(self._callback_events)

    @property
    def data_events(self) -> tuple[dict[str, float | complex], ...]:
        return tuple(self._data_events)

    @property
    def init_data_events(self) -> tuple[NgspiceInitData, ...]:
        return tuple(self._init_data_events)

    @property
    def is_running(self) -> bool:
        try:
            return bool(self._lib.ngSpice_running())
        except AttributeError:
            return self._is_running

    def clear_output(self) -> None:
        self._stdout.clear()
        self._stderr.clear()
        self._stats.clear()

    def clear_callback_events(self) -> None:
        self._callback_events.clear()
        self._data_events.clear()
        self._init_data_events.clear()

    def command(self, command: str) -> str:
        """Execute an ngspice command and return callback stdout."""

        with self._lock:
            self.clear_output()
            encoded = str(command).encode("utf-8")
            rc = self._lib.ngSpice_Command(encoded)
            if rc:
                output = self.stderr or self.stdout
                detail = f": {output}" if output else ""
                raise NgspiceSharedCommandError(f"ngSpice_Command {command!r} returned {rc}{detail}")
            if self.stderr:
                lowered = self.stderr.lower()
                if "error" in lowered or "fatal" in lowered:
                    raise NgspiceSharedCommandError(f"ngspice command failed: {command}: {self.stderr}")
            return self.stdout

    def load_circuit(self, circuit: str | Circuit) -> None:
        """Load a rendered circuit into libngspice."""

        text = render_ngspice(circuit) if isinstance(circuit, Circuit) else str(circuit)
        lines = [line for line in text.splitlines() if line.strip()]
        lines.append("")
        with self._lock:
            self.clear_output()
            keepalive = [self._ffi.new("char[]", line.encode("utf-8")) for line in lines]
            keepalive.append(self._ffi.NULL)
            array = self._ffi.new("char *[]", keepalive)
            rc = self._lib.ngSpice_Circ(array)
            if rc:
                output = self.stderr or self.stdout
                detail = f": {output}" if output else ""
                raise NgspiceSharedCommandError(f"ngSpice_Circ returned {rc}{detail}")
            if self.stderr and "error" in self.stderr.lower():
                raise NgspiceSharedCommandError(f"ngspice circuit load failed: {self.stderr}")

    def source(self, path: str | Path) -> str:
        return self.command(f"source {_control_path(Path(path))}")

    def run(self, *, background: bool = False) -> str:
        return self.command("bg_run" if background else "run")

    def halt(self) -> str:
        return self.command("bg_halt")

    def resume(self, *, background: bool = True) -> str:
        return self.command("bg_resume" if background else "resume")

    def set_breakpoint(self, time_value: float) -> bool:
        try:
            return bool(self._lib.ngSpice_SetBkpt(float(time_value)))
        except AttributeError as exc:
            raise NgspiceSharedLibraryError("libngspice does not expose ngSpice_SetBkpt") from exc

    def reset(self) -> str:
        return self.command("reset")

    def delete(self, debug_number: int | str) -> str:
        """Remove a trace or breakpoint by ngspice debug number."""

        return self.command(f"delete {_command_part(debug_number, 'debug number')}")

    def remove_circuit(self) -> str:
        return self.command("remcirc")

    def destroy(self, plot_name: str = "all") -> str:
        try:
            return self.command(f"destroy {_command_part(plot_name, 'plot name')}")
        except NgspiceSharedCommandError:
            return ""

    def device_help(self, device: str) -> str:
        """Return ngspice help text for a device type."""

        return self.command(f"devhelp {_command_part(device, 'device').lower()}")

    def save(self, vector: str) -> str:
        return self.command(f"save {_command_part(vector, 'save vector')}")

    def alter_device(self, device: str, **params: object) -> str:
        return self._alter("alter", device, params)

    def alter_model(self, model: str, **params: object) -> str:
        return self._alter("altermod", model, params)

    def show(self, device: str) -> dict[str, object]:
        """Return parsed ``show`` output for a device instance."""

        return _parse_key_value_output(self.command(f"show {_command_part(device, 'device').lower()}"))

    def showmod(self, model: str) -> dict[str, object]:
        """Return parsed ``showmod`` output for a model."""

        return _parse_key_value_output(self.command(f"showmod {_command_part(model, 'model').lower()}"))

    def option(self, *flags: str, **options: object) -> str:
        """Set ngspice options and return any command output."""

        return self._set_command("option", "option flag", "option name", flags, options)

    def quit(self) -> str:
        """Ask ngspice to quit without an interactive confirmation prompt."""

        return _join_command_outputs([self.set("noaskquit"), self.command("quit")])

    def resource_usage(self, *resources: str) -> dict[str, object]:
        """Return parsed ngspice ``rusage`` statistics."""

        requested = tuple(resources) or ("everything",)
        command = "rusage " + " ".join(_command_part(resource, "resource name") for resource in requested)
        return _parse_key_value_output(self.command(command))

    def set(self, *flags: str, **variables: object) -> str:
        """Set ngspice shell variables."""

        return self._set_command("set", "set flag", "set variable", flags, variables)

    def set_circuit(self, name: str) -> str:
        """Change the active ngspice circuit."""

        return self.command(f"setcirc {_command_part(name, 'circuit name')}")

    def status(self) -> str:
        """Return ngspice breakpoint status output."""

        return self.command("status")

    def step(self, number_of_steps: int | None = None) -> str:
        """Run one or more transient time points."""

        if number_of_steps is None:
            return self.command("step")
        if number_of_steps <= 0:
            raise ValueError("number_of_steps must be positive")
        return self.command(f"step {number_of_steps}")

    def stop(self, *conditions: str, after: int | float | None = None) -> str:
        """Set ngspice breakpoints for one or more ``when`` conditions."""

        parts = ["stop"]
        if after is not None:
            parts.extend(("after", _command_part(after, "stop after value")))
        for condition in conditions:
            parts.extend(("when", _command_part(condition, "stop condition")))
        return self.command(" ".join(parts))

    def trace(self, *vectors: str) -> str:
        """Trace one or more ngspice vectors."""

        if not vectors:
            raise ValueError("trace requires at least one vector")
        return self.command("trace " + " ".join(_command_part(vector, "trace vector") for vector in vectors))

    def unset(self, *variables: str) -> str:
        """Unset ngspice shell variables."""

        if not variables:
            raise ValueError("unset requires at least one variable")
        return _join_command_outputs(
            [self.command(f"unset {_command_part(variable, 'unset variable')}") for variable in variables]
        )

    def where(self) -> str:
        """Return ngspice diagnostics for a troublesome node or device."""

        return self.command("where")

    def listing(self) -> str:
        """Return the current ngspice circuit listing."""

        return self.command("listing")

    def plot_names(self) -> tuple[str, ...]:
        return self._string_array(self._lib.ngSpice_AllPlots())

    def current_plot(self) -> str | None:
        value = self._lib.ngSpice_CurPlot()
        if value == self._ffi.NULL:
            return None
        return self._string(value)

    def vector_names(self, plot_name: str | None = None) -> tuple[str, ...]:
        plot = plot_name or self.current_plot()
        if not plot:
            return ()
        return self._string_array(self._lib.ngSpice_AllVecs(plot.encode("utf-8")))

    def vector(self, name: str, *, plot_name: str | None = None) -> np.ndarray:
        plot = plot_name or self.current_plot()
        vector_name = str(name)
        qualified = vector_name if plot is None or "." in vector_name else f"{plot}.{vector_name}"
        info = self._lib.ngGet_Vec_Info(qualified.encode("utf-8"))
        if info == self._ffi.NULL:
            raise KeyError(vector_name)
        length = int(info.v_length)
        if info.v_compdata != self._ffi.NULL:
            raw = np.frombuffer(self._ffi.buffer(info.v_compdata, length * 16), dtype=np.float64)
            values = np.array(raw[0::2], dtype=np.complex128)
            values.imag = raw[1::2]
            return values
        if info.v_realdata == self._ffi.NULL:
            return np.array([], dtype=float)
        raw = np.frombuffer(self._ffi.buffer(info.v_realdata, length * 8), dtype=np.float64)
        return np.array(raw, dtype=np.float64)

    def vectors(self, plot_name: str | None = None) -> dict[str, np.ndarray]:
        return {name: self.vector(name, plot_name=plot_name) for name in self.vector_names(plot_name)}

    def _set_command(
        self,
        command: str,
        flag_label: str,
        variable_label: str,
        flags: tuple[str, ...],
        variables: dict[str, object],
    ) -> str:
        outputs = [self.command(f"{command} {_command_part(flag, flag_label)}") for flag in flags]
        for name, value in variables.items():
            variable_name = _command_part(name, variable_label)
            outputs.append(self.command(f"{command} {variable_name} = {_command_value(value)}"))
        return _join_command_outputs(outputs)

    def _alter(self, command: str, target: str, params: dict[str, object]) -> str:
        outputs = []
        target_name = _command_part(target, "alter target")
        for name, value in params.items():
            parameter_name = _command_part(name, "alter parameter")
            outputs.append(self.command(f"{command} {target_name} {parameter_name}={_command_value(value)}"))
        return _join_command_outputs(outputs)

    def _init_sync_callbacks(self) -> None:
        self._get_vsrc_data_cb = self._ffi.callback("GetVSRCData *", self._get_vsrc_data)
        self._get_isrc_data_cb = self._ffi.callback("GetISRCData *", self._get_isrc_data)
        self._sync_id = self._ffi.new("int *", 0)
        try:
            rc = self._lib.ngSpice_Init_Sync(
                self._get_vsrc_data_cb,
                self._get_isrc_data_cb,
                self._ffi.NULL,
                self._sync_id,
                self._handle,
            )
        except AttributeError as exc:
            raise NgspiceSharedLibraryError("libngspice does not expose ngSpice_Init_Sync") from exc
        if rc:
            raise NgspiceSharedLibraryError(f"ngSpice_Init_Sync returned {rc}")

    def _string_array(self, array: Any) -> tuple[str, ...]:
        if array == self._ffi.NULL:
            return ()
        values: list[str] = []
        index = 0
        while array[index] != self._ffi.NULL:
            values.append(self._string(array[index]))
            index += 1
        return tuple(values)

    def _string(self, value: Any) -> str:
        return self._ffi.string(value).decode("utf-8", errors="replace")

    def _send_char(self, message: Any, ngspice_id: int, user_data: Any) -> int:
        text = self._string(message).strip()
        lowered = text.lower()
        if lowered.startswith("stderr "):
            self._stderr.append(text[7:])
        elif lowered.startswith("stdout "):
            self._stdout.append(text[7:])
        elif "error" in lowered or "fatal" in lowered:
            self._stderr.append(text)
        elif text:
            self._stdout.append(text)
        self._record_callback("send_char", ngspice_id, {"message": text})
        return _callback_rc(self.callbacks.send_char(self, text, ngspice_id))

    def _send_stat(self, message: Any, ngspice_id: int, user_data: Any) -> int:
        text = self._string(message).strip()
        if text:
            self._stats.append(text)
        self._record_callback("send_stat", ngspice_id, {"message": text})
        return _callback_rc(self.callbacks.send_stat(self, text, ngspice_id))

    def _controlled_exit(self, status: int, immediate: bool, quit_exit: bool, ngspice_id: int, user_data: Any) -> int:
        payload = {
            "status": int(status),
            "immediate": bool(immediate),
            "quit_exit": bool(quit_exit),
        }
        self._record_callback("controlled_exit", ngspice_id, payload)
        rc = self.callbacks.controlled_exit(self, int(status), bool(immediate), bool(quit_exit), ngspice_id)
        return int(status) if rc is None else int(rc)

    def _background_thread_running(self, is_running: bool, ngspice_id: int, user_data: Any) -> int:
        self._is_running = bool(is_running)
        self._record_callback("background_thread_running", ngspice_id, {"is_running": self._is_running})
        return _callback_rc(self.callbacks.background_thread_running(self, self._is_running, ngspice_id))

    def _send_data(self, data: Any, number_of_vectors: int, ngspice_id: int, user_data: Any) -> int:
        values = self._vector_values(data, number_of_vectors)
        self._data_events.append(values)
        self._record_callback("send_data", ngspice_id, {"values": values, "vector_count": int(number_of_vectors)})
        return _callback_rc(self.callbacks.send_data(self, values, int(number_of_vectors), ngspice_id))

    def _send_init_data(self, data: Any, ngspice_id: int, user_data: Any) -> int:
        init_data = self._init_data(data)
        self._init_data_events.append(init_data)
        self._record_callback("send_init_data", ngspice_id, {"data": init_data})
        return _callback_rc(self.callbacks.send_init_data(self, init_data, ngspice_id))

    def _get_vsrc_data(self, voltage: Any, time_value: float, node: Any, ngspice_id: int, user_data: Any) -> int:
        return self._get_source_data(
            voltage,
            time_value,
            node,
            ngspice_id,
            "get_vsrc_data",
            self.callbacks.get_vsrc_data,
        )

    def _get_isrc_data(self, current: Any, time_value: float, node: Any, ngspice_id: int, user_data: Any) -> int:
        return self._get_source_data(
            current,
            time_value,
            node,
            ngspice_id,
            "get_isrc_data",
            self.callbacks.get_isrc_data,
        )

    def _get_source_data(
        self,
        target: Any,
        time_value: float,
        node: Any,
        ngspice_id: int,
        kind: str,
        callback: Any,
    ) -> int:
        node_name = self._string(node)
        value = callback(self, float(time_value), node_name, ngspice_id)
        if value is not None:
            target[0] = float(value)
        self._record_callback(
            kind,
            ngspice_id,
            {"time": float(time_value), "node": node_name, "value": value},
        )
        return 0

    def _vector_values(self, data: Any, number_of_vectors: int) -> dict[str, float | complex]:
        if data == self._ffi.NULL:
            return {}
        values: dict[str, float | complex] = {}
        count = int(number_of_vectors or data.veccount)
        for index in range(count):
            item = data.vecsa[index]
            if item == self._ffi.NULL:
                continue
            name = self._string(item.name)
            value = complex(float(item.creal), float(item.cimag)) if item.is_complex else float(item.creal)
            values[name] = value
        return values

    def _init_data(self, data: Any) -> NgspiceInitData:
        if data == self._ffi.NULL:
            return NgspiceInitData("", "", "", "", ())
        vectors = []
        for index in range(int(data.veccount)):
            item = data.vecs[index]
            if item == self._ffi.NULL:
                continue
            vectors.append(
                NgspiceInitVector(
                    number=int(item.number),
                    name=self._string(item.vecname),
                    is_real=bool(item.is_real),
                )
            )
        return NgspiceInitData(
            name=self._string(data.name),
            title=self._string(data.title),
            date=self._string(data.date),
            analysis_type=self._string(data.type),
            vectors=tuple(vectors),
        )

    def _record_callback(self, kind: str, ngspice_id: int, payload: dict[str, Any]) -> None:
        self._callback_events.append(NgspiceCallbackEvent(kind, int(ngspice_id), payload))


def _callback_rc(value: int | None) -> int:
    return 0 if value is None else int(value)


__all__ = [
    "NgspiceCallbackEvent",
    "NgspiceInitData",
    "NgspiceInitVector",
    "NgspiceSharedCommandError",
    "NgspiceSharedCallbacks",
    "NgspiceSharedError",
    "NgspiceSharedLibraryError",
    "NgspiceSharedSession",
]
