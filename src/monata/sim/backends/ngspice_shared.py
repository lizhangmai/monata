"""Shared-library ngspice backend runner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import RLock
import tempfile
import time
from typing import Any

from monata.netlist import render_ngspice
from monata.sim.backends.base import BackendCapabilities, BackendTaskPlan
from monata.sim.backends.ngspice_common import (
    NGSPICE_ANALYSES as _NGSPICE_ANALYSES,
    NGSPICE_MODEL_ARTIFACTS as _NGSPICE_MODEL_ARTIFACTS,
    NGSPICE_PARSER_CONTRACT as _NGSPICE_PARSER_CONTRACT,
    NGSPICE_RESULT_MODES as _NGSPICE_RESULT_MODES,
    NGSPICE_SOURCE_MUTATIONS as _NGSPICE_SOURCE_MUTATIONS,
    NgspicePreflightFailure as _NgspicePreflightFailure,
    backend_task_plan as _backend_task_plan,
    exception_metadata as _exception_metadata,
    failure_result as _common_failure_result,
    ngspice_control_path as _control_path,
    netlist_lines_with_task_directives as _netlist_lines_with_task_directives,
    plan_command_lines as _plan_command_lines,
    persist_runner_artifacts as _persist_runner_artifacts,
    prepare_runner_task as _prepare_runner_task,
    raise_backend_exceptions as _raise_backend_exceptions,
    result_metadata as _common_result_metadata,
    task_measure_specs as _task_measure_specs,
)
from monata.sim.backends.ngspice_stdout import noise_aborted as _noise_aborted, parse_measure_print
from monata.sim.backends.ngspice_parse import (
    parse_output as _parse_output,
)
from monata.sim.backends.ngspice_plan import NgspiceTaskPlan
from monata.sim.backends.ngspice_shared_session import (
    NgspiceCallbackEvent,
    NgspiceInitData,
    NgspiceInitVector,
    NgspiceSharedCallbacks,
    NgspiceSharedCommandError,
    NgspiceSharedError,
    NgspiceSharedLibraryError,
    NgspiceSharedSession,
)
from monata.sim.results import SimResult
from monata.sim.task import SimTask


SessionFactory = Callable[[str | None], Any]


class NgspiceSharedRunner:
    """Run Monata tasks through libngspice instead of the ngspice executable."""

    name = "ngspice-shared"
    capabilities = BackendCapabilities(
        analyses=_NGSPICE_ANALYSES,
        result_modes=_NGSPICE_RESULT_MODES,
        source_mutations=_NGSPICE_SOURCE_MUTATIONS,
        model_artifacts=_NGSPICE_MODEL_ARTIFACTS,
        native_monte_carlo=False,
        renderer_contract="ngspice-netlist",
        runner_contract="shared-library",
        parser_contract=_NGSPICE_PARSER_CONTRACT,
    )

    _run_lock = RLock()

    def __init__(
        self,
        library: str | None = None,
        session_factory: SessionFactory = NgspiceSharedSession,
    ) -> None:
        self.library = library
        self._session_factory = session_factory

    @classmethod
    def available(cls, library: str | None = None) -> bool:
        return NgspiceSharedSession.available(library)

    def validate_task(self, task: SimTask) -> SimResult | None:
        prepared = self._prepare_task(task)
        if isinstance(prepared, _NgspicePreflightFailure):
            return _failed(task, prepared.message, prepared.reason, time.time())
        return None

    def plan_task(self, task: SimTask) -> BackendTaskPlan:
        prepared = self._prepare_task(task)
        if isinstance(prepared, _NgspicePreflightFailure):
            raise ValueError(prepared.message)
        return _backend_task_plan(self.name, prepared.plan)

    def run(self, task: SimTask) -> SimResult:
        started = time.time()
        prepared = self._prepare_task(task)
        if isinstance(prepared, _NgspicePreflightFailure):
            return _failed(task, prepared.message, prepared.reason, started)
        plan = prepared.plan
        mutation_projection = prepared.mutation_projection

        try:
            with self._run_lock:
                with tempfile.TemporaryDirectory(prefix="monata-ngspice-shared-") as tmp:
                    tmp_path = Path(tmp)
                    output_path = tmp_path / ("result.raw" if plan.extraction == "rawfile" else "result.dat")
                    netlist_text = _with_task_directives(
                        render_ngspice(mutation_projection.circuit),
                        task,
                        param_overrides=mutation_projection.param_overrides,
                    )
                    with self._session_factory(self.library) as session:
                        session.load_circuit(netlist_text)
                        try:
                            stdout = _execute_plan(session, plan, output_path)
                        except NgspiceSharedCommandError as exc:
                            elapsed = time.time() - started
                            artifact_metadata = _persist_shared_artifacts(
                                task,
                                plan,
                                netlist_text=netlist_text,
                                tmp_path=tmp_path,
                                output_path=output_path,
                                stdout="",
                                stderr=str(exc),
                                status="failed",
                                reason="shared_library_failed",
                                elapsed=elapsed,
                            )
                            return _failed(
                                task,
                                str(exc),
                                "shared_library_failed",
                                started,
                                extra_metadata=artifact_metadata,
                            )
                        stderr = getattr(session, "stderr", "")
                        if plan.analysis_name == "noise" and _noise_aborted(stdout, stderr):
                            elapsed = time.time() - started
                            artifact_metadata = _persist_shared_artifacts(
                                task,
                                plan,
                                netlist_text=netlist_text,
                                tmp_path=tmp_path,
                                output_path=output_path,
                                stdout=stdout,
                                stderr=stderr,
                                status="failed",
                                reason="shared_library_failed",
                                elapsed=elapsed,
                            )
                            return _failed(
                                task,
                                (stderr or stdout or "ngspice noise simulation aborted").strip(),
                                "shared_library_failed",
                                started,
                                extra_metadata=artifact_metadata,
                            )
                        try:
                            sweep_var, waveforms, analysis_result, extra_metadata = _parse_output(output_path, stdout, plan)
                        except Exception as exc:
                            if _raise_backend_exceptions(task):
                                raise
                            elapsed = time.time() - started
                            artifact_metadata = _persist_shared_artifacts(
                                task,
                                plan,
                                netlist_text=netlist_text,
                                tmp_path=tmp_path,
                                output_path=output_path,
                                stdout=stdout,
                                stderr=stderr,
                                status="failed",
                                reason="parser_failed",
                                elapsed=elapsed,
                            )
                            return _failed(
                                task,
                                str(exc),
                                "parser_failed",
                                started,
                                extra_metadata=artifact_metadata,
                            )
                        measures = parse_measure_print(stdout, _task_measure_specs(task))
                        if measures:
                            extra_metadata = {**extra_metadata, "measures": measures.to_dict()}
                        if not waveforms:
                            elapsed = time.time() - started
                            artifact_metadata = _persist_shared_artifacts(
                                task,
                                plan,
                                netlist_text=netlist_text,
                                tmp_path=tmp_path,
                                output_path=output_path,
                                stdout=stdout,
                                stderr=stderr,
                                status="failed",
                                reason="parser_failed",
                                elapsed=elapsed,
                            )
                            return _failed(
                                task,
                                "ngspice produced no waveform data",
                                "parser_failed",
                                started,
                                extra_metadata=artifact_metadata,
                            )
                        elapsed = time.time() - started
                        artifact_metadata = _persist_shared_artifacts(
                            task,
                            plan,
                            netlist_text=netlist_text,
                            tmp_path=tmp_path,
                            output_path=output_path,
                            stdout=stdout,
                            stderr=stderr,
                            status="ok",
                            reason=None,
                            elapsed=elapsed,
                        )
                        return SimResult(
                            status="ok",
                            waveforms=waveforms,
                            sweep_var=sweep_var,
                            corner=task.corner,
                            metadata=_result_metadata(
                                task,
                                plan,
                                elapsed,
                                {**mutation_projection.metadata, **extra_metadata, **artifact_metadata},
                            ),
                            analysis_result=analysis_result,
                            measures=measures,
                        )
        except NgspiceSharedLibraryError as exc:
            return _failed(
                task,
                str(exc),
                "simulator_missing",
                started,
                extra_metadata={"library": self.library or "ngspice"},
            )
        except Exception as exc:
            if _raise_backend_exceptions(task):
                raise
            return _failed(task, str(exc), "backend_error", started, extra_metadata=_exception_metadata(exc))

    def _prepare_task(self, task: SimTask):
        return _prepare_runner_task(
            task,
            invalid_circuit_message="shared ngspice execution requires a monata.netlist.Circuit",
            no_outputs_message="ngspice shared execution requires explicit output_names",
        )


def _with_task_directives(netlist_text: str, task: SimTask, param_overrides: dict | None = None) -> str:
    lines = _netlist_lines_with_task_directives(netlist_text, task, param_overrides=param_overrides)
    lines.append(".end")
    return "\n".join(lines) + "\n"


def _persist_shared_artifacts(
    task: SimTask,
    plan: NgspiceTaskPlan,
    *,
    netlist_text: str,
    tmp_path: Path,
    output_path: Path,
    stdout: str | None,
    stderr: str | None,
    status: str,
    reason: str | None,
    elapsed: float,
) -> dict:
    netlist_path = tmp_path / "circuit.cir"
    netlist_path.write_text(netlist_text, encoding="utf-8")
    return _persist_runner_artifacts(
        task,
        simulator=NgspiceSharedRunner.name,
        plan=plan,
        netlist_path=netlist_path,
        output_path=output_path,
        stdout=stdout,
        stderr=stderr,
        status=status,
        reason=reason,
        elapsed=elapsed,
    )


def _execute_plan(session: Any, plan: NgspiceTaskPlan, output_path: Path) -> str:
    outputs: list[str] = []
    for command in _plan_command_lines(
        plan,
        output_path,
        path_formatter=_control_path,
        analysis_command_formatter=_shared_analysis_command,
    ):
        outputs.append(session.command(command))
    return "\n".join(output for output in outputs if output)


def _shared_analysis_command(command: str) -> str:
    lowered = command.lower()
    analysis_prefixes = ("dc ", "tran ", "ac ", "op", "noise ", "sens ", "pz ", "disto ", "tf ", "fourier ")
    if lowered == "op" or lowered.startswith(analysis_prefixes):
        return lowered
    return command


def _result_metadata(task: SimTask, plan: NgspiceTaskPlan, elapsed: float, extra_metadata: dict | None = None) -> dict:
    return _common_result_metadata(task, plan, NgspiceSharedRunner.name, elapsed, extra_metadata)


def _failed(
    task: SimTask,
    message: str,
    reason: str,
    started: float,
    extra_metadata: dict | None = None,
) -> SimResult:
    return _common_failure_result(
        task,
        message,
        simulator=NgspiceSharedRunner.name,
        reason=reason,
        elapsed=time.time() - started,
        extra_metadata=extra_metadata,
    )


__all__ = [
    "NgspiceCallbackEvent",
    "NgspiceInitData",
    "NgspiceInitVector",
    "NgspiceSharedCommandError",
    "NgspiceSharedCallbacks",
    "NgspiceSharedError",
    "NgspiceSharedLibraryError",
    "NgspiceSharedRunner",
    "NgspiceSharedSession",
]
