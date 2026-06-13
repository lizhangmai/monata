"""Subprocess ngspice backend for native Monata netlists."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from monata.netlist import render_ngspice
from monata.sim.backends.ngspice_stdout import noise_aborted as _noise_aborted, parse_measure_print
from monata.sim.backends.ngspice_parse import (
    parse_output as _parse_output,
)
from monata.sim.backends.base import BackendCapabilities
from monata.sim.backends.ngspice_common import (
    NGSPICE_ANALYSES as _NGSPICE_ANALYSES,
    NGSPICE_MODEL_ARTIFACTS as _NGSPICE_MODEL_ARTIFACTS,
    NGSPICE_PARSER_CONTRACT as _NGSPICE_PARSER_CONTRACT,
    NGSPICE_RESULT_MODES as _NGSPICE_RESULT_MODES,
    NGSPICE_SOURCE_MUTATIONS as _NGSPICE_SOURCE_MUTATIONS,
    NgspicePreflightFailure as _NgspicePreflightFailure,
    backend_task_plan as _backend_task_plan,
    bounded_process_text as _bounded_process_text,
    bounded_text as _bounded_text,
    exception_metadata as _exception_metadata,
    failure_result as _common_failure_result,
    netlist_lines_with_task_directives as _netlist_lines_with_task_directives,
    plan_command_lines as _plan_command_lines,
    persist_runner_artifacts as _persist_runner_artifacts,
    prepare_runner_task as _prepare_runner_task,
    raise_backend_exceptions as _raise_backend_exceptions,
    result_metadata as _common_result_metadata,
    task_measure_specs as _task_measure_specs,
)
from monata.sim.backends.base import BackendTaskPlan
from monata.sim.backends.ngspice_plan import NgspiceTaskPlan
from monata.sim.results import SimResult
from monata.sim.task import SimTask


class NgspiceRunner:
    """Run Monata-generated circuits through the ngspice executable."""

    name = "ngspice-subprocess"
    capabilities = BackendCapabilities(
        analyses=_NGSPICE_ANALYSES,
        result_modes=_NGSPICE_RESULT_MODES,
        source_mutations=_NGSPICE_SOURCE_MUTATIONS,
        model_artifacts=_NGSPICE_MODEL_ARTIFACTS,
        native_monte_carlo=False,
        renderer_contract="ngspice-netlist",
        runner_contract="subprocess",
        parser_contract=_NGSPICE_PARSER_CONTRACT,
    )

    def __init__(self, executable: str = "ngspice") -> None:
        self.executable = executable

    def validate_task(self, task: SimTask) -> SimResult | None:
        if self._resolve_executable() is None:
            return _failed(
                task,
                "ngspice executable not found",
                reason="simulator_missing",
                elapsed=0.0,
            )
        prepared = self._prepare_task(task)
        if isinstance(prepared, _NgspicePreflightFailure):
            return _failed(
                task,
                prepared.message,
                reason=prepared.reason,
                elapsed=0.0,
            )
        return None

    def plan_task(self, task: SimTask) -> BackendTaskPlan:
        prepared = self._prepare_task(task)
        if isinstance(prepared, _NgspicePreflightFailure):
            raise ValueError(prepared.message)
        return _backend_task_plan(self.name, prepared.plan)

    def run(self, task: SimTask) -> SimResult:
        started = time.time()
        executable = self._resolve_executable()
        if executable is None:
            return _failed(
                task,
                "ngspice executable not found",
                reason="simulator_missing",
                elapsed=time.time() - started,
            )

        prepared = self._prepare_task(task)
        if isinstance(prepared, _NgspicePreflightFailure):
            return _failed(
                task,
                prepared.message,
                reason=prepared.reason,
                elapsed=time.time() - started,
            )
        plan = prepared.plan
        mutation_projection = prepared.mutation_projection

        try:
            with tempfile.TemporaryDirectory(prefix="monata-ngspice-") as tmp:
                tmp_path = Path(tmp)
                output_path = tmp_path / ("result.raw" if plan.extraction == "rawfile" else "result.dat")
                netlist_text = _with_control_block(
                    render_ngspice(mutation_projection.circuit),
                    plan,
                    task,
                    output_path,
                    param_overrides=mutation_projection.param_overrides,
                )
                netlist_path = tmp_path / "circuit.cir"
                netlist_path.write_text(netlist_text)
                command = [executable, "-b", str(netlist_path)]

                try:
                    proc = subprocess.run(
                        command,
                        cwd=tmp_path,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=task.timeout,
                    )
                except subprocess.TimeoutExpired as exc:
                    elapsed = time.time() - started
                    timeout = task.timeout
                    message = (
                        f"ngspice subprocess timed out after {timeout:g} seconds"
                        if timeout is not None
                        else "ngspice subprocess timed out"
                    )
                    artifact_metadata = _persist_ngspice_artifacts(
                        task,
                        plan,
                        netlist_path=netlist_path,
                        output_path=output_path,
                        command=command,
                        stdout=_bounded_process_text(getattr(exc, "stdout", None) or getattr(exc, "output", None)),
                        stderr=_bounded_process_text(getattr(exc, "stderr", None)),
                        returncode=None,
                        status="failed",
                        reason="timeout",
                        elapsed=elapsed,
                    )
                    return _failed(
                        task,
                        message,
                        reason="timeout",
                        elapsed=elapsed,
                        extra_metadata={
                            "timeout_seconds": timeout,
                            "ngspice_stdout": _bounded_process_text(
                                getattr(exc, "stdout", None) or getattr(exc, "output", None)
                            ),
                            "ngspice_stderr": _bounded_process_text(getattr(exc, "stderr", None)),
                            **artifact_metadata,
                        },
                    )
                if proc.returncode != 0:
                    elapsed = time.time() - started
                    artifact_metadata = _persist_ngspice_artifacts(
                        task,
                        plan,
                        netlist_path=netlist_path,
                        output_path=output_path,
                        command=command,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        returncode=proc.returncode,
                        status="failed",
                        reason="subprocess_failed",
                        elapsed=elapsed,
                    )
                    return _failed(
                        task,
                        (proc.stderr or proc.stdout or "ngspice subprocess failed").strip(),
                        reason="subprocess_failed",
                        elapsed=elapsed,
                        extra_metadata=artifact_metadata,
                    )

                if plan.analysis_name == "noise" and _noise_aborted(proc.stdout, proc.stderr):
                    elapsed = time.time() - started
                    artifact_metadata = _persist_ngspice_artifacts(
                        task,
                        plan,
                        netlist_path=netlist_path,
                        output_path=output_path,
                        command=command,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        returncode=proc.returncode,
                        status="failed",
                        reason="subprocess_failed",
                        elapsed=elapsed,
                    )
                    return _failed(
                        task,
                        (proc.stderr or proc.stdout or "ngspice noise simulation aborted").strip(),
                        reason="subprocess_failed",
                        elapsed=elapsed,
                        extra_metadata=artifact_metadata,
                    )
                try:
                    sweep_var, waveforms, analysis_result, extra_metadata = _parse_output(output_path, proc.stdout, plan)
                except Exception as exc:
                    if _raise_backend_exceptions(task):
                        raise
                    elapsed = time.time() - started
                    artifact_metadata = _persist_ngspice_artifacts(
                        task,
                        plan,
                        netlist_path=netlist_path,
                        output_path=output_path,
                        command=command,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        returncode=proc.returncode,
                        status="failed",
                        reason="parser_failed",
                        elapsed=elapsed,
                    )
                    return _failed(
                        task,
                        str(exc),
                        reason="parser_failed",
                        elapsed=elapsed,
                        extra_metadata={
                            "ngspice_stdout": _bounded_text(proc.stdout),
                            "ngspice_stderr": _bounded_text(proc.stderr),
                            **artifact_metadata,
                        },
                    )
                measures = parse_measure_print(proc.stdout, _task_measure_specs(task))
                if measures:
                    extra_metadata = {
                        **extra_metadata,
                        "measures": measures.to_dict(),
                    }
                if not waveforms:
                    elapsed = time.time() - started
                    artifact_metadata = _persist_ngspice_artifacts(
                        task,
                        plan,
                        netlist_path=netlist_path,
                        output_path=output_path,
                        command=command,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        returncode=proc.returncode,
                        status="failed",
                        reason="parser_failed",
                        elapsed=elapsed,
                    )
                    return _failed(
                        task,
                        "ngspice produced no waveform data",
                        reason="parser_failed",
                        elapsed=elapsed,
                        extra_metadata=artifact_metadata,
                    )

                elapsed = time.time() - started
                artifact_metadata = _persist_ngspice_artifacts(
                    task,
                    plan,
                    netlist_path=netlist_path,
                    output_path=output_path,
                    command=command,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
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
        except subprocess.TimeoutExpired as exc:
            timeout = task.timeout
            message = (
                f"ngspice subprocess timed out after {timeout:g} seconds"
                if timeout is not None
                else "ngspice subprocess timed out"
            )
            return _failed(
                task,
                message,
                reason="timeout",
                elapsed=time.time() - started,
                extra_metadata={
                    "timeout_seconds": timeout,
                    "ngspice_stdout": _bounded_process_text(getattr(exc, "stdout", None) or getattr(exc, "output", None)),
                    "ngspice_stderr": _bounded_process_text(getattr(exc, "stderr", None)),
                },
            )
        except Exception as exc:
            if _raise_backend_exceptions(task):
                raise
            return _failed(
                task,
                str(exc),
                reason="backend_error",
                elapsed=time.time() - started,
                extra_metadata=_exception_metadata(exc),
            )

    def _resolve_executable(self) -> str | None:
        direct = shutil.which(self.executable)
        if direct:
            return direct
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return None
        conda_ngspice = Path(conda_prefix) / "bin" / self.executable
        if conda_ngspice.is_file():
            return str(conda_ngspice)
        return None

    def _prepare_task(self, task: SimTask):
        return _prepare_runner_task(
            task,
            invalid_circuit_message="native ngspice execution requires a monata.netlist.Circuit",
            no_outputs_message="ngspice execution requires explicit output_names",
        )


def _with_control_block(
    netlist_text: str,
    plan: NgspiceTaskPlan,
    task: SimTask,
    output_path: Path,
    param_overrides: dict | None = None,
) -> str:
    lines = _netlist_lines_with_task_directives(netlist_text, task, param_overrides=param_overrides)
    lines.extend(_control_block(plan, output_path))
    lines.append(".end")
    return "\n".join(lines) + "\n"


def _control_block(plan: NgspiceTaskPlan, output_path: Path) -> list[str]:
    return [".control", *_plan_command_lines(plan, output_path), "quit", ".endc"]


def _persist_ngspice_artifacts(
    task: SimTask,
    plan: NgspiceTaskPlan,
    *,
    netlist_path: Path,
    output_path: Path,
    command: list[str],
    stdout: str | None,
    stderr: str | None,
    returncode: int | None,
    status: str,
    reason: str | None,
    elapsed: float,
) -> dict:
    return _persist_runner_artifacts(
        task,
        simulator=NgspiceRunner.name,
        plan=plan,
        netlist_path=netlist_path,
        output_path=output_path,
        stdout=stdout,
        stderr=stderr,
        status=status,
        reason=reason,
        elapsed=elapsed,
        metadata={
            "command": command,
            "returncode": returncode,
        },
    )


def _result_metadata(task: SimTask, plan: NgspiceTaskPlan, elapsed: float, extra_metadata: dict | None = None) -> dict:
    return _common_result_metadata(task, plan, NgspiceRunner.name, elapsed, extra_metadata)


def _failed(
    task: SimTask,
    message: str,
    reason: str,
    elapsed: float,
    extra_metadata: dict | None = None,
) -> SimResult:
    return _common_failure_result(
        task,
        message,
        simulator=NgspiceRunner.name,
        reason=reason,
        elapsed=elapsed,
        extra_metadata=extra_metadata,
    )
