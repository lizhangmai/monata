"""Shared helpers for ngspice backend runners."""

from __future__ import annotations

from collections.abc import Callable
import re
import traceback
from dataclasses import dataclass
from pathlib import Path

from monata.corner import validate_model_section
from monata.netlist import Circuit, MutationError, MutationProjection, project_param_overrides
from monata.sim.analysis_spec import (
    ACSpec,
    DCSpec,
    DistortionSpec,
    FourierSpec,
    NoiseSpec,
    OPSpec,
    PoleZeroSpec,
    SensitivitySpec,
    TranSpec,
    TransferFunctionSpec,
)
from monata.sim.artifacts import persist_simulation_artifacts
from monata.sim.backends.base import BackendTaskPlan, backend_failure_result
from monata.sim.backends.ngspice_plan import NgspiceTaskPlan, task_plan, validate_scalar_value
from monata.sim.results import SimResult
from monata.sim.task import SimTask


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NGSPICE_ANALYSES = (
    "ac",
    "dc",
    "tran",
    "op",
    "noise",
    "sensitivity",
    "pole-zero",
    "distortion",
    "transfer-function",
    "fourier",
)
NGSPICE_ANALYSIS_SPEC_TYPES = (
    ACSpec,
    DCSpec,
    DistortionSpec,
    FourierSpec,
    NoiseSpec,
    OPSpec,
    PoleZeroSpec,
    SensitivitySpec,
    TranSpec,
    TransferFunctionSpec,
)
NGSPICE_RESULT_MODES = ("rawfile", "stdout-print")
NGSPICE_SOURCE_MUTATIONS = (
    "output_names",
    "param_overrides",
    "corner",
    "save",
    "measure",
)
NGSPICE_MODEL_ARTIFACTS = ("include", "lib", "osdi")
NGSPICE_PARSER_CONTRACT = "ngspice-rawfile/wrdata-fallback/stdout"
_DOUBLE_QUOTE = '"'
_SINGLE_QUOTE = "'"


@dataclass(frozen=True)
class NgspicePreflightFailure:
    message: str
    reason: str


@dataclass(frozen=True)
class NgspicePreparedTask:
    plan: NgspiceTaskPlan
    mutation_projection: MutationProjection


def print_lines(vectors: tuple[str, ...], chunk_size: int = 64) -> list[str]:
    return [
        f"print {' '.join(vectors[index:index + chunk_size])}"
        for index in range(0, len(vectors), chunk_size)
    ]


def fallback_wrdata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".dat")


def ngspice_path(path: Path) -> str:
    """Return a double-quoted ngspice netlist token for a filesystem path."""

    return f"{_DOUBLE_QUOTE}{_safe_path_text(path, label='ngspice path', quote_char=_DOUBLE_QUOTE)}{_DOUBLE_QUOTE}"


def ngspice_control_path(path: Path) -> str:
    """Return a single-quoted ngspice control-command token for a filesystem path."""

    return f"{_SINGLE_QUOTE}{_safe_path_text(path, label='ngspice control path', quote_char=_SINGLE_QUOTE)}{_SINGLE_QUOTE}"


def plan_command_lines(
    plan: NgspiceTaskPlan,
    output_path: Path,
    *,
    path_formatter: Callable[[Path], str] = ngspice_control_path,
    analysis_command_formatter: Callable[[str], str] | None = None,
) -> list[str]:
    lines = ["set wr_singlescale"]
    lines.extend(f"pre_osdi {path_formatter(path)}" for path in plan.osdi_paths)
    for command in str(plan.command).splitlines():
        stripped = command.strip()
        if not stripped:
            continue
        if analysis_command_formatter is not None:
            stripped = analysis_command_formatter(stripped)
        lines.append(stripped)
    if plan.extraction == "rawfile":
        rawfile_path = path_formatter(output_path)
        fallback_path = path_formatter(fallback_wrdata_path(output_path))
        lines.append(f"set filetype={plan.metadata.get('rawfile_format', 'ascii')}")
        if plan.metadata.get("write_all_vectors"):
            lines.append(f"write {rawfile_path} all")
        else:
            vectors = " ".join(plan.output_vectors)
            lines.append(f"write {rawfile_path} {vectors}")
            lines.append(f"wrdata {fallback_path} {vectors}")
    elif plan.extraction == "stdout-print":
        if plan.analysis_name == "op":
            lines.extend(print_lines(plan.output_vectors))
        elif plan.analysis_name == "noise":
            lines.extend([
                "setplot noise1",
                "print frequency onoise_spectrum inoise_spectrum",
                "setplot noise2",
                "print onoise_total inoise_total",
            ])
        elif plan.analysis_name == "tf":
            lines.append("print all")
    else:
        raise ValueError(f"unsupported ngspice extraction mode: {plan.extraction}")
    return lines


def persist_runner_artifacts(
    task: SimTask,
    plan: NgspiceTaskPlan,
    *,
    simulator: str,
    netlist_path: Path,
    output_path: Path,
    stdout: str | None,
    stderr: str | None,
    status: str,
    reason: str | None,
    elapsed: float,
    metadata: dict | None = None,
) -> dict:
    return persist_simulation_artifacts(
        task,
        simulator=simulator,
        files=artifact_output_files(plan, netlist_path=netlist_path, output_path=output_path),
        text_files={"stdout": stdout, "stderr": stderr},
        metadata=artifact_metadata(
            task,
            plan,
            status=status,
            reason=reason,
            elapsed=elapsed,
            extra_metadata=metadata,
        ),
    )


def artifact_output_files(
    plan: NgspiceTaskPlan,
    *,
    netlist_path: Path,
    output_path: Path,
) -> dict[str, Path | None]:
    output_files: dict[str, Path | None] = {"netlist": netlist_path}
    if plan.extraction == "rawfile":
        output_files["rawfile"] = output_path
        output_files["wrdata"] = fallback_wrdata_path(output_path)
    elif plan.extraction != "stdout-print":
        raise ValueError(f"unsupported ngspice extraction mode: {plan.extraction}")
    return output_files


def artifact_metadata(
    task: SimTask,
    plan: NgspiceTaskPlan,
    *,
    status: str,
    reason: str | None,
    elapsed: float,
    extra_metadata: dict | None = None,
) -> dict:
    return {
        "analysis": plan.analysis_name,
        "extraction": plan.extraction,
        **(extra_metadata or {}),
        "status": status,
        "reason": reason,
        "elapsed_time": elapsed,
        "outputs": list(plan.output_names),
        "output_vectors": list(plan.output_vectors),
        "osdi_paths": [str(path) for path in plan.osdi_paths],
        "plan": plan.metadata,
        "task_metadata": dict(task.metadata),
    }


def missing_osdi_path(plan: NgspiceTaskPlan) -> Path | None:
    for path in plan.osdi_paths:
        if not path.is_file():
            return path
    return None


def result_metadata(
    task: SimTask,
    plan: NgspiceTaskPlan,
    simulator: str,
    elapsed: float,
    extra_metadata: dict | None = None,
) -> dict:
    return attach_backend_metadata(
        task,
        success_backend_metadata(plan, simulator, elapsed, extra_metadata),
        top_level_keys=set(plan.metadata) | _SUCCESS_TOP_LEVEL_EXTRA_KEYS,
    )


def success_backend_metadata(
    plan: NgspiceTaskPlan,
    simulator: str,
    elapsed: float,
    extra_metadata: dict | None = None,
) -> dict:
    return {
        **plan.metadata,
        **(extra_metadata or {}),
        "simulator": simulator,
        "elapsed_time": elapsed,
        "outputs": list(plan.output_names),
    }


def failure_metadata(
    task: SimTask,
    simulator: str,
    reason: str,
    elapsed: float,
    extra_metadata: dict | None = None,
) -> dict:
    return attach_backend_metadata(
        task,
        failure_backend_metadata(simulator, reason, elapsed, extra_metadata),
        top_level_keys=_FAILURE_TOP_LEVEL_KEYS,
    )


def failure_result(
    task: SimTask,
    message: str,
    *,
    simulator: str,
    reason: str,
    elapsed: float,
    extra_metadata: dict | None = None,
) -> SimResult:
    return backend_failure_result(
        task,
        message,
        metadata=failure_metadata(task, simulator, reason, elapsed, extra_metadata),
    )


def raise_backend_exceptions(task: SimTask) -> bool:
    """Return whether backend adapter exceptions should escape failed-result wrapping."""

    value = task.backend_options.get("raise_backend_exceptions", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def failure_backend_metadata(
    simulator: str,
    reason: str,
    elapsed: float,
    extra_metadata: dict | None = None,
) -> dict:
    return {
        "simulator": simulator,
        "elapsed_time": elapsed,
        "reason": reason,
        **(extra_metadata or {}),
    }


def attach_backend_metadata(task: SimTask, backend_metadata: dict, *, top_level_keys: set[str] | frozenset[str]) -> dict:
    metadata = dict(task.metadata)
    metadata[metadata_key(metadata, "ngspice")] = backend_metadata
    for key in top_level_keys:
        if key in backend_metadata:
            metadata.setdefault(key, backend_metadata[key])
    return metadata


def metadata_key(metadata: dict, name: str) -> str:
    if name not in metadata:
        return name
    index = 2
    while f"{name}_{index}" in metadata:
        index += 1
    return f"{name}_{index}"


_SUCCESS_TOP_LEVEL_EXTRA_KEYS = frozenset({
    "artifacts",
    "elapsed_time",
    "fourier_frequency",
    "fourier_grid_size",
    "fourier_harmonic_count",
    "fourier_interpolation_degree",
    "fourier_output_vector",
    "fourier_period_count",
    "fourier_thd_percent",
    "measures",
    "noise_totals",
    "outputs",
    "simulator",
    "structured_mutations",
    "vector_kinds",
    "vector_metadata",
    "vector_quantities",
    "vector_raw_names",
    "vector_units",
})

_FAILURE_TOP_LEVEL_KEYS = frozenset({
    "artifacts",
    "elapsed_time",
    "ngspice_stderr",
    "ngspice_stdout",
    "reason",
    "simulator",
    "timeout_seconds",
})


def missing_corner_model_path(task: SimTask) -> Path | None:
    corner = task.corner
    model_file = getattr(corner, "model_file", None)
    if model_file:
        path = Path(model_file)
        if not path.is_file():
            return path
    return None


def unsupported_param_overrides(param_overrides: dict) -> str | None:
    for name, value in param_overrides.items():
        if not _IDENTIFIER_RE.match(str(name)):
            return f"unsupported parameter override: {name}"
        validate_scalar_value(value, f"parameter override {name}")
    return None


def unsupported_corner(task: SimTask) -> str | None:
    corner = task.corner
    if corner is None:
        return None
    process = getattr(corner, "process", None)
    model_file = getattr(corner, "model_file", None)
    model_deck = getattr(corner, "model_deck", None)
    if process and not model_file:
        return "corner process requires a model_file to affect ngspice execution"
    if model_deck and not model_file:
        return "corner model_deck requires a model_file to affect ngspice execution"
    return None


def task_mutation_overrides(task: SimTask) -> dict:
    overrides = dict(task.param_overrides)
    corner = task.corner
    voltages = getattr(corner, "voltages", None) if corner is not None else None
    if voltages:
        for source, value in voltages.items():
            target = str(source) if "." in str(source) else f"{source}.V"
            if target in overrides and overrides[target] != value:
                raise MutationError(f"conflicting override for corner voltage target: {target}")
            overrides[target] = value
    return overrides


def prepare_runner_task(
    task: SimTask,
    *,
    invalid_circuit_message: str,
    no_outputs_message: str,
) -> NgspicePreparedTask | NgspicePreflightFailure:
    if not isinstance(task.analysis_spec, NGSPICE_ANALYSIS_SPEC_TYPES):
        return NgspicePreflightFailure(
            f"unknown analysis spec: {type(task.analysis_spec).__name__}",
            "unsupported_analysis",
        )
    if not isinstance(task.circuit, Circuit):
        return NgspicePreflightFailure(invalid_circuit_message, "invalid_circuit")
    try:
        plan = task_plan(task)
    except ValueError as exc:
        return NgspicePreflightFailure(str(exc), "invalid_task")
    if plan is None:
        return NgspicePreflightFailure(no_outputs_message, "no_outputs_requested")
    try:
        mutation_projection = project_param_overrides(task.circuit, task_mutation_overrides(task))
    except MutationError as exc:
        return NgspicePreflightFailure(str(exc), "unsupported_param_overrides")
    unsupported_params = unsupported_param_overrides(mutation_projection.param_overrides)
    if unsupported_params:
        return NgspicePreflightFailure(unsupported_params, "unsupported_param_overrides")
    unsupported_corner_message = unsupported_corner(task)
    if unsupported_corner_message:
        return NgspicePreflightFailure(unsupported_corner_message, "unsupported_corner")
    missing_osdi = missing_osdi_path(plan)
    if missing_osdi is not None:
        return NgspicePreflightFailure(f"OSDI model file not found: {missing_osdi}", "model_missing")
    missing_corner_model = missing_corner_model_path(task)
    if missing_corner_model is not None:
        return NgspicePreflightFailure(f"corner model file not found: {missing_corner_model}", "model_missing")
    return NgspicePreparedTask(plan, mutation_projection)


def backend_task_plan(backend_name: str, plan: NgspiceTaskPlan) -> BackendTaskPlan:
    return BackendTaskPlan(
        backend_name=backend_name,
        analysis_name=plan.analysis_name,
        output_names=plan.output_names,
        output_vectors=plan.output_vectors,
        metadata=plan.metadata,
    )


def task_directives(
    task: SimTask,
    netlist_text: str = "",
    param_overrides: dict | None = None,
) -> list[str]:
    lines = []
    corner = task.corner
    if corner is not None:
        temperature = getattr(corner, "temperature", None)
        if temperature is not None:
            validate_scalar_value(temperature, "corner temperature")
            lines.append(f".temp {temperature}")
        model_file = getattr(corner, "model_file", None)
        if model_file and not corner_model_directive_present(
            netlist_text,
            Path(model_file),
            getattr(corner, "section", None),
        ):
            section = getattr(corner, "section", None)
            if section:
                lines.append(f".lib {ngspice_path(Path(model_file))} {validate_model_section(section)}")
            else:
                lines.append(f".include {ngspice_path(Path(model_file))}")
    for name, value in (param_overrides if param_overrides is not None else task.param_overrides).items():
        lines.append(f".param {name}={value}")
    return lines


def netlist_lines_with_task_directives(
    netlist_text: str,
    task: SimTask,
    *,
    param_overrides: dict | None = None,
) -> list[str]:
    lines = netlist_text.rstrip().splitlines()
    if lines and lines[-1].lower() == ".end":
        lines.pop()
    lines.extend(task_directives(task, netlist_text=netlist_text, param_overrides=param_overrides))
    return lines


def task_measure_specs(task: SimTask) -> dict[str, str]:
    measures: dict[str, str] = {}
    for directive in getattr(task.circuit, "directives", ()):
        if getattr(directive, "raw", False):
            continue
        if getattr(directive, "name", None) != "measure":
            continue
        args = getattr(directive, "args", ())
        if len(args) >= 2:
            measures[str(args[1])] = str(args[0]).lower()
    return measures


def include_path(path: Path) -> str:
    return _safe_path_text(path, label="include path", quote_char=_DOUBLE_QUOTE)


def _safe_path_text(path: Path, *, label: str, quote_char: str) -> str:
    text = str(path)
    if any(char in text for char in "\r\n;") or quote_char in text:
        raise ValueError(f"invalid {label}: {path}")
    return text


def corner_model_directive_present(
    netlist_text: str,
    model_file: Path,
    section: str | None,
) -> bool:
    path_text = include_path(model_file)
    quoted_path = ngspice_path(model_file)
    expected = (
        {f".lib {quoted_path} {section}", f".lib {path_text} {section}"}
        if section
        else {f".include {quoted_path}", f'.include "{path_text}"'}
    )
    return any(line.strip() in expected for line in netlist_text.splitlines())


def bounded_text(value: str | None, limit: int = 4000) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def bounded_process_text(value: str | bytes | None, limit: int = 4000) -> str:
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = value or ""
    return bounded_text(text, limit=limit)


def exception_metadata(exc: BaseException) -> dict[str, str]:
    return {
        "exception_type": type(exc).__name__,
        "traceback": bounded_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))),
    }
