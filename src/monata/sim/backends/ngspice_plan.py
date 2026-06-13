"""Task planning helpers for the subprocess ngspice backend."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from monata.sim.task import SimTask
from monata.sim._vector_identity import simple_vector_inner
from monata.sim.vector_names import VectorName, normalize_vector_name


_NODE_RE = re.compile(r"^[A-Za-z0-9_.$:+-]+$")
NgspiceExtraction = Literal["rawfile", "stdout-print"]


@dataclass(frozen=True)
class NgspiceOutputRequest:
    output_name: str
    vector: str
    identity: VectorName
    transform: str | None = None


@dataclass(frozen=True)
class NgspiceTaskPlan:
    analysis_name: str
    output_names: tuple[str, ...]
    output_vectors: tuple[str, ...]
    output_requests: tuple[NgspiceOutputRequest, ...]
    command: str
    osdi_paths: tuple[Path, ...]
    metadata: dict
    extraction: NgspiceExtraction = "rawfile"


@dataclass(frozen=True)
class _PlannedAnalysis:
    analysis_name: str
    output_names: tuple[str, ...]
    command: str
    write_all_vectors: bool = False


_NgspiceAnalysisPlanner = Callable[[Any, tuple[str, ...]], _PlannedAnalysis]


def task_plan(task: SimTask) -> NgspiceTaskPlan | None:
    for path in task.osdi_paths:
        control_path(path)
    planned = _plan_analysis(task.analysis_spec, tuple(task.output_names))
    if planned is None:
        return None

    analysis_name = planned.analysis_name
    output_names = planned.output_names
    output_requests = output_requests_for(output_names, analysis_name)
    output_vectors = tuple(request.vector for request in output_requests)
    stdout_fallback = analysis_name == "noise" or (analysis_name == "op" and not planned.write_all_vectors)
    extraction = "rawfile" if planned.write_all_vectors else (
        "stdout-print" if analysis_name in {"four", "noise", "op", "tf"} else "rawfile"
    )
    return NgspiceTaskPlan(
        analysis_name=analysis_name,
        output_names=output_names,
        output_vectors=output_vectors,
        output_requests=output_requests,
        command=planned.command,
        osdi_paths=tuple(task.osdi_paths),
        metadata={
            "analysis": analysis_name,
            "output_vectors": list(output_vectors),
            "output_requests": [output_request_metadata(request) for request in output_requests],
            "rawfile_format": rawfile_format(task),
            "osdi_paths": [str(path) for path in task.osdi_paths],
            "extraction_preference": "rawfile",
            "fallback_used": stdout_fallback,
            "write_all_vectors": planned.write_all_vectors or analysis_name in {"disto", "pz", "sens"},
            **typed_vector_metadata(analysis_name, output_names),
            **(
                {
                    "fallback_reason": "rawfile_not_enabled_for_analysis",
                }
                if stdout_fallback
                else {}
            ),
            **analysis_metadata(task.analysis_spec, analysis_name),
        },
        extraction=extraction,
    )


def _plan_analysis(spec, output_names: tuple[str, ...]) -> _PlannedAnalysis | None:
    for spec_type, planner in _ANALYSIS_PLANNERS:
        if isinstance(spec, spec_type):
            return planner(spec, output_names)
    if not output_names:
        return None
    for name in output_names:
        validate_output_request_name(name)
    raise TypeError(f"unsupported analysis spec: {type(spec).__name__}")


def _plan_dc(spec: DCSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names, write_all_vectors = _core_output_names(output_names)
    parts = dc_sweep_parts(spec.source, spec.start, spec.stop, spec.step, "dc")
    if spec.secondary is not None:
        parts = (*parts, *dc_sweep_parts(
            spec.secondary.source,
            spec.secondary.start,
            spec.secondary.stop,
            spec.secondary.step,
            "dc secondary",
        ))
    return _PlannedAnalysis("dc", output_names, "dc " + " ".join(parts), write_all_vectors)


def _plan_tran(spec: TranSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names, write_all_vectors = _core_output_names(output_names)
    step = spec.step if spec.step else spec.stop / 1000
    validate_numeric(step, "tran step")
    validate_numeric(spec.stop, "tran stop")
    validate_numeric(spec.start, "tran start")
    command = f"tran {step} {spec.stop} {spec.start}"
    if spec.max_step is not None:
        validate_numeric(spec.max_step, "tran max_step")
        command = f"{command} {spec.max_step}"
    if getattr(spec, "uic", False):
        command = f"{command} uic"
    return _PlannedAnalysis("tran", output_names, command, write_all_vectors)


def _plan_ac(spec: ACSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names, write_all_vectors = _core_output_names(output_names, validator=validate_ac_output_name)
    variation = str(spec.variation).lower()
    if variation not in {"dec", "oct", "lin"}:
        raise ValueError(f"invalid AC variation: {spec.variation}")
    validate_numeric(spec.points, "ac points")
    validate_numeric(spec.start, "ac start")
    validate_numeric(spec.stop, "ac stop")
    command = f"ac {variation} {spec.points} {spec.start} {spec.stop}"
    return _PlannedAnalysis("ac", output_names, command, write_all_vectors)


def _plan_op(output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names, write_all_vectors = _core_output_names(output_names)
    return _PlannedAnalysis("op", output_names, "op", write_all_vectors)


def _plan_op_spec(_spec: OPSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    return _plan_op(output_names)


def _plan_noise(spec: NoiseSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    if output_names:
        raise ValueError("NoiseSpec uses output_node/input_source; task output_names must be empty")
    validate_node_name(str(spec.output_node))
    validate_node_name(str(spec.reference_node))
    source = source_name(str(spec.input_source))
    validate_source_name(source)
    validate_numeric(spec.points, "noise points")
    validate_numeric(spec.start, "noise start")
    validate_numeric(spec.stop, "noise stop")
    variation = str(spec.variation).lower()
    if variation not in {"dec", "oct", "lin"}:
        raise ValueError(f"invalid noise variation: {spec.variation}")
    output = (
        f"v({spec.output_node})"
        if str(spec.reference_node) == "0"
        else f"v({spec.output_node},{spec.reference_node})"
    )
    summary = "" if spec.points_per_summary is None else f" {spec.points_per_summary}"
    command = f"noise {output} {source} {variation} {spec.points} {spec.start} {spec.stop}{summary}"
    return _PlannedAnalysis("noise", ("onoise_spectrum", "inoise_spectrum"), command)


def _plan_sensitivity(spec: SensitivitySpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names = _validate_or_default_output_names(output_names, ("sensitivity",))
    output = analysis_output_vector(spec.output)
    if spec.start is None and spec.stop is None and spec.points is None:
        command = f"sens {output}"
    elif spec.start is not None and spec.stop is not None and spec.points is not None:
        variation = str(spec.variation).lower()
        if variation not in {"dec", "oct", "lin"}:
            raise ValueError(f"invalid sensitivity variation: {spec.variation}")
        validate_numeric(spec.points, "sensitivity points")
        validate_numeric(spec.start, "sensitivity start")
        validate_numeric(spec.stop, "sensitivity stop")
        command = f"sens {output} ac {variation} {spec.points} {spec.start} {spec.stop}"
    else:
        raise ValueError("SensitivitySpec AC sweep requires start, stop, and points together")
    return _PlannedAnalysis("sens", output_names, command)


def _plan_pole_zero(spec: PoleZeroSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names = _validate_or_default_output_names(output_names, ("pole_zero",))
    for value in (
        spec.input_pos,
        spec.input_neg,
        spec.output_pos,
        spec.output_neg,
    ):
        validate_node_name(str(value))
    transfer = str(spec.transfer).lower()
    mode = str(spec.mode).lower()
    if transfer not in {"vol", "cur"}:
        raise ValueError(f"invalid pole-zero transfer type: {spec.transfer}")
    if mode not in {"pol", "zer", "pz"}:
        raise ValueError(f"invalid pole-zero mode: {spec.mode}")
    command = (
        f"pz {spec.input_pos} {spec.input_neg} {spec.output_pos} {spec.output_neg} "
        f"{transfer} {mode}"
    )
    return _PlannedAnalysis("pz", output_names, command)


def _plan_distortion(spec: DistortionSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names = _validate_or_default_output_names(output_names, ("distortion",))
    variation = str(spec.variation).lower()
    if variation not in {"dec", "oct", "lin"}:
        raise ValueError(f"invalid distortion variation: {spec.variation}")
    validate_numeric(spec.points, "distortion points")
    validate_numeric(spec.start, "distortion start")
    validate_numeric(spec.stop, "distortion stop")
    f2overf1 = "" if spec.f2overf1 is None else f" {spec.f2overf1}"
    if spec.f2overf1 is not None:
        validate_numeric(spec.f2overf1, "distortion f2overf1")
    command = f"disto {variation} {spec.points} {spec.start} {spec.stop}{f2overf1}"
    return _PlannedAnalysis("disto", output_names, command)


def _plan_transfer_function(spec: TransferFunctionSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names = _validate_or_default_output_names(
        output_names,
        ("transfer_function", "input_resistance", "output_resistance"),
    )
    output = analysis_output_vector(spec.output)
    source = source_name(str(spec.input_source))
    validate_source_name(source)
    return _PlannedAnalysis("tf", output_names, f"tf {output} {source}")


def _plan_fourier(spec: FourierSpec, output_names: tuple[str, ...]) -> _PlannedAnalysis:
    output_names = _validate_or_default_output_names(output_names, ("fourier_magnitude", "fourier_phase"))
    output = analysis_output_vector(spec.output)
    validate_positive_numeric(spec.frequency, "fourier frequency")
    step = (1 / spec.frequency) / 100 if spec.step is None else spec.step
    validate_positive_numeric(step, "fourier tran step")
    validate_numeric(spec.stop, "fourier tran stop")
    validate_numeric(spec.start, "fourier tran start")
    command = f"tran {step} {spec.stop} {spec.start}\nfourier {spec.frequency} {output}"
    return _PlannedAnalysis("four", output_names, command)


_ANALYSIS_PLANNERS: tuple[tuple[type[Any], _NgspiceAnalysisPlanner], ...] = (
    (DCSpec, _plan_dc),
    (TranSpec, _plan_tran),
    (ACSpec, _plan_ac),
    (OPSpec, _plan_op_spec),
    (NoiseSpec, _plan_noise),
    (SensitivitySpec, _plan_sensitivity),
    (PoleZeroSpec, _plan_pole_zero),
    (DistortionSpec, _plan_distortion),
    (TransferFunctionSpec, _plan_transfer_function),
    (FourierSpec, _plan_fourier),
)


def _core_output_names(
    output_names: tuple[str, ...],
    *,
    validator: Callable[[str], None] | None = None,
) -> tuple[tuple[str, ...], bool]:
    if not output_names:
        return output_names, True
    validate = validate_output_request_name if validator is None else validator
    for name in output_names:
        validate(name)
    return output_names, False


def _validate_or_default_output_names(
    output_names: tuple[str, ...],
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if not output_names:
        return default
    for name in output_names:
        validate_output_request_name(name)
    return output_names


def rawfile_format(task: SimTask) -> str:
    rawfile_format_name = str(task.backend_options.get("rawfile_format", "ascii")).lower()
    if rawfile_format_name not in {"ascii", "binary"}:
        raise ValueError(f"invalid rawfile_format: {rawfile_format_name}")
    return rawfile_format_name


def output_requests_for(output_names: tuple[str, ...], analysis_name: str) -> tuple[NgspiceOutputRequest, ...]:
    requests: list[NgspiceOutputRequest] = []
    for name in output_names:
        if analysis_name == "ac":
            request = ac_output_request(name)
        elif analysis_name == "noise":
            vector = name
            identity = normalize_vector_name(name)
            request = NgspiceOutputRequest(output_name=name, vector=vector, identity=identity)
        elif analysis_name in {"disto", "four", "pz", "sens", "tf"}:
            vector = name
            identity = normalize_vector_name(name)
            request = NgspiceOutputRequest(output_name=name, vector=vector, identity=identity)
        else:
            request = direct_output_request(name)
        requests.append(request)
    return tuple(requests)


def direct_output_request(name: str) -> NgspiceOutputRequest:
    text = str(name)
    if is_explicit_output_vector(text) or is_expression_vector(text):
        validate_output_request_name(text)
        vector = text
        identity = normalize_vector_name(vector, display_name=text)
        return NgspiceOutputRequest(output_name=text, vector=vector, identity=identity)
    validate_node_name(text)
    vector = f"v({text})"
    identity = normalize_vector_name(vector, display_name=text)
    return NgspiceOutputRequest(output_name=text, vector=vector, identity=identity)


def ac_output_request(name: str) -> NgspiceOutputRequest:
    text = str(name)
    lowered = text.lower()
    for transform in ("phase", "real", "imag", "db", "mag"):
        prefix = f"{transform}("
        if lowered.startswith(prefix) and text.endswith(")"):
            inner = text[len(prefix) : -1]
            vector = ac_raw_vector(inner)
            identity = normalize_vector_name(vector, display_name=text)
            return NgspiceOutputRequest(output_name=text, vector=vector, identity=identity, transform=transform)
    if is_explicit_output_vector(text):
        vector = text
        validate_vector_token(vector)
        identity = normalize_vector_name(vector, display_name=text)
        return NgspiceOutputRequest(output_name=text, vector=vector, identity=identity)
    if is_expression_vector(text):
        validate_output_request_name(text)
        identity = normalize_vector_name(text, display_name=text)
        return NgspiceOutputRequest(output_name=text, vector=text, identity=identity)
    validate_node_name(text)
    vector = f"v({text})"
    identity = normalize_vector_name(f"v({text})", display_name=text)
    return NgspiceOutputRequest(output_name=text, vector=vector, identity=identity, transform="mag")


def ac_raw_vector(name: str) -> str:
    text = str(name)
    if is_explicit_output_vector(text):
        validate_vector_token(text)
        return text
    validate_node_name(text)
    return f"v({text})"


def analysis_output_vector(name: str) -> str:
    text = str(name)
    validate_ac_output_name(text)
    if text.lower().startswith(("v(", "i(")):
        validate_vector_token(text)
        return text
    validate_node_name(text)
    return f"v({text})"


def validate_ac_output_name(name: str) -> None:
    validate_output_request_name(name, label="AC output name")


def validate_output_request_name(name: str, *, label: str = "output request") -> None:
    text = str(name)
    if any(char in text for char in "\r\n;"):
        raise ValueError(f"invalid {label} for ngspice vector: {name}")
    if any(char.isspace() for char in text):
        raise ValueError(f"invalid {label} for ngspice vector: {name}")
    if text.count("(") != text.count(")"):
        raise ValueError(f"invalid {label} for ngspice vector: {name}")
    if text.count("[") != text.count("]"):
        raise ValueError(f"invalid {label} for ngspice vector: {name}")
    if is_explicit_output_vector(text):
        validate_vector_token(text)


def validate_vector_token(name: str) -> None:
    text = str(name)
    lower = text.lower()
    if lower.startswith("@"):
        if re.search(r"\s", text) or text.count("[") != text.count("]"):
            raise ValueError(f"invalid ngspice vector: {name}")
        return
    if lower.startswith(("v(", "i(")) and text.endswith(")"):
        inner = text[2:-1]
        if not inner or any(char.isspace() for char in inner) or any(char in inner for char in "()"):
            raise ValueError(f"invalid ngspice vector: {name}")
        return
    raise ValueError(f"invalid ngspice vector: {name}")


def is_explicit_output_vector(name: str) -> bool:
    text = str(name).strip()
    lower = text.lower()
    if lower.startswith("@"):
        return True
    return simple_vector_inner(text, "v") is not None or simple_vector_inner(text, "i") is not None


def is_expression_vector(name: str) -> bool:
    text = str(name).strip()
    if not text or text.startswith("@"):
        return False
    if is_explicit_output_vector(text):
        return False
    return "(" in text or any(operator in text for operator in ("*", "/"))


def output_request_metadata(request: NgspiceOutputRequest) -> dict:
    identity = request.identity
    return {
        "output_name": request.output_name,
        "vector": request.vector,
        "display_name": identity.display_name,
        "normalized_name": identity.normalized_name,
        "raw_vector_name": identity.raw_vector_name,
        "vector_kind": identity.vector_kind,
        "quantity": identity.quantity,
        "transform": request.transform,
    }


def typed_vector_metadata(analysis_name: str, output_names: tuple[str, ...]) -> dict:
    vector_kind = {
        "disto": "distortion",
        "four": "fourier_component",
        "pz": "pole",
        "sens": "sensitivity",
        "tf": "transfer_function",
    }.get(analysis_name)
    if vector_kind is None:
        return {}
    return {
        "vector_kinds": {name: vector_kind for name in output_names},
    }


def source_name(source: str) -> str:
    if source and source[0].upper() in {"V", "I"}:
        return source
    return f"V{source}"


def dc_source_name(source: str) -> str:
    text = str(source)
    if not text:
        return text
    if text.lower() == "temp" or text[0].upper() in {"V", "I", "R"}:
        return text
    return f"V{text}"


def dc_sweep_parts(source: str, start, stop, step, label: str) -> tuple[str, str, str, str]:
    sweep_source = dc_source_name(str(source))
    validate_source_name(sweep_source)
    validate_numeric(start, f"{label} start")
    validate_numeric(stop, f"{label} stop")
    validate_numeric(step, f"{label} step")
    return (sweep_source, str(start), str(stop), str(step))


def analysis_metadata(spec, analysis_name: str) -> dict:
    if analysis_name == "disto":
        return {
            "variation": str(spec.variation).lower(),
            "start": spec.start,
            "stop": spec.stop,
            "points": spec.points,
            "f2overf1": spec.f2overf1,
            "extraction": "rawfile",
        }
    if analysis_name == "four":
        step = spec.step if spec.step is not None else (1 / spec.frequency) / 100
        return {
            "frequency": spec.frequency,
            "output": analysis_output_vector(spec.output),
            "start": spec.start,
            "stop": spec.stop,
            "step": step,
            "extraction": "stdout-print",
        }
    if analysis_name == "noise":
        return {
            "noise_output_node": str(spec.output_node),
            "noise_reference_node": str(spec.reference_node),
            "noise_input_source": source_name(str(spec.input_source)),
            "variation": str(spec.variation).lower(),
            "points": spec.points,
            "start": spec.start,
            "stop": spec.stop,
            "points_per_summary": spec.points_per_summary,
            "extraction": "stdout-print",
        }
    if analysis_name == "op":
        return {"extraction": "stdout-print"}
    if analysis_name == "pz":
        return {
            "input_pos": str(spec.input_pos),
            "input_neg": str(spec.input_neg),
            "output_pos": str(spec.output_pos),
            "output_neg": str(spec.output_neg),
            "transfer": str(spec.transfer).lower(),
            "mode": str(spec.mode).lower(),
            "extraction": "rawfile",
        }
    if analysis_name == "sens":
        return {
            "output": analysis_output_vector(spec.output),
            "variation": str(spec.variation).lower(),
            "start": spec.start,
            "stop": spec.stop,
            "points": spec.points,
            "extraction": "rawfile",
        }
    if analysis_name == "tf":
        return {
            "output": analysis_output_vector(spec.output),
            "input_source": source_name(str(spec.input_source)),
            "extraction": "stdout-print",
        }
    if analysis_name == "tran":
        step = spec.step if spec.step is not None else spec.stop / 1000
        return {
            "start": spec.start,
            "stop": spec.stop,
            "step": step,
            "max_step": spec.max_step,
            "uic": bool(getattr(spec, "uic", False)),
            "extraction": "rawfile",
        }
    if analysis_name == "dc":
        metadata = {
            "source": dc_source_name(str(spec.source)),
            "start": spec.start,
            "stop": spec.stop,
            "step": spec.step,
            "extraction": "rawfile",
        }
        if spec.secondary is not None:
            metadata["secondary"] = {
                "source": dc_source_name(str(spec.secondary.source)),
                "start": spec.secondary.start,
                "stop": spec.secondary.stop,
                "step": spec.secondary.step,
            }
        return metadata
    return {"extraction": "rawfile"}


def validate_node_name(name: str) -> None:
    if not _NODE_RE.match(name) or any(char in name for char in "()"):
        raise ValueError(f"invalid output name for ngspice vector: {name}")


def validate_source_name(name: str) -> None:
    if not _NODE_RE.match(name):
        raise ValueError(f"invalid DC source name: {name}")


def validate_numeric(value, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"invalid {label}: expected numeric scalar")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: expected numeric scalar") from exc
    if not math.isfinite(number):
        raise ValueError(f"invalid {label}: expected finite numeric scalar")
    validate_scalar_value(value, label)


def validate_positive_numeric(value, label: str) -> None:
    validate_numeric(value, label)
    if float(value) <= 0:
        raise ValueError(f"invalid {label}: expected positive numeric scalar")


def validate_scalar_value(value, label: str) -> None:
    text = str(value)
    if any(char in text for char in "\r\n;"):
        raise ValueError(f"invalid {label}: control characters are not allowed")


def control_path(path: Path) -> str:
    text = str(path)
    if any(char in text for char in "\r\n;'"):
        raise ValueError(f"invalid OSDI path: {path}")
    return text
