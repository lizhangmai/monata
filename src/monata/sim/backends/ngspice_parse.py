"""Output parsing helpers for the subprocess ngspice backend."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from monata.sim.backends.ngspice_output import (
    RawfileParseError,
    parse_rawfile,
    parse_wrdata,
)
from monata.sim.backends.ngspice_common import fallback_wrdata_path
from monata.sim.backends.ngspice_plan import NgspiceOutputRequest, NgspiceTaskPlan
from monata.sim.backends.ngspice_stdout import (
    parse_fourier_stdout,
    parse_noise_print,
    parse_op_print,
    parse_tf_stdout,
)
from monata.sim.results import analysis_result_from_arrays
from monata.sim._vector_identity import p3_vector_metadata, pole_zero_vector_kind
from monata.sim.vector_names import normalize_vector_name

_RAWFILE_ANALYSES_WITHOUT_ABSCISSA = {"op", "pz"}
_STDOUT_VECTOR_KINDS = {
    "four": "fourier_component",
    "tf": "transfer_function",
}
_STDOUT_VECTOR_QUANTITIES = {
    "four": {
        "harmonic": "harmonic",
        "frequency": "frequency",
        "fourier_magnitude": "magnitude",
        "fourier_phase": "phase",
        "fourier_normalized_magnitude": "normalized_magnitude",
        "fourier_normalized_phase": "normalized_phase",
    },
    "tf": {
        "transfer_function": "gain",
        "input_resistance": "resistance",
        "output_resistance": "resistance",
    },
}
_STDOUT_VECTOR_UNITS = {
    "four": {
        "frequency": "Hz",
        "fourier_phase": "deg",
        "fourier_normalized_phase": "deg",
    },
    "tf": {
        "input_resistance": "Ohm",
        "output_resistance": "Ohm",
    },
}


def parse_output(output_path: Path, stdout: str, plan: NgspiceTaskPlan):
    if plan.analysis_name == "op" and plan.extraction == "stdout-print":
        waveforms = parse_op_print(stdout, list(plan.output_names))
        extra_metadata = {
            "fallback_used": True,
            "fallback_reason": "rawfile_not_enabled_for_analysis",
        }
        return _with_typed_result(None, waveforms, plan, extra_metadata)
    if plan.analysis_name == "noise":
        noise_sweep_var, noise_waveforms, totals = parse_noise_print(stdout)
        extra_metadata = {
            "noise_totals": totals,
            "fallback_used": True,
            "fallback_reason": "rawfile_not_enabled_for_analysis",
        }
        return _with_typed_result(noise_sweep_var, noise_waveforms, plan, extra_metadata)
    if plan.analysis_name == "tf":
        waveforms = parse_tf_stdout(stdout)
        extra_metadata = {
            "extraction": "stdout-print",
            "fallback_used": False,
            **_observed_output_metadata(waveforms),
        }
        return _with_typed_result(None, waveforms, plan, extra_metadata)
    if plan.analysis_name == "four":
        waveforms, fourier_metadata = parse_fourier_stdout(stdout)
        extra_metadata = {
            "extraction": "stdout-print",
            "fallback_used": False,
            **fourier_metadata,
            **_observed_output_metadata(waveforms),
        }
        return _with_typed_result(waveforms.get("frequency"), waveforms, plan, extra_metadata)
    try:
        rawfile = parse_rawfile(output_path)
        sweep_var = (
            None
            if _rawfile_analysis_without_abscissa(plan)
            else rawfile.scale.data
        )
        waveforms = (
            _rawfile_all_waveforms(rawfile, plan)
            if plan.metadata.get("write_all_vectors")
            else _rawfile_waveforms(rawfile, plan)
        )
        observed_metadata = _observed_output_metadata(waveforms) if plan.metadata.get("write_all_vectors") else {}
        extra_metadata = {
            **rawfile.metadata,
            "extraction": "rawfile",
            "fallback_used": False,
            **observed_metadata,
            **_p3_rawfile_metadata(plan, waveforms),
        }
        return _with_typed_result(sweep_var, waveforms, plan, extra_metadata)
    except (KeyError, RawfileParseError) as exc:
        if plan.metadata.get("write_all_vectors"):
            raise ValueError(f"ngspice rawfile extraction failed ({exc})") from exc
        fallback_path = fallback_wrdata_path(output_path)
        try:
            wrdata_sweep_var, wrdata_waveforms = parse_wrdata(fallback_path, list(plan.output_names))
        except Exception as fallback_exc:
            raise ValueError(
                f"ngspice rawfile extraction failed ({exc}); wrdata fallback also failed ({fallback_exc})"
            ) from fallback_exc
        extra_metadata = {
            "extraction": "wrdata",
            "fallback_used": True,
            "fallback_reason": "rawfile_parser_failed",
            "rawfile_error": str(exc),
        }
        return _with_typed_result(wrdata_sweep_var, wrdata_waveforms, plan, extra_metadata)


def _with_typed_result(sweep_var, waveforms, plan: NgspiceTaskPlan, extra_metadata: dict):
    typed_metadata = _typed_result_metadata(plan, waveforms)
    metadata = {
        **plan.metadata,
        **extra_metadata,
        **typed_metadata,
    }
    analysis_result = analysis_result_from_arrays(
        waveforms,
        sweep_var,
        metadata,
        analysis=plan.analysis_name,
        source=metadata.get("extraction"),
        source_vectors=metadata.get("vector_raw_names"),
    )
    return sweep_var, waveforms, analysis_result, {**extra_metadata, **typed_metadata}


def _observed_output_metadata(waveforms: Mapping[str, object]) -> dict:
    output_vectors = list(waveforms)
    vector_raw_names = {}
    vector_kinds = {}
    vector_quantities = {}
    vector_units = {}
    output_requests = []
    for name in output_vectors:
        identity = normalize_vector_name(name)
        vector_raw_names[name] = identity.raw_vector_name
        vector_kinds[name] = identity.vector_kind
        if identity.quantity is not None:
            vector_quantities[name] = identity.quantity
        if identity.quantity == "voltage":
            vector_units[name] = "V"
        elif identity.quantity == "current":
            vector_units[name] = "A"
        output_requests.append({
            "output_name": name,
            "vector": name,
            "display_name": identity.display_name,
            "normalized_name": identity.normalized_name,
            "raw_vector_name": identity.raw_vector_name,
            "vector_kind": identity.vector_kind,
            "quantity": identity.quantity,
            "transform": None,
        })
    return {
        "output_vectors": output_vectors,
        "output_requests": output_requests,
        "vector_raw_names": vector_raw_names,
        "vector_kinds": vector_kinds,
        "vector_quantities": vector_quantities,
        "vector_units": vector_units,
    }


def _typed_result_metadata(plan: NgspiceTaskPlan, waveforms: dict[str, object]) -> dict:
    metadata = {
        "vector_raw_names": {
            request.output_name: request.identity.raw_vector_name
            for request in plan.output_requests
            if request.output_name in waveforms
        },
        "vector_kinds": {},
        "vector_quantities": {},
        "vector_units": {},
    }
    for request in plan.output_requests:
        name = request.output_name
        if name not in waveforms:
            continue
        metadata["vector_kinds"][name] = request.identity.vector_kind
        if request.identity.quantity is not None:
            metadata["vector_quantities"][name] = request.identity.quantity
        if request.identity.quantity == "voltage":
            metadata["vector_units"][name] = "V"
        elif request.identity.quantity == "current":
            metadata["vector_units"][name] = "A"
        transform = request.transform
        if transform is None:
            continue
        metadata["vector_kinds"][name] = (
            request.identity.vector_kind if _preserves_identity_kind(request) else "ac_component"
        )
        if transform == "phase":
            metadata["vector_quantities"][name] = "phase"
            metadata["vector_units"][name] = "rad"
        elif transform == "db":
            metadata["vector_quantities"][name] = "gain_db"
            metadata["vector_units"][name] = "dB"
        elif transform in {"mag", "real", "imag"}:
            metadata["vector_quantities"][name] = request.identity.quantity
            if request.identity.quantity == "voltage":
                metadata["vector_units"][name] = "V"
            elif request.identity.quantity == "current":
                metadata["vector_units"][name] = "A"
    if plan.analysis_name in {"four", "tf"}:
        metadata["vector_raw_names"] = {name: name for name in waveforms}
        metadata["vector_quantities"].update(_stdout_vector_quantities(plan.analysis_name, waveforms))
        metadata["vector_units"].update(_stdout_vector_units(plan.analysis_name, waveforms))
        metadata["vector_kinds"].update(_stdout_vector_kinds(plan.analysis_name, waveforms))
        metadata["vector_metadata"] = _stdout_vector_metadata(plan, waveforms)
    return {key: value for key, value in metadata.items() if value}


def _preserves_identity_kind(request: NgspiceOutputRequest) -> bool:
    return request.transform == "mag" and request.output_name == request.identity.normalized_name


def _stdout_vector_kinds(analysis_name: str, waveforms: dict[str, object]) -> dict[str, str]:
    vector_kind = _STDOUT_VECTOR_KINDS.get(analysis_name)
    return {} if vector_kind is None else {name: vector_kind for name in waveforms}


def _stdout_vector_quantities(analysis_name: str, waveforms: dict[str, object]) -> dict[str, str]:
    return _metadata_for_waveforms(_STDOUT_VECTOR_QUANTITIES.get(analysis_name, {}), waveforms)


def _stdout_vector_units(analysis_name: str, waveforms: dict[str, object]) -> dict[str, str]:
    return _metadata_for_waveforms(_STDOUT_VECTOR_UNITS.get(analysis_name, {}), waveforms)


def _metadata_for_waveforms(metadata: Mapping[str, str], waveforms: Mapping[str, object]) -> dict[str, str]:
    return {name: value for name, value in metadata.items() if name in waveforms}


def _stdout_vector_metadata(plan: NgspiceTaskPlan, waveforms: dict[str, object]) -> dict[str, dict[str, object]]:
    if plan.analysis_name == "four":
        context = _compact_metadata({
            "fourier_output_vector": plan.metadata.get("output"),
            "fourier_frequency": plan.metadata.get("frequency"),
        })
        return {name: dict(context) for name in waveforms} if context else {}
    if plan.analysis_name == "tf":
        output = plan.metadata.get("output")
        input_source = plan.metadata.get("input_source")
        return {
            name: metadata
            for name, metadata in {
                "transfer_function": _compact_metadata({
                    "tf_output_vector": output,
                    "tf_input_source": input_source,
                }),
                "input_resistance": _compact_metadata({"tf_input_source": input_source}),
                "output_resistance": _compact_metadata({"tf_output_vector": output}),
            }.items()
            if name in waveforms and metadata
        }
    return {}


def _compact_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metadata.items() if value is not None}


def _rawfile_waveforms(rawfile, plan: NgspiceTaskPlan) -> dict[str, object]:
    waveforms = {}
    for request in plan.output_requests:
        try:
            data = rawfile.vector(request.vector).data
        except KeyError as exc:
            raise RawfileParseError(f"ngspice rawfile missing vector {request.vector}") from exc
        waveforms[request.output_name] = _apply_output_transform(data, request.transform)
    return waveforms


def _rawfile_all_waveforms(rawfile, plan: NgspiceTaskPlan) -> dict[str, object]:
    waveforms = {
        vector.name: vector.data
        for vector in rawfile.vectors
        if _rawfile_analysis_without_abscissa(plan) or vector.index != rawfile.scale.index
    }
    if not waveforms:
        raise RawfileParseError(f"ngspice {plan.analysis_name} rawfile contained no result vectors")
    return waveforms


def _rawfile_analysis_without_abscissa(plan: NgspiceTaskPlan) -> bool:
    return bool(plan.metadata.get("write_all_vectors")) and plan.analysis_name in _RAWFILE_ANALYSES_WITHOUT_ABSCISSA


def _p3_rawfile_metadata(plan: NgspiceTaskPlan, waveforms: dict[str, object]) -> dict:
    vector_kind = {
        "disto": "distortion",
        "pz": "pole",
        "sens": "sensitivity",
    }.get(plan.analysis_name)
    if vector_kind is None:
        return {}
    vector_kinds = {
        name: pole_zero_vector_kind(name) if plan.analysis_name == "pz" else vector_kind
        for name in waveforms
    }
    return {
        "output_vectors": list(waveforms),
        "vector_kinds": vector_kinds,
        "vector_quantities": {
            name: _p3_vector_quantity(plan.analysis_name, name, kind)
            for name, kind in vector_kinds.items()
        },
        "vector_units": {
            name: _p3_vector_unit(plan.analysis_name, name)
            for name in waveforms
        },
        "vector_raw_names": {name: name for name in waveforms},
        "vector_metadata": {
            name: p3_vector_metadata(plan.analysis_name, name, kind)
            for name, kind in vector_kinds.items()
        },
    }


def _p3_vector_quantity(analysis_name: str, name: str, vector_kind: str) -> str | None:
    if analysis_name == "sens":
        return "sensitivity"
    if analysis_name == "pz":
        return vector_kind
    return normalize_vector_name(name).quantity


def _p3_vector_unit(analysis_name: str, name: str) -> str | None:
    if analysis_name in {"pz", "sens"}:
        return None
    quantity = normalize_vector_name(name).quantity
    if quantity == "voltage":
        return "V"
    if quantity == "current":
        return "A"
    return None


def _apply_output_transform(data, transform: str | None):
    if transform is None:
        return data
    if transform == "mag":
        return abs(data)
    if transform == "phase":
        return np.angle(data)
    if transform == "real":
        return np.real(data)
    if transform == "imag":
        return np.imag(data)
    if transform == "db":
        with np.errstate(divide="ignore"):
            return 20 * np.log10(abs(data))
    return data
