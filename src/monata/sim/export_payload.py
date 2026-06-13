"""Payload conversion helpers for simulation result export."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from monata._json import json_safe as _json_safe
from monata.corner import corner_from_payload, corner_to_payload
from monata.measure.result import MeasureSet
from monata.measure.summary import AnalysisSummary
from monata.sim.results import AnalysisResult, Waveform
from monata.sim.results import SimResult

SimResultJsonPayload: TypeAlias = Mapping[str, Any]
MutableSimResultJsonPayload: TypeAlias = dict[str, Any]
ArrayPayload: TypeAlias = Mapping[str, Any]
MutableArrayPayload: TypeAlias = dict[str, Any]
ArrayStore: TypeAlias = Mapping[str, Any]
MutableArrayStore: TypeAlias = MutableMapping[str, Any]
ArrayPayloadBuilder: TypeAlias = Callable[[Any, str], MutableArrayPayload]

_SIM_RESULT_PAYLOAD_FIELDS = frozenset({
    "status",
    "waveforms",
    "sweep_var",
    "corner",
    "metadata",
    "error_message",
    "measures",
    "summaries",
    "analysis_result",
})

__all__ = [
    "ArrayPayload",
    "ArrayPayloadBuilder",
    "ArrayStore",
    "MutableArrayPayload",
    "MutableArrayStore",
    "MutableSimResultJsonPayload",
    "SimResultJsonPayload",
    "sim_result_from_dict",
    "sim_result_to_dict",
]


def sim_result_to_dict(
    result: SimResult,
    *,
    array_store: MutableArrayStore | None = None,
    array_prefix: str = "",
) -> MutableSimResultJsonPayload:
    analysis = result.analysis_result
    array_payload = _array_payload_builder(array_store, array_prefix)
    return {
        "status": result.status,
        "waveforms": {
            name: array_payload(values, f"waveforms/{name}")
            for name, values in result.waveforms.items()
        },
        "sweep_var": array_payload(result.sweep_var, "sweep_var") if result.sweep_var is not None else None,
        "corner": corner_to_payload(result.corner),
        "metadata": _json_safe(result.metadata),
        "error_message": result.error_message,
        "measures": _json_safe(result.measures.to_dict()),
        "summaries": _json_safe(result.summaries),
        "analysis_result": {
            "analysis": analysis.analysis,
            "source": analysis.source,
            "metadata": _json_safe(analysis.metadata),
            "abscissa": _waveform_to_payload(analysis.abscissa, array_payload, "analysis/abscissa")
            if analysis.abscissa is not None
            else None,
            "waveforms": {
                name: _waveform_to_payload(waveform, array_payload, f"analysis/waveforms/{name}")
                for name, waveform in analysis.waveforms.items()
            },
        }
        if analysis is not None
        else None,
    }


def sim_result_from_dict(payload: SimResultJsonPayload, *, array_store: ArrayStore | None = None) -> SimResult:
    unknown = sorted(key for key in payload if key not in _SIM_RESULT_PAYLOAD_FIELDS)
    if unknown:
        raise ValueError(f"simulation result payload has unknown fields: {', '.join(unknown)}")
    waveforms = {
        name: _array_from_payload(values, array_store=array_store)
        for name, values in payload.get("waveforms", {}).items()
    }
    sweep_var = (
        _array_from_payload(payload["sweep_var"], array_store=array_store)
        if payload.get("sweep_var") is not None
        else None
    )
    return SimResult(
        status=str(payload["status"]),
        waveforms=waveforms,
        sweep_var=sweep_var,
        corner=corner_from_payload(payload.get("corner")),
        metadata=dict(payload.get("metadata", {})),
        error_message=payload.get("error_message"),
        analysis_result=_analysis_result_from_payload(payload.get("analysis_result"), array_store=array_store),
        measures=MeasureSet(payload.get("measures", {})),
        summaries=_summaries_from_payload(payload.get("summaries", {})),
    )


def _analysis_result_from_payload(
    payload: SimResultJsonPayload | None,
    *,
    array_store: ArrayStore | None,
) -> AnalysisResult | None:
    if payload is None:
        return None
    return AnalysisResult(
        analysis=payload.get("analysis"),
        waveforms={
            name: _waveform_from_payload(
                metadata,
                _array_from_payload(metadata["data"], array_store=array_store),
                array_store=array_store,
            )
            for name, metadata in payload.get("waveforms", {}).items()
        },
        abscissa=_waveform_from_payload(
            payload["abscissa"],
            _array_from_payload(payload["abscissa"]["data"], array_store=array_store),
            array_store=array_store,
        )
        if payload.get("abscissa") is not None
        else None,
        metadata=dict(payload.get("metadata", {})),
        source=payload.get("source"),
    )


def _waveform_to_payload(
    waveform: Waveform,
    array_payload: ArrayPayloadBuilder,
    key: str,
) -> MutableSimResultJsonPayload:
    return {
        "name": waveform.name,
        "data": array_payload(waveform.data, key),
        "abscissa_data": (
            array_payload(waveform.abscissa_data, f"{key}/abscissa_data")
            if waveform.abscissa_data is not None
            else None
        ),
        "unit": waveform.unit,
        "quantity": waveform.quantity,
        "title": waveform.title,
        "metadata": _json_safe(waveform.metadata),
        "display_name": waveform.display_name,
        "normalized_name": waveform.normalized_name,
        "raw_vector_name": waveform.raw_vector_name,
        "vector_kind": waveform.vector_kind,
        "source_vector": waveform.source_vector,
        "abscissa_name": waveform.abscissa_name,
        "analysis": waveform.analysis,
        "source": waveform.source,
        "extraction": waveform.extraction,
        "plot_name": waveform.plot_name,
    }


def _waveform_from_payload(
    payload: SimResultJsonPayload,
    data: np.ndarray,
    *,
    array_store: ArrayStore | None,
) -> Waveform:
    return Waveform(
        name=_waveform_name_from_payload(payload),
        data=data,
        unit=payload.get("unit"),
        quantity=payload.get("quantity"),
        title=payload.get("title"),
        abscissa_data=(
            _array_from_payload(payload["abscissa_data"], array_store=array_store)
            if payload.get("abscissa_data") is not None
            else None
        ),
        metadata=dict(payload.get("metadata", {})),
        display_name=payload.get("display_name"),
        normalized_name=payload.get("normalized_name"),
        raw_vector_name=payload.get("raw_vector_name"),
        vector_kind=payload.get("vector_kind", "unknown"),
        source_vector=payload.get("source_vector"),
        abscissa=payload.get("abscissa_name"),
        analysis=payload.get("analysis"),
        source=payload.get("source"),
        extraction=payload.get("extraction"),
        plot_name=payload.get("plot_name"),
    )


def _waveform_name_from_payload(payload: SimResultJsonPayload) -> str:
    name = payload.get("name")
    if name is None or str(name) == "":
        raise ValueError("waveform payload missing name")
    return str(name)


def _summaries_from_payload(summaries: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, summary in dict(summaries or {}).items():
        if isinstance(summary, dict) and {"analysis", "values"}.issubset(summary):
            result[str(name)] = AnalysisSummary(
                analysis=summary["analysis"],
                values=summary.get("values", {}),
                units=summary.get("units", {}),
                reasons=summary.get("reasons", {}),
                source=summary.get("source", "summary"),
                metadata=summary.get("metadata", {}),
            )
        else:
            result[str(name)] = summary
    return result


def _array_payload_builder(
    array_store: MutableArrayStore | None,
    array_prefix: str,
) -> ArrayPayloadBuilder:
    if array_store is None:
        return lambda values, key: _array_to_payload(values)

    def build(values: Any, key: str) -> MutableArrayPayload:
        array = np.asarray(values)
        storage_key = _unique_array_storage_key(_array_storage_key(array_prefix, key), array_store)
        array_store[storage_key] = array
        return {
            "storage": "npz",
            "key": storage_key,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "kind": "complex" if np.iscomplexobj(array) else "real",
        }

    return build


def _write_npz_array_store(path: str | Path, array_store: Mapping[str, Any]) -> None:
    arrays: dict[str, Any] = {key: np.asarray(value) for key, value in array_store.items()}
    with Path(path).open("wb") as file:
        np.savez_compressed(file, **arrays)


def _read_npz_array_store(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as store:
        return {key: np.asarray(store[key]) for key in store.files}


def _unique_array_storage_key(base_key: str, array_store: MutableArrayStore) -> str:
    if base_key not in array_store:
        return base_key
    index = 1
    while f"{base_key}__{index}" in array_store:
        index += 1
    return f"{base_key}__{index}"


def _array_storage_key(prefix: str, key: str) -> str:
    text = f"{prefix}{key}" if prefix else key
    return text.replace("/", "__")


def _array_to_payload(values: Any) -> MutableArrayPayload:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        data: Any = {"real": np.real(array).tolist(), "imag": np.imag(array).tolist()}
        kind = "complex"
    else:
        data = array.tolist()
        kind = "real"
    return {"dtype": str(array.dtype), "shape": list(array.shape), "kind": kind, "data": data}


def _array_from_payload(payload: ArrayPayload, *, array_store: ArrayStore | None = None) -> np.ndarray:
    if payload.get("storage") == "npz":
        if array_store is None:
            raise ValueError("array payload references external storage but no array_store was provided")
        data = np.asarray(array_store[str(payload["key"])])
        return data.astype(np.dtype(payload["dtype"]), copy=False).reshape(tuple(payload["shape"]))
    if payload["kind"] == "complex":
        data = np.asarray(payload["data"]["real"]) + 1j * np.asarray(payload["data"]["imag"])
    else:
        data = np.asarray(payload["data"])
    return data.astype(np.dtype(payload["dtype"]), copy=False).reshape(tuple(payload["shape"]))
