"""Experiment result persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from monata._json import json_safe as _json_safe
from monata._paths import validate_path_segment
from monata.sim.corner import CornerResults
from monata.sim.export import (
    ArrayStore,
    MutableArrayStore,
    SimResultJsonPayload,
    sim_result_from_dict,
    sim_result_to_dict,
)
from monata.sim.results import SimResult


@dataclass(frozen=True)
class ExperimentResultBundle:
    """Experiment result plus P4 scalar sidecar data."""

    name: str
    result: Any
    measures: Any = field(default_factory=dict)
    summaries: Any = field(default_factory=dict)
    specs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


def save_results(results_dir: Path, name: str, results, specs=None, overwrite: bool = False) -> None:
    safe_name, npz_path, meta_path = _result_paths(results_dir, name)

    if npz_path.exists() and not overwrite:
        raise FileExistsError(f"Results '{safe_name}' already exist. Use overwrite=True.")

    results_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(results, CornerResults):
        _save_corner_results(npz_path, meta_path, results, specs=specs)
    elif isinstance(results, SimResult):
        _save_sim_result(npz_path, meta_path, results, specs=specs)
    else:
        raise TypeError("Experiment.save_results expects SimResult or CornerResults")


def load_results(results_dir: Path, name: str):
    safe_name, npz_path, meta_path = _result_paths(results_dir, name)

    _require_result_files(safe_name, npz_path, meta_path)

    with open(meta_path) as file:
        meta = json.load(file)

    result_type = meta.get("type", "SimResult")
    if result_type == "SimResult":
        return _load_sim_result(npz_path, meta)
    if result_type == "CornerResults":
        return _load_corner_results(npz_path, meta)
    raise ValueError(f"Unknown result type: {result_type}")


def load_result_bundle(results_dir: Path, name: str) -> ExperimentResultBundle:
    safe_name, npz_path, meta_path = _result_paths(results_dir, name)

    _require_result_files(safe_name, npz_path, meta_path)

    with open(meta_path) as file:
        meta = json.load(file)

    result = load_results(results_dir, name)
    if isinstance(result, SimResult):
        measures: Any = result.measures
        summaries: Any = result.summaries
    else:
        measures = _corner_sidecar(meta, "measures")
        summaries = _corner_sidecar(meta, "summaries")
    return ExperimentResultBundle(
        name=safe_name,
        result=result,
        measures=measures,
        summaries=summaries,
        specs=meta.get("specs"),
        metadata=meta,
    )


def _result_paths(results_dir: Path, name: str) -> tuple[str, Path, Path]:
    safe_name = validate_path_segment(name, "result name")
    return safe_name, results_dir / f"{safe_name}.npz", results_dir / f"{safe_name}.json"


def _require_result_files(safe_name: str, npz_path: Path, meta_path: Path) -> None:
    missing = [path.name for path in (npz_path, meta_path) if not path.exists()]
    if not missing:
        return
    if len(missing) == 2:
        raise FileNotFoundError(f"Results '{safe_name}' not found")
    raise FileNotFoundError(f"Results '{safe_name}' is incomplete; missing {missing[0]}")


def _save_sim_result(npz_path: Path, meta_path: Path, result: SimResult, specs=None) -> None:
    arrays: MutableArrayStore = {}
    payload = sim_result_to_dict(result, array_store=arrays)

    np.savez_compressed(npz_path, **arrays)

    meta = {
        "type": "SimResult",
        "payload": payload,
        **_sim_result_sidecar_metadata(payload),
        "specs": _json_safe(specs),
    }
    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)


def _load_sim_result(npz_path: Path, meta: dict) -> SimResult:
    with cast(Any, np.load(npz_path)) as data:
        array_store = cast(ArrayStore, data)
        if "payload" not in meta:
            raise ValueError("SimResult metadata is missing canonical payload")
        return sim_result_from_dict(meta["payload"], array_store=array_store)


def _save_corner_results(npz_path: Path, meta_path: Path, results: CornerResults, specs=None) -> None:
    arrays: MutableArrayStore = {}
    corners_meta = []
    for index, result in enumerate(results):
        payload = sim_result_to_dict(result, array_store=arrays, array_prefix=f"r{index}_")
        corners_meta.append({
            "index": index,
            "payload": payload,
            **_sim_result_sidecar_metadata(payload),
        })

    np.savez_compressed(npz_path, **arrays)
    meta = {"type": "CornerResults", "results": corners_meta, "specs": _json_safe(specs)}
    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)


def _load_corner_results(npz_path: Path, meta: dict) -> CornerResults:
    results = []
    with cast(Any, np.load(npz_path)) as data:
        array_store = cast(ArrayStore, data)
        for result_meta in meta["results"]:
            if "payload" not in result_meta:
                raise ValueError("CornerResults metadata item is missing canonical payload")
            results.append(sim_result_from_dict(result_meta["payload"], array_store=array_store))
    return CornerResults(results)


def _sim_result_sidecar_metadata(payload: SimResultJsonPayload) -> dict[str, Any]:
    return {
        "status": payload["status"],
        "waveform_keys": list(payload["waveforms"].keys()),
        "corner": payload["corner"],
        "metadata": payload["metadata"],
        "error_message": payload["error_message"],
        "measures": payload["measures"],
        "summaries": payload["summaries"],
    }


def _corner_sidecar(meta: dict, key: str) -> dict[str, Any]:
    sidecar = {}
    for item in meta.get("results", []):
        corner = item.get("corner") or {}
        name = corner.get("name") or str(item.get("index"))
        sidecar[str(name)] = item.get(key, {})
    return sidecar
