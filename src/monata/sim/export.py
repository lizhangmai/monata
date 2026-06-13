"""Lightweight simulation result export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monata.sim.export_hdf5 import export_sim_result_hdf5, load_sim_result_hdf5
from monata.sim.export_payload import (
    ArrayPayload,
    ArrayPayloadBuilder,
    ArrayStore,
    MutableArrayPayload,
    MutableArrayStore,
    MutableSimResultJsonPayload,
    SimResultJsonPayload,
    _read_npz_array_store,
    _write_npz_array_store,
    sim_result_from_dict,
    sim_result_to_dict,
)
from monata.sim.results import SimResult

__all__ = [
    "ArrayPayload",
    "ArrayPayloadBuilder",
    "ArrayStore",
    "MutableArrayPayload",
    "MutableArrayStore",
    "MutableSimResultJsonPayload",
    "SimResultJsonPayload",
    "export_sim_result_hdf5",
    "export_sim_result_json",
    "load_sim_result_hdf5",
    "load_sim_result_json",
    "sim_result_from_dict",
    "sim_result_to_dict",
]


def export_sim_result_json(
    result: SimResult,
    path: str | Path,
    *,
    array_store_path: str | Path | None = None,
    array_prefix: str = "",
) -> Path:
    """Export a ``SimResult`` to a portable JSON file."""

    output_path = Path(path)
    array_store: dict[str, Any] | None = {} if array_store_path is not None else None
    payload = sim_result_to_dict(result, array_store=array_store, array_prefix=array_prefix)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if array_store_path is not None:
        _write_npz_array_store(array_store_path, array_store or {})
    return output_path


def load_sim_result_json(path: str | Path, *, array_store_path: str | Path | None = None) -> SimResult:
    """Load a ``SimResult`` previously written by ``export_sim_result_json``."""

    payload = json.loads(Path(path).read_text())
    array_store = _read_npz_array_store(array_store_path) if array_store_path is not None else None
    return sim_result_from_dict(payload, array_store=array_store)
