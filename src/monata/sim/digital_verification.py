from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path

from monata.sim.digital_plan import digital_task_metadata
from monata.sim.digital_results import DigitalTruthTableResult
from monata.sim.digital_table import DigitalTruthTable, DigitalTruthTableMode
from monata.sim.results import SimResult


def write_digital_verification_artifacts(
    artifact_dir: str | Path | None,
    *,
    table: DigitalTruthTable,
    analysis: DigitalTruthTableMode,
    result: DigitalTruthTableResult,
    extra_sim_results: Iterable[SimResult] = (),
) -> None:
    if artifact_dir is None:
        return
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    measures = result.measurements_as_dict()
    (root / "measures.json").write_text(
        json.dumps(measures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_payload = {
        "schema": "monata-digital-verification-run-v1",
        "view": "verification",
        "analysis": analysis,
        "dut": table.dut_name,
        "measures": sorted(measures),
        "tasks": _artifact_task_payloads(result, extra_sim_results=extra_sim_results),
    }
    (root / "run.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_task_payloads(
    result: DigitalTruthTableResult,
    *,
    extra_sim_results: Iterable[SimResult] = (),
) -> list[dict[str, object]]:
    payloads = []
    seen: set[object] = set()
    for sim_result in (
        *result.sim_results,
        *result.propagation_delay_sim_results,
        *tuple(extra_sim_results),
    ):
        metadata = dict(sim_result.metadata)
        artifacts = metadata.get("artifacts")
        if isinstance(artifacts, Mapping):
            directory = artifacts.get("directory")
        else:
            directory = None
        control_metadata = digital_task_metadata(metadata)
        measurements = control_metadata.get("measurements")
        stimulus = control_metadata.get("stimulus")
        index = metadata.get("simulation_artifact_index")
        task_payload: dict[str, object] = {}
        if index is not None:
            task_payload["index"] = index
        if directory is not None:
            task_payload["directory"] = directory
        if isinstance(measurements, (list, tuple)):
            task_payload["measures"] = list(measurements)
        if isinstance(stimulus, Mapping):
            task_payload["stimulus"] = dict(stimulus)
        if task_payload:
            key = _artifact_task_identity(task_payload, sim_result)
            if key in seen:
                continue
            seen.add(key)
            payloads.append(task_payload)
    return sorted(payloads, key=_artifact_task_sort_key)


def _artifact_task_identity(task_payload: Mapping[str, object], sim_result: SimResult) -> object:
    directory = task_payload.get("directory")
    if isinstance(directory, str):
        return ("directory", directory)
    index = task_payload.get("index")
    if isinstance(index, int):
        return ("index", index)
    return ("result", id(sim_result))


def _artifact_task_sort_key(task_payload: Mapping[str, object]) -> tuple[int, int | str]:
    index = task_payload.get("index")
    if isinstance(index, int):
        return (0, index)
    directory = task_payload.get("directory")
    if isinstance(directory, str):
        return (1, directory)
    return (2, "")
