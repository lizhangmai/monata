"""Simulation artifact persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
import json
import shutil
from pathlib import Path
from typing import Any

from monata.sim.task import SimTask


ARTIFACT_SCHEMA = "monata-simulation-artifacts-v1"

_FILE_NAMES = {
    "netlist": "circuit.cir",
    "rawfile": "result.raw",
    "wrdata": "result.dat",
    "stdout": "stdout.txt",
    "stderr": "stderr.txt",
    "commands": "commands.txt",
}


def simulation_artifact_dir(task: SimTask) -> Path | None:
    return task.artifacts.directory


def persist_simulation_artifacts(
    task: SimTask,
    *,
    simulator: str,
    files: Mapping[str, str | Path | None] | None = None,
    text_files: Mapping[str, str | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = simulation_artifact_dir(task)
    if target is None:
        return {}

    file_targets: list[tuple[str, Path, Path]] = []
    for key, source in (files or {}).items():
        filename = _artifact_file_name(key)
        if source is None:
            continue
        source_path = Path(source)
        if not source_path.is_file():
            continue
        file_targets.append((str(key), source_path, target / filename))

    text_targets = [
        (str(key), text, target / _artifact_file_name(key))
        for key, text in (text_files or {}).items()
    ]
    metadata_path = target / "metadata.json"
    _validate_destination_set(
        [destination for _, _, destination in file_targets],
        [destination for _, _, destination in text_targets],
        metadata_path,
        overwrite=overwrite or task.artifacts.overwrite,
    )

    target.mkdir(parents=True, exist_ok=True)
    saved_files: dict[str, str] = {}
    for key, source_path, destination in file_targets:
        shutil.copy2(source_path, destination)
        saved_files[key] = str(destination)

    for key, text, destination in text_targets:
        destination.write_text(text or "", encoding="utf-8")
        saved_files[key] = str(destination)

    payload = {
        "schema": ARTIFACT_SCHEMA,
        "simulator": simulator,
        "directory": str(target),
        "files": dict(saved_files),
        **dict(metadata or {}),
    }
    metadata_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    saved_files["metadata"] = str(metadata_path)
    return {
        "artifacts": {
            "schema": ARTIFACT_SCHEMA,
            "directory": str(target),
            "files": saved_files,
        }
    }


def _validate_destination_set(
    file_destinations: list[Path],
    text_destinations: list[Path],
    metadata_path: Path,
    *,
    overwrite: bool,
) -> None:
    planned = [*file_destinations, *text_destinations, metadata_path]
    duplicates = _duplicate_paths(planned)
    if duplicates:
        rendered = ", ".join(str(path) for path in duplicates)
        raise ValueError(f"duplicate simulation artifact destination: {rendered}")
    if overwrite:
        return
    existing = [path for path in planned if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"simulation artifact destination already exists: {rendered}")


def _duplicate_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for path in paths:
        if path in seen:
            duplicates.append(path)
            continue
        seen.add(path)
    return duplicates


def _artifact_file_name(key: object) -> str:
    key_text = str(key)
    try:
        return _FILE_NAMES[key_text]
    except KeyError as exc:
        raise ValueError(f"unknown simulation artifact key: {key_text}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "value") and isinstance(value.value, str | int | float | bool):
        return value.value
    return str(value)


__all__ = [
    "ARTIFACT_SCHEMA",
    "persist_simulation_artifacts",
    "simulation_artifact_dir",
]
