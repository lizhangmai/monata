"""Experiment — stores simulation results, circuits, and notes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monata._config import read_toml, reject_unknown_fields
from monata._paths import toml_string
from monata.workspace.result_store import (
    ExperimentResultBundle,
    load_result_bundle,
    load_results,
    save_results,
)


_EXPERIMENT_CONFIG_FIELDS = frozenset({"experiment"})
_EXPERIMENT_TABLE_FIELDS = frozenset({"name", "description"})


def _validate_experiment_meta(meta: dict[str, Any]) -> None:
    reject_unknown_fields(meta, _EXPERIMENT_CONFIG_FIELDS, "experiment.toml")
    experiment = meta.get("experiment")
    if experiment is None:
        raise ValueError("experiment.toml is missing [experiment]")
    if not isinstance(experiment, dict):
        raise ValueError("[experiment] must be a table")
    reject_unknown_fields(experiment, _EXPERIMENT_TABLE_FIELDS, "experiment table")


class Experiment:
    def __init__(self, path, description: str = ""):
        self.path = Path(path)
        if not self.path.exists():
            self._create(description)
        self._results_path().mkdir(exist_ok=True)
        self._description = self._load_description(default=description)

    @property
    def name(self) -> str:
        return self.path.name

    def _create(self, description: str):
        self.path.mkdir(parents=True, exist_ok=True)
        self._results_path().mkdir(exist_ok=True)
        meta = (
            f'[experiment]\nname = "{toml_string(self.path.name)}"\n'
            f'description = "{toml_string(description)}"\n'
        )
        (self.path / "experiment.toml").write_text(meta)

    def _load_description(self, *, default: str) -> str:
        meta_path = self.path / "experiment.toml"
        if not meta_path.exists():
            return default
        data = read_toml(meta_path)
        _validate_experiment_meta(data)
        return str(data["experiment"].get("description", default))

    def _results_path(self) -> Path:
        return self.path / "results"

    def save_results(self, name: str, results, specs=None, overwrite: bool = False):
        save_results(self._results_path(), name, results, specs=specs, overwrite=overwrite)

    def load_results(self, name: str):
        return load_results(self._results_path(), name)

    def load_result_bundle(self, name: str) -> ExperimentResultBundle:
        return load_result_bundle(self._results_path(), name)

    def note(self, text: str):
        notes_path = self.path / "notes.md"
        with open(notes_path, "a") as f:
            f.write(text + "\n")

    @property
    def summary(self) -> dict:
        results_dir = self._results_path()
        result_names = (
            sorted(p.stem for p in results_dir.glob("*.npz") if p.with_suffix(".json").exists())
            if results_dir.exists()
            else []
        )
        return {
            "name": self.name,
            "description": self._description,
            "results": result_names,
        }


__all__ = ["Experiment", "ExperimentResultBundle"]
