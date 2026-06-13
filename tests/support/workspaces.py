from __future__ import annotations

from pathlib import Path

from monata.workspace.experiment import Experiment
from monata.workspace.project import Project

__all__ = ["create_experiment", "create_project", "open_project"]


def create_project(tmp_path: Path, name: str = "proj", *, exist_ok: bool = False) -> Project:
    return Project.create(tmp_path / name, exist_ok=exist_ok)


def open_project(tmp_path: Path, name: str = "proj") -> Project:
    return Project(tmp_path / name)


def create_experiment(tmp_path: Path, project_name: str = "proj", experiment_name: str = "exp") -> Experiment:
    return create_project(tmp_path, project_name).new_experiment(experiment_name)
