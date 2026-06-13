"""Project — top-level directory for circuit exploration experiments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from monata.workspace.experiment import Experiment

from monata._config import read_toml, reject_unknown_fields, write_project_config
from monata._paths import validate_path_segment

PROJECT_CONFIG_FILENAME = "project.toml"
EXPERIMENTS_DIRNAME = "experiments"
LIBRARIES_DIRNAME = "libraries"
MODEL_MANIFEST_FILENAME = "models.toml"
_PROJECT_CONFIG_FIELDS = frozenset({"project", "libraries"})
_PROJECT_TABLE_FIELDS = frozenset({"name"})
_PROJECT_LIBRARY_FIELDS = frozenset({"name", "path"})


def _validate_project_meta(meta: dict[str, Any]) -> None:
    reject_unknown_fields(meta, _PROJECT_CONFIG_FIELDS, "project.toml")
    project = meta.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("[project] must be a table")
    reject_unknown_fields(project, _PROJECT_TABLE_FIELDS, "project table")
    libraries = meta.get("libraries", [])
    if not isinstance(libraries, list):
        raise ValueError("project libraries must be an array of tables")
    for index, entry in enumerate(libraries):
        if not isinstance(entry, dict):
            raise ValueError(f"project libraries[{index}] must be a table")
        reject_unknown_fields(
            entry,
            _PROJECT_LIBRARY_FIELDS,
            f"project libraries[{index}]",
        )
        for field in ("name", "path"):
            if field not in entry:
                raise ValueError(f"project libraries[{index}] is missing {field}")


class Project:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Project does not exist: {self.path}; use Project.create(path) to create it")
        self._meta = self._load_meta()

    @classmethod
    def open(cls, path) -> "Project":
        """Open an existing project without creating filesystem state."""

        return cls(path)

    @classmethod
    def create(cls, path, *, exist_ok: bool = False) -> "Project":
        """Create a project directory and return the opened project."""

        project = cls.__new__(cls)
        project.path = Path(path)
        if project.path.exists() and not exist_ok:
            raise FileExistsError(f"Project already exists: {project.path}")
        project._create()
        project._meta = project._load_meta()
        return project

    @property
    def name(self) -> str:
        return self._meta.get("project", {}).get("name", self.path.name)

    def _create(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._experiments_path().mkdir(exist_ok=True)
        self._libraries_path().mkdir(exist_ok=True)
        if not self._project_config_path().exists():
            write_project_config(self._project_config_path(), name=self.path.name)

    def _load_meta(self) -> dict:
        toml_path = self._project_config_path()
        if toml_path.exists():
            meta = read_toml(toml_path)
            _validate_project_meta(meta)
            return meta
        return {}

    def _project_config_path(self) -> Path:
        return self.path / PROJECT_CONFIG_FILENAME

    def _experiments_path(self) -> Path:
        return self.path / EXPERIMENTS_DIRNAME

    def _experiment_path(self, name: str) -> tuple[str, Path]:
        safe_name = validate_path_segment(name, "experiment name")
        return safe_name, self._experiments_path() / safe_name

    def _libraries_path(self) -> Path:
        return self.path / LIBRARIES_DIRNAME

    def create_library(
        self,
        name: str,
        tech_model_paths: list[str] | None = None,
        techlib_attachments: list[str] | None = None,
        default_corner: str | None = None,
        description: str = "",
    ):
        """Create a project-local library and record it in project metadata."""
        from monata.library import Library

        safe_name = validate_path_segment(name, "library name")
        lib = Library.create(
            self._libraries_path() / safe_name,
            name=safe_name,
            tech_model_paths=tech_model_paths or [],
            techlib_attachments=techlib_attachments or [],
            default_corner=default_corner,
            description=description,
        )
        return self._record_library(lib, name=safe_name)

    def add_library(self, path, name: str | None = None):
        """Register an existing library path with this project."""
        from monata.library import Library

        lib_path = Path(path).resolve()
        lib = Library(lib_path)
        return self._record_library(lib, name=name)

    def _record_library(self, lib, name: str | None = None):
        lib_name = name or lib.name
        libraries = [entry for entry in self._library_entries() if entry["name"] != lib_name]
        libraries.append({"name": lib_name, "path": _project_path(self.path, lib.path)})
        self._write_meta(libraries)
        self._meta = self._load_meta()
        return lib

    def list_libraries(self) -> list[str]:
        return sorted(entry["name"] for entry in self._library_entries())

    def get_library(self, name: str):
        from monata.errors import LibraryNotFoundError
        from monata.library import Library

        for entry in self._library_entries():
            if entry["name"] == name:
                return Library(_resolve_project_path(self.path, entry["path"]))
        raise LibraryNotFoundError(name)

    def library_registry(self):
        """Return a registry populated with the project's recorded libraries."""
        from monata.registry import LibraryRegistry

        registry = LibraryRegistry()
        for entry in self._library_entries():
            registry.add_library(_resolve_project_path(self.path, entry["path"]))
        return registry

    def model_manifest_path(self) -> Path:
        """Return the project-local model manifest path."""
        return self.path / MODEL_MANIFEST_FILENAME

    def model_manifest(self):
        """Load the project-local model manifest, or return an empty manifest."""
        from monata.models.manifest import ModelManifest

        path = self.model_manifest_path()
        if not path.exists():
            return ModelManifest()
        return ModelManifest.load(path, project_path=self.path)

    def save_model_manifest(self, manifest) -> None:
        """Persist project-local model metadata without rewriting project.toml."""
        manifest.save(self.model_manifest_path(), project_path=self.path)

    def new_experiment(self, name: str, description: str = "") -> Experiment:
        from monata.workspace.experiment import Experiment

        safe_name, exp_path = self._experiment_path(name)
        if exp_path.exists():
            raise FileExistsError(f"Experiment already exists: {safe_name}")
        return Experiment(exp_path, description=description)

    def list_experiments(self) -> list[str]:
        exp_dir = self._experiments_path()
        if not exp_dir.exists():
            return []
        return sorted(d.name for d in exp_dir.iterdir() if d.is_dir() and (d / "experiment.toml").exists())

    def compare(self, *experiment_names: str) -> list[dict[str, Any]] | None:
        rows = []
        names = experiment_names or tuple(self.list_experiments())
        for name in names:
            from monata.workspace.experiment import Experiment

            safe_name, exp_path = self._experiment_path(name)
            if exp_path.exists():
                exp = Experiment(exp_path)
                row = {"experiment": safe_name}
                row.update(exp.summary)
                rows.append(row)
        return rows if rows else None

    def _library_entries(self) -> list[dict[str, str]]:
        entries = self._meta.get("libraries", [])
        return [
            {"name": str(entry["name"]), "path": str(entry["path"])}
            for entry in entries
        ]

    def _write_meta(self, libraries: list[dict[str, str]]) -> None:
        project = self._meta.get("project", {})
        name = project.get("name", self.path.name)
        write_project_config(self._project_config_path(), name=name, libraries=libraries)


def _project_path(project_path: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_path.resolve()))
    except ValueError:
        return str(path)


def _resolve_project_path(project_path: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_path / candidate
