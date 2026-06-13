"""Project-local model manifest persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from monata._paths import toml_string
from monata._types import ReferenceMode
from monata.models.diagnostics import ModelDiagnostic, ModelDiagnosticError
from monata.models.registry import ModelEntry

_MANIFEST_FIELDS = frozenset({"models", "devices"})
_PATH_FIELDS = ("osdi_path", "source_va", "model_file")
_DEVICE_METADATA_FIELDS = frozenset({
    "name",
    "family",
    "module_name",
    "model_name",
    "parameters",
    "documentation",
    "provenance",
})


class ModelManifest:
    """Project-local model metadata loaded from ``models.toml``."""

    def __init__(self, entries=None, devices=None):
        self.entries = list(entries or [])
        self.devices = list(devices or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [entry.to_dict() for entry in self.entries],
            "devices": [device.to_dict() for device in self.devices],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelManifest":
        try:
            models = data.get("models", [])
        except AttributeError as exc:
            raise _invalid_manifest("manifest root must be a table") from exc
        unknown = sorted(key for key in data if key not in _MANIFEST_FIELDS)
        if unknown:
            raise _invalid_manifest(f"manifest has unknown fields: {', '.join(unknown)}")
        if not isinstance(models, list):
            raise _invalid_manifest("models must be an array")
        entries = []
        for index, item in enumerate(models):
            if not isinstance(item, dict):
                raise _invalid_manifest(f"models[{index}] must be a table")
            try:
                entries.append(ModelEntry.from_dict(item))
            except TypeError as exc:
                raise _invalid_manifest(f"models[{index}] has invalid fields: {exc}") from exc
            except KeyError as exc:
                raise _invalid_manifest(f"models[{index}] is missing {exc.args[0]}") from exc
        devices = data.get("devices", [])
        if not isinstance(devices, list):
            raise _invalid_manifest("devices must be an array")
        device_entries = []
        for index, item in enumerate(devices):
            if not isinstance(item, dict):
                raise _invalid_manifest(f"devices[{index}] must be a table")
            try:
                device_entries.append(DeviceMetadata.from_dict(item))
            except TypeError as exc:
                raise _invalid_manifest(f"devices[{index}] has invalid fields: {exc}") from exc
            except KeyError as exc:
                raise _invalid_manifest(f"devices[{index}] is missing {exc.args[0]}") from exc
        return cls(entries, device_entries)

    @classmethod
    def load(cls, path, project_path=None) -> "ModelManifest":
        path = Path(path)
        project_root = Path(project_path) if project_path is not None else path.parent
        with open(path, "rb") as file:
            data = tomllib.load(file)
        return cls.from_project_dict(data, project_root)

    def save(self, path, project_path=None) -> None:
        path = Path(path)
        project_root = Path(project_path) if project_path is not None else path.parent
        path.write_text(_manifest_toml(self.to_project_dict(project_root)))

    @classmethod
    def from_project_dict(cls, data: dict[str, Any], project_path) -> "ModelManifest":
        project_root = Path(project_path)
        manifest = cls.from_dict(data)
        return cls(
            (_resolve_entry_paths(entry, project_root) for entry in manifest.entries),
            manifest.devices,
        )

    def to_project_dict(self, project_path) -> dict[str, Any]:
        project_root = Path(project_path)
        return {
            "models": [
                _relativize_entry_paths(entry, project_root).to_dict()
                for entry in self.entries
            ],
            "devices": [device.to_dict() for device in self.devices],
        }

    def resolve(self, name=None, family=None, level=None, version=None) -> "ModelSelection":
        """Resolve manifest entries into concrete simulator inputs."""
        candidates = [
            entry for entry in self.entries
            if _matches(entry, name=name, family=family, level=level, version=version)
        ]
        if not candidates:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="model_selection_missing",
                    message="no model entry matches the requested selection",
                    context={
                        "name": name,
                        "family": family,
                        "level": level,
                        "version": version,
                    },
                )
            )
        if len(candidates) > 1 and name is None:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="model_selection_ambiguous",
                    message="multiple model entries match the requested selection",
                    context={
                        "family": family,
                        "level": level,
                        "version": version,
                        "candidates": [entry.to_dict() for entry in candidates],
                    },
                )
            )
        return ModelSelection(candidates).validate_files()

    def list_devices(self, family=None) -> list["DeviceMetadata"]:
        devices = list(self.devices)
        if family is not None:
            devices = [device for device in devices if device.family == family]
        return sorted(devices, key=lambda device: (device.family, device.name))

    def device(self, name: str) -> "DeviceMetadata":
        for device in self.devices:
            if device.name == name:
                return device
        raise ModelDiagnosticError(
            ModelDiagnostic(
                code="device_metadata_missing",
                message="device metadata not found",
                context={"name": name},
            )
        )


class DeviceMetadata:
    """Inspectable metadata for a reusable device/model family."""

    def __init__(
        self,
        name,
        family,
        module_name=None,
        model_name=None,
        parameters=None,
        documentation=None,
        provenance=None,
    ):
        self.name = name
        self.family = family
        self.module_name = module_name
        self.model_name = model_name
        self.parameters = dict(parameters or {})
        self.documentation = documentation
        self.provenance = dict(provenance or {})

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "family": self.family,
        }
        optional = {
            "module_name": self.module_name,
            "model_name": self.model_name,
            "parameters": self.parameters,
            "documentation": self.documentation,
            "provenance": self.provenance,
        }
        for key, value in optional.items():
            if value not in (None, {}, []):
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceMetadata":
        unknown = sorted(key for key in data if key not in _DEVICE_METADATA_FIELDS)
        if unknown:
            raise TypeError(f"unknown device metadata fields: {', '.join(unknown)}")
        return cls(
            name=data["name"],
            family=data["family"],
            module_name=data.get("module_name"),
            model_name=data.get("model_name"),
            parameters=data.get("parameters"),
            documentation=data.get("documentation"),
            provenance=data.get("provenance"),
        )


class ModelSelection:
    """Concrete model artifacts resolved from a manifest selection."""

    def __init__(self, entries):
        self.entries = list(entries)

    @property
    def osdi_paths(self) -> list[str]:
        return _dedupe(entry.osdi_path for entry in self.entries if entry.osdi_path)

    @property
    def includes(self) -> list[str]:
        return _dedupe(
            entry.model_file
            for entry in self.entries
            if entry.model_file and not entry.lib_section
        )

    @property
    def lib_sections(self) -> list[tuple[str, str]]:
        result = []
        seen = set()
        for entry in self.entries:
            if not entry.model_file or not entry.lib_section:
                continue
            item = (entry.model_file, entry.lib_section)
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "models": [entry.to_dict() for entry in self.entries],
            "osdi_paths": list(self.osdi_paths),
            "includes": list(self.includes),
            "lib_sections": [
                {"path": path, "section": section}
                for path, section in self.lib_sections
            ],
        }

    def apply_to_circuit(
        self,
        circuit,
        *,
        reference_mode: ReferenceMode = "concrete",
    ):
        """Add model references to a circuit.

        ``concrete`` emits simulator-ready include/.lib directives. ``logical``
        emits path-free Monata model-reference directives for generated
        artifacts. Logical output is a Monata carrier, not raw ngspice input.
        """
        if reference_mode == "logical":
            for entry in self.entries:
                if not entry.model_file:
                    continue
                circuit.model_ref(
                    techlib=entry.family,
                    corner=entry.name,
                    deck=entry.module_name,
                    section=entry.lib_section,
                    simulator="ngspice",
                )
            return circuit
        if reference_mode != "concrete":
            raise ValueError(f"unsupported model reference mode: {reference_mode}")
        for path in self.includes:
            circuit.include(path)
        for path, section in self.lib_sections:
            circuit.lib(path, section)
        return circuit

    def task_metadata(self) -> dict[str, Any]:
        return {"model_selection": self.metadata}

    def validate_files(self) -> "ModelSelection":
        """Validate concrete artifacts before runtime projection."""
        missing = []
        for entry in self.entries:
            for field in ("osdi_path", "model_file", "source_va"):
                path = getattr(entry, field)
                if path and not Path(path).exists():
                    missing.append({
                        "model": entry.name,
                        "field": field,
                        "path": path,
                    })
            for include_path in entry.include_paths:
                if include_path and not Path(include_path).exists():
                    missing.append({
                        "model": entry.name,
                        "field": "include_paths",
                        "path": include_path,
                    })
        if missing:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="model_artifact_missing",
                    message="model manifest references missing artifact files",
                    context={"missing": missing},
                )
            )
        return self


def _matches(entry: ModelEntry, name=None, family=None, level=None, version=None) -> bool:
    if name is not None and entry.name != name:
        return False
    if family is not None and entry.family != family:
        return False
    if level is not None and entry.level != level:
        return False
    if version is not None and entry.version != version:
        return False
    return True


def _dedupe(values) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _resolve_entry_paths(entry: ModelEntry, project_root: Path) -> ModelEntry:
    data = entry.to_dict()
    for field in _PATH_FIELDS:
        if data.get(field):
            data[field] = str(_resolve_project_path(project_root, data[field]))
    if data.get("include_paths"):
        data["include_paths"] = [
            str(_resolve_project_path(project_root, path))
            for path in data["include_paths"]
        ]
    return ModelEntry.from_dict(data)


def _relativize_entry_paths(entry: ModelEntry, project_root: Path) -> ModelEntry:
    data = entry.to_dict()
    for field in _PATH_FIELDS:
        if data.get(field):
            data[field] = _project_path(project_root, Path(data[field]))
    if data.get("include_paths"):
        data["include_paths"] = [
            _project_path(project_root, Path(path))
            for path in data["include_paths"]
        ]
    return ModelEntry.from_dict(data)


def _resolve_project_path(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _project_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _invalid_manifest(message: str) -> ModelDiagnosticError:
    return ModelDiagnosticError(
        ModelDiagnostic(
            code="model_manifest_invalid",
            message=message,
            context={},
        )
    )


def _manifest_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for model in data.get("models", []):
        _write_table(lines, "models", model)
    for device in data.get("devices", []):
        _write_table(lines, "devices", device)
    return "".join(lines)


def _write_table(lines: list[str], name: str, values: dict[str, Any]) -> None:
    lines.append(f"[[{name}]]\n")
    nested = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested[key] = value
        else:
            lines.append(f"{key} = {_toml_value(value)}\n")
    for key, child_values in nested.items():
        if child_values:
            lines.append(f"[{name}.{key}]\n")
            for nested_key, nested_value in child_values.items():
                lines.append(f"{nested_key} = {_toml_value(nested_value)}\n")
    lines.append("\n")


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return f'"{toml_string(value)}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    return f'"{toml_string(str(value))}"'
