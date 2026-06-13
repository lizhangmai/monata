"""Technology-library schema records and shared validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from monata.corner import OperatingCorner


class TechlibError(ValueError):
    """Raised for invalid techlib metadata or unresolved PDK instances."""


@dataclass(frozen=True)
class TechlibDiscoveryError:
    """Diagnostic for a failed optional techlib entry point."""

    group: str
    entry_point: str
    message: str


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    default: Any = None
    type: str | None = None
    unit: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class DeviceView:
    name: str
    primitive: str
    pin_order: tuple[str, ...]
    subckt: str | None = None
    model: str | None = None
    params: tuple[str, ...] = ()
    model_deck: str | None = None
    corner_models: dict[str, str] = field(default_factory=dict)

    @property
    def is_projectable(self) -> bool:
        return self.primitive in {"subckt", "mos"}

    def model_for_corner(self, corner: OperatingCorner | None) -> str | None:
        if corner is not None and corner.name in self.corner_models:
            return self.corner_models[corner.name]
        return self.model


@dataclass(frozen=True)
class DeviceCell:
    name: str
    kind: str
    pins: tuple[str, ...]
    params: dict[str, ParameterSpec] = field(default_factory=dict)
    views: dict[str, DeviceView] = field(default_factory=dict)

    def view(self, name: str) -> DeviceView:
        try:
            return self.views[name]
        except KeyError as exc:
            raise TechlibError(f"unknown device view: {self.name}/{name}") from exc


@dataclass(frozen=True)
class ModelDeck:
    name: str
    path: str
    description: str | None = None

    def resolve_path(self, root: Path) -> Path:
        return _resolve_under_root(root, self.path, "model deck path")


@dataclass(frozen=True)
class TechlibAttachment:
    name: str
    default_corner: str | None = None


def _assert_projectable_view(view: Any) -> None:
    device_view = getattr(view, "view", view)
    if not device_view.is_projectable:
        raise TechlibError(
            f"device view {device_view.name} is source-only and cannot project to runtime IR"
        )


def _validate_relative_path(value: Any, label: str) -> str:
    text = str(value)
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise TechlibError(f"{label} must be relative to the techlib root: {text}")
    return text


def _resolve_under_root(root: Path, value: str, label: str) -> Path:
    relative = _validate_relative_path(value, label)
    root_resolved = root.resolve()
    path = (root_resolved / relative).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise TechlibError(f"{label} escapes techlib root: {value}") from exc
    return path


def _index_by_name(items: Iterable[Any], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if item.name in result:
            raise TechlibError(f"duplicate {label}: {item.name}")
        result[item.name] = item
    return result


__all__ = [
    "DeviceCell",
    "DeviceView",
    "ModelDeck",
    "ParameterSpec",
    "TechlibAttachment",
    "TechlibDiscoveryError",
    "TechlibError",
]
