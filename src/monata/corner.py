"""Canonical operating-corner model shared by techlib and simulation code."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from monata._paths import validate_path_segment

CORNER_SCHEMA = "monata.operating_corner.v1"
_MODEL_SECTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_OPERATING_POINT_PAYLOAD_FIELDS = frozenset({"name", "temperature", "voltages"})
_MODEL_CORNER_PAYLOAD_FIELDS = frozenset({
    "techlib",
    "model_deck",
    "section",
    "model_file",
    "nominal_vdd",
    "process",
    "process_node",
    "flavor",
    "device_defaults",
    "metadata",
})
_OPERATING_CORNER_PAYLOAD_FIELDS = (
    frozenset({"schema"})
    | _OPERATING_POINT_PAYLOAD_FIELDS
    | _MODEL_CORNER_PAYLOAD_FIELDS
)


@dataclass(frozen=True, init=False)
class OperatingPoint:
    """Simulator operating conditions independent of process/model metadata."""

    name: str
    temperature: float
    voltages: Mapping[str, float]

    def __init__(
        self,
        name: str,
        temperature: float = 27,
        *,
        voltages: Mapping[str, float] | None = None,
    ) -> None:
        object.__setattr__(self, "name", validate_path_segment(name, "operating point name"))
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "voltages", _voltages(voltages))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "temperature": self.temperature,
            "voltages": dict(self.voltages),
        }
        return {key: value for key, value in payload.items() if value not in ({}, [])}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatingPoint":
        if "name" not in payload:
            raise ValueError("operating point payload is missing name")
        _reject_unknown_fields(payload, _OPERATING_POINT_PAYLOAD_FIELDS, "operating point")
        voltages = payload.get("voltages", {})
        return cls(
            name=str(payload["name"]),
            temperature=float(payload.get("temperature", 27)),
            voltages=_voltages(voltages if isinstance(voltages, Mapping) else {}),
        )


@dataclass(frozen=True, init=False)
class ModelCornerRef:
    """Process and model-deck metadata associated with an operating corner."""

    process: str | None
    model_file: str | None
    techlib: str | None
    model_deck: str | None
    section: str | None
    nominal_vdd: float | None
    process_node: str | None
    flavor: str | None
    device_defaults: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        process: str | None = None,
        model_file: str | Path | None = None,
        *,
        techlib: str | None = None,
        model_deck: str | None = None,
        section: str | None = None,
        nominal_vdd: float | None = None,
        process_node: str | None = None,
        flavor: str | None = None,
        device_defaults: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "process", process)
        object.__setattr__(self, "model_file", str(model_file) if model_file is not None else None)
        object.__setattr__(self, "techlib", techlib)
        object.__setattr__(self, "model_deck", model_deck)
        object.__setattr__(self, "section", validate_model_section(section))
        object.__setattr__(self, "nominal_vdd", float(nominal_vdd) if nominal_vdd is not None else None)
        object.__setattr__(self, "process_node", process_node)
        object.__setattr__(self, "flavor", flavor)
        object.__setattr__(self, "device_defaults", _device_defaults(device_defaults))
        object.__setattr__(self, "metadata", _read_only_mapping(metadata))

    def defaults_for_device(self, device_name: str) -> dict[str, Any]:
        return dict(self.device_defaults.get(device_name, {}))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "techlib": self.techlib,
            "model_deck": self.model_deck,
            "section": self.section,
            "model_file": self.model_file,
            "nominal_vdd": self.nominal_vdd,
            "process": self.process,
            "process_node": self.process_node,
            "flavor": self.flavor,
            "device_defaults": {
                device: dict(defaults)
                for device, defaults in self.device_defaults.items()
            },
            "metadata": dict(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelCornerRef":
        _reject_unknown_fields(payload, _MODEL_CORNER_PAYLOAD_FIELDS, "model corner")
        return cls(
            process=payload.get("process"),
            model_file=payload.get("model_file"),
            techlib=payload.get("techlib"),
            model_deck=payload.get("model_deck"),
            section=payload.get("section"),
            nominal_vdd=payload.get("nominal_vdd"),
            process_node=payload.get("process_node"),
            flavor=payload.get("flavor"),
            device_defaults=payload.get("device_defaults", {}),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, init=False)
class OperatingCorner:
    """A simulator operating point plus optional process/model metadata."""

    name: str
    temperature: float
    voltages: Mapping[str, float]
    process: str | None
    model_file: str | None
    techlib: str | None
    model_deck: str | None
    section: str | None
    nominal_vdd: float | None
    process_node: str | None
    flavor: str | None
    device_defaults: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        name: str,
        temperature: float = 27,
        *,
        voltages: Mapping[str, float] | None = None,
        process: str | None = None,
        model_file: str | Path | None = None,
        techlib: str | None = None,
        model_deck: str | None = None,
        section: str | None = None,
        nominal_vdd: float | None = None,
        process_node: str | None = None,
        flavor: str | None = None,
        device_defaults: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "name", validate_path_segment(name, "corner name"))
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "voltages", _voltages(voltages))
        object.__setattr__(self, "process", process)
        object.__setattr__(self, "model_file", str(model_file) if model_file is not None else None)
        object.__setattr__(self, "techlib", techlib)
        object.__setattr__(self, "model_deck", model_deck)
        object.__setattr__(self, "section", validate_model_section(section))
        object.__setattr__(self, "nominal_vdd", float(nominal_vdd) if nominal_vdd is not None else None)
        object.__setattr__(self, "process_node", process_node)
        object.__setattr__(self, "flavor", flavor)
        object.__setattr__(self, "device_defaults", _device_defaults(device_defaults))
        object.__setattr__(self, "metadata", _read_only_mapping(metadata))

    @property
    def operating_point(self) -> OperatingPoint:
        return OperatingPoint(
            name=self.name,
            temperature=self.temperature,
            voltages=self.voltages,
        )

    @property
    def model_corner(self) -> ModelCornerRef:
        return ModelCornerRef(
            process=self.process,
            model_file=self.model_file,
            techlib=self.techlib,
            model_deck=self.model_deck,
            section=self.section,
            nominal_vdd=self.nominal_vdd,
            process_node=self.process_node,
            flavor=self.flavor,
            device_defaults=self.device_defaults,
            metadata=self.metadata,
        )

    def defaults_for_device(self, device_name: str) -> dict[str, Any]:
        return self.model_corner.defaults_for_device(device_name)

    def with_updates(self, **updates: Any) -> "OperatingCorner":
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": CORNER_SCHEMA,
            **self.operating_point.to_dict(),
            **self.model_corner.to_dict(),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatingCorner":
        if "name" not in payload:
            raise ValueError("corner payload is missing name")
        _reject_unknown_fields(payload, _OPERATING_CORNER_PAYLOAD_FIELDS, "corner")
        voltages = payload.get("voltages", {})
        metadata = dict(payload.get("metadata", {}))
        return cls(
            name=str(payload["name"]),
            temperature=float(payload.get("temperature", 27)),
            voltages=_voltages(voltages if isinstance(voltages, Mapping) else {}),
            process=payload.get("process"),
            model_file=payload.get("model_file"),
            techlib=payload.get("techlib"),
            model_deck=payload.get("model_deck"),
            section=payload.get("section"),
            nominal_vdd=payload.get("nominal_vdd"),
            process_node=payload.get("process_node"),
            flavor=payload.get("flavor"),
            device_defaults=payload.get("device_defaults", {}),
            metadata=metadata,
        )

    @classmethod
    def from_parts(
        cls,
        operating_point: OperatingPoint,
        model_corner: ModelCornerRef | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "OperatingCorner":
        model = model_corner or ModelCornerRef()
        merged_metadata = dict(model.metadata)
        merged_metadata.update(metadata or {})
        return cls(
            name=operating_point.name,
            temperature=operating_point.temperature,
            voltages=operating_point.voltages,
            process=model.process,
            model_file=model.model_file,
            techlib=model.techlib,
            model_deck=model.model_deck,
            section=model.section,
            nominal_vdd=model.nominal_vdd,
            process_node=model.process_node,
            flavor=model.flavor,
            device_defaults=model.device_defaults,
            metadata=merged_metadata,
        )


CornerLike = OperatingCorner | OperatingPoint | str | Mapping[str, Any] | None


def validate_model_section(section: str | None) -> str | None:
    """Validate a simulator model library section token."""

    if section is None:
        return None
    if not isinstance(section, str):
        raise ValueError("corner model section must be a string")
    if not section:
        raise ValueError("corner model section must be non-empty")
    if not _MODEL_SECTION_RE.fullmatch(section):
        raise ValueError(
            "corner model section must be a single token using letters, digits, "
            "underscore, dot, colon, or hyphen"
        )
    return section


def coerce_operating_corner(value: CornerLike | object) -> OperatingCorner | None:
    if value is None:
        return None
    if isinstance(value, OperatingCorner):
        return value
    if isinstance(value, OperatingPoint):
        return OperatingCorner.from_parts(value)
    if isinstance(value, str):
        return OperatingCorner(value)
    if isinstance(value, Mapping):
        return OperatingCorner.from_dict(value)
    raise TypeError(f"unsupported operating corner: {type(value).__name__}")


def corner_to_payload(corner: object) -> dict[str, Any] | None:
    resolved = coerce_operating_corner(corner)
    return resolved.to_dict() if resolved is not None else None


def corner_from_payload(payload: Mapping[str, Any] | None) -> OperatingCorner | None:
    return OperatingCorner.from_dict(payload) if payload is not None else None


def _voltages(values: Mapping[str, float] | None) -> Mapping[str, float]:
    return MappingProxyType({str(key): float(value) for key, value in dict(values or {}).items()})


def _device_defaults(values: Mapping[str, Mapping[str, Any]] | None) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType({
        str(device): MappingProxyType({str(param): value for param, value in dict(defaults).items()})
        for device, defaults in dict(values or {}).items()
    })


def _read_only_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(key for key in payload if key not in allowed)
    if unknown:
        raise ValueError(f"{label} payload has unknown fields: {', '.join(unknown)}")
