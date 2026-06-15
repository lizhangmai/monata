"""Typed structured schematic records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from monata._json import json_safe_dict

JsonMap = Mapping[str, Any]


@dataclass(frozen=True)
class Pin:
    name: str
    direction: str = "inout"
    net: str | None = None
    order: int | None = None
    properties: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "pin.name"))
        object.__setattr__(self, "direction", _required_text(self.direction, f"pin {self.name}.direction"))
        net = self.name if self.net is None else self.net
        object.__setattr__(self, "net", _required_text(net, f"pin {self.name}.net"))
        if self.order is not None and (not isinstance(self.order, int) or isinstance(self.order, bool) or self.order < 0):
            raise ValueError(f"pin {self.name}.order must be a non-negative integer")
        object.__setattr__(self, "properties", _frozen_json_map(self.properties, f"pin {self.name}.properties"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "direction": self.direction,
            "net": self.net,
            "order": self.order,
        }
        if self.properties:
            result["properties"] = dict(self.properties)
        return result


@dataclass(frozen=True)
class Net:
    name: str
    kind: str = "signal"
    properties: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "net.name"))
        object.__setattr__(self, "kind", _required_text(self.kind, f"net {self.name}.kind"))
        object.__setattr__(self, "properties", _frozen_json_map(self.properties, f"net {self.name}.properties"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.properties:
            result["properties"] = dict(self.properties)
        return result


@dataclass(frozen=True)
class InstanceRef:
    kind: str
    lib: str | None = None
    cell: str | None = None
    view: str | None = None
    device: str | None = None
    model: str | None = None
    subckt: str | None = None

    def __post_init__(self) -> None:
        kind = _required_text(self.kind, "instance.ref.kind").lower()
        object.__setattr__(self, "kind", kind)
        for field_name in ("lib", "cell", "view", "device", "model", "subckt"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _required_text(value, f"instance.ref.{field_name}"))
        if kind == "pdk":
            _require(self.lib, "pdk instance ref requires lib")
            _require(self.cell, "pdk instance ref requires cell")
            _require(self.view, "pdk instance ref requires view")
        elif kind in {"subckt", "subcircuit", "instance", "x"}:
            _require(self.subckt or self.cell or self.model or self.device, "subckt instance ref requires subckt/cell/model")
        elif kind in {"mos", "mosfet", "nmos", "pmos", "resistor", "res", "r", "capacitor", "cap", "c", "inductor", "ind", "l", "voltage", "vsource", "v", "current", "isource", "i", "vpulse", "ipulse"}:
            pass
        else:
            raise ValueError(f"unsupported instance ref kind: {kind}")

    @property
    def subckt_name(self) -> str:
        name = self.subckt or self.cell or self.model or self.device
        if name is None:
            raise ValueError("subckt instance ref requires subckt/cell/model")
        return name

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        for key in ("lib", "cell", "view", "device", "model", "subckt"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class Instance:
    name: str
    ref: InstanceRef
    connections: JsonMap = field(default_factory=dict)
    parameters: JsonMap = field(default_factory=dict)
    nodes: tuple[str, ...] = ()
    pin_order: tuple[str, ...] = ()
    properties: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "instance.name"))
        if not isinstance(self.ref, InstanceRef):
            raise TypeError(f"instance {self.name}.ref must be an InstanceRef")
        object.__setattr__(self, "connections", _frozen_text_map(self.connections, f"instance {self.name}.connections"))
        object.__setattr__(self, "parameters", _frozen_json_map(self.parameters, f"instance {self.name}.parameters"))
        object.__setattr__(self, "nodes", tuple(_required_text(node, f"instance {self.name}.nodes[]") for node in self.nodes))
        object.__setattr__(
            self,
            "pin_order",
            tuple(_required_text(pin, f"instance {self.name}.pin_order[]") for pin in self.pin_order),
        )
        object.__setattr__(self, "properties", _frozen_json_map(self.properties, f"instance {self.name}.properties"))
        if not self.connections and not self.nodes:
            raise ValueError(f"instance {self.name} requires connections or nodes")
        if self.nodes and self.connections:
            raise ValueError(f"instance {self.name} cannot mix nodes and connections")

    def referenced_nets(self) -> tuple[str, ...]:
        return (*self.connections.values(), *self.nodes)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "ref": self.ref.to_dict(),
            "connections": dict(self.connections),
            "parameters": dict(self.parameters),
        }
        if self.nodes:
            result["nodes"] = list(self.nodes)
        if self.pin_order:
            result["pin_order"] = list(self.pin_order)
        if self.properties:
            result["properties"] = dict(self.properties)
        return result


@dataclass(frozen=True)
class Provenance:
    kind: str
    source: str
    sha256: str | None = None
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _required_text(self.kind, "provenance.kind"))
        object.__setattr__(self, "source", _required_text(self.source, "provenance.source"))
        if self.sha256 is not None:
            object.__setattr__(self, "sha256", _required_text(self.sha256, "provenance.sha256"))
        object.__setattr__(self, "metadata", _frozen_json_map(self.metadata, "provenance.metadata"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "source": self.source}
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class SchematicData:
    cell: str
    pins: tuple[Pin, ...]
    nets: tuple[Net, ...] = ()
    instances: tuple[Instance, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    properties: JsonMap = field(default_factory=dict)
    annotations: tuple[JsonMap, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _required_text(self.cell, "schematic.cell.name"))
        pins = _normalize_pins(self.pins)
        instances = _normalize_instances(self.instances)
        nets = _normalize_nets(self.nets) or _infer_nets(pins, instances)
        provenance = tuple(_coerce_provenance(item, index) for index, item in enumerate(self.provenance))
        annotations = tuple(_frozen_json_map(item, f"schematic.annotations[{index}]") for index, item in enumerate(self.annotations))

        _reject_duplicates((pin.name for pin in pins), "pin")
        _reject_duplicates((net.name for net in nets), "net")
        _reject_duplicates((instance.name for instance in instances), "instance")
        _validate_referenced_nets(pins, instances, nets)

        object.__setattr__(self, "pins", pins)
        object.__setattr__(self, "nets", nets)
        object.__setattr__(self, "instances", instances)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "properties", _frozen_json_map(self.properties, "schematic.properties"))
        object.__setattr__(self, "annotations", annotations)

    @property
    def pin_names(self) -> tuple[str, ...]:
        return tuple(pin.name for pin in self.pins)

    @property
    def net_names(self) -> tuple[str, ...]:
        return tuple(net.name for net in self.nets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "view_type": "schematic",
            "cell": {"name": self.cell},
            "interface": {"pins": [pin.to_dict() for pin in self.pins]},
            "nets": [net.to_dict() for net in self.nets],
            "instances": [instance.to_dict() for instance in self.instances],
            "provenance": [item.to_dict() for item in self.provenance],
            "properties": dict(self.properties),
            "annotations": [dict(item) for item in self.annotations],
        }


def _normalize_pins(values: Iterable[Pin | Mapping[str, Any]]) -> tuple[Pin, ...]:
    pins = []
    for index, value in enumerate(values):
        pin = value if isinstance(value, Pin) else Pin(**dict(value))
        if pin.order is None:
            pin = Pin(pin.name, direction=pin.direction, net=pin.net, order=index, properties=pin.properties)
        pins.append(pin)
    if not pins:
        raise ValueError("schematic.interface.pins is required")
    return tuple(pins)


def _normalize_nets(values: Iterable[Net | Mapping[str, Any]]) -> tuple[Net, ...]:
    return tuple(value if isinstance(value, Net) else Net(**dict(value)) for value in values)


def _normalize_instances(values: Iterable[Instance | Mapping[str, Any]]) -> tuple[Instance, ...]:
    return tuple(_coerce_instance(value, index) for index, value in enumerate(values))


def _coerce_instance(value: Instance | Mapping[str, Any], index: int) -> Instance:
    if isinstance(value, Instance):
        return value
    data = dict(value)
    ref = data.get("ref")
    if not isinstance(ref, InstanceRef):
        if not isinstance(ref, Mapping):
            raise ValueError(f"schematic.instances[{index}].ref must be an object")
        data["ref"] = InstanceRef(**dict(ref))
    return Instance(**data)


def _coerce_provenance(value: Provenance | Mapping[str, Any], index: int) -> Provenance:
    if isinstance(value, Provenance):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"schematic.provenance[{index}] must be an object")
    return Provenance(**dict(value))


def _infer_nets(pins: Sequence[Pin], instances: Sequence[Instance]) -> tuple[Net, ...]:
    names = [str(pin.net or pin.name) for pin in pins]
    for instance in instances:
        for net in instance.referenced_nets():
            if net not in names:
                names.append(net)
    return tuple(Net(name) for name in names)


def _validate_referenced_nets(pins: Sequence[Pin], instances: Sequence[Instance], nets: Sequence[Net]) -> None:
    known = {net.name for net in nets}
    missing = [str(pin.net or pin.name) for pin in pins if str(pin.net or pin.name) not in known]
    for instance in instances:
        missing.extend(net for net in instance.referenced_nets() if net not in known)
    if missing:
        raise ValueError("schematic.nets is missing referenced net(s): " + ", ".join(sorted(set(missing))))


def _reject_duplicates(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            duplicates.append(value)
        seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate {label} name(s): " + ", ".join(duplicates))


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a single line")
    return value


def _frozen_text_map(value: Mapping[str, Any], label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return MappingProxyType({str(key): _required_text(item, f"{label}.{key}") for key, item in value.items()})


def _frozen_json_map(value: Mapping[str, Any], label: str) -> JsonMap:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    try:
        safe = json_safe_dict(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be JSON-compatible") from exc
    return MappingProxyType(safe)


def _require(value: Any, message: str) -> None:
    if value is None:
        raise ValueError(message)
