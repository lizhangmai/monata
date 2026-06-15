"""JSON v2 IO for structured schematic data."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from monata.schematic.data import Instance, InstanceRef, Net, Pin, Provenance, SchematicData

SCHEMA_VERSION = 2
_TOP_FIELDS = {"schema_version", "view_type", "cell", "interface", "nets", "instances", "provenance", "properties", "annotations"}
_CELL_FIELDS = {"name"}
_INTERFACE_FIELDS = {"pins"}
_PIN_FIELDS = {"name", "direction", "net", "order", "properties"}
_NET_FIELDS = {"name", "kind", "properties"}
_INSTANCE_FIELDS = {"name", "ref", "connections", "parameters", "nodes", "pin_order", "properties"}
_REF_FIELDS = {"kind", "lib", "cell", "view", "device", "model", "subckt"}
_PROVENANCE_FIELDS = {"kind", "source", "sha256", "metadata"}


def read_schematic(path: str | Path, *, default_cell: str | None = None) -> SchematicData:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_schematic(data, default_cell=default_cell)


def write_schematic(path: str | Path, schematic: SchematicData) -> Path:
    target = Path(path)
    target.write_text(dump_schematic(schematic), encoding="utf-8")
    return target


def load_schematic(payload: Mapping[str, Any], *, default_cell: str | None = None) -> SchematicData:
    data = _expect_mapping(payload, "schematic")
    _reject_unknown(data, _TOP_FIELDS, "schematic")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schematic.schema_version must be {SCHEMA_VERSION}")
    if data.get("view_type") != "schematic":
        raise ValueError("schematic.view_type must be 'schematic'")

    cell = _cell_name(data.get("cell"), default_cell=default_cell)
    pins = _pins(data.get("interface"))
    nets = _nets(data.get("nets", []))
    instances = _instances(data.get("instances", []))
    provenance = _provenance(data.get("provenance", []))
    properties = data.get("properties", {})
    annotations = data.get("annotations", [])
    if not isinstance(properties, Mapping):
        raise ValueError("schematic.properties must be an object")
    if not isinstance(annotations, list):
        raise ValueError("schematic.annotations must be an array")
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"schematic.annotations[{index}] must be an object")

    return SchematicData(
        cell=cell,
        pins=tuple(pins),
        nets=tuple(nets),
        instances=tuple(instances),
        provenance=tuple(provenance),
        properties=dict(properties),
        annotations=tuple(dict(item) for item in annotations),
    )


def dump_schematic(schematic: SchematicData | Mapping[str, Any]) -> str:
    data = schematic if isinstance(schematic, SchematicData) else load_schematic(schematic)
    return json.dumps(data.to_dict(), indent=2, ensure_ascii=False) + "\n"


def _cell_name(value: Any, *, default_cell: str | None) -> str:
    if value is None:
        if default_cell:
            return default_cell
        raise ValueError("schematic.cell is required")
    if isinstance(value, str):
        return value
    cell = _expect_mapping(value, "schematic.cell")
    _reject_unknown(cell, _CELL_FIELDS, "schematic.cell")
    name = cell.get("name", default_cell)
    if not isinstance(name, str) or not name:
        raise ValueError("schematic.cell.name is required")
    return name


def _pins(interface_value: Any) -> list[Pin]:
    interface = _expect_mapping(interface_value, "schematic.interface")
    _reject_unknown(interface, _INTERFACE_FIELDS, "schematic.interface")
    pins_value = interface.get("pins")
    if not isinstance(pins_value, list):
        raise ValueError("schematic.interface.pins must be an array")
    pins = []
    for index, value in enumerate(pins_value):
        item = _expect_mapping(value, f"schematic.interface.pins[{index}]")
        _reject_unknown(item, _PIN_FIELDS, f"schematic.interface.pins[{index}]")
        name = _string(item.get("name"), f"schematic.interface.pins[{index}].name")
        pins.append(
            Pin(
                name=name,
                direction=_string_field(item, "direction", f"schematic.interface.pins[{index}].direction", "inout"),
                net=_string_field(item, "net", f"schematic.interface.pins[{index}].net", name),
                order=_optional_int(item.get("order"), f"schematic.interface.pins[{index}].order"),
                properties=_mapping(item.get("properties", {}), f"schematic.interface.pins[{index}].properties"),
            )
        )
    return pins


def _nets(value: Any) -> list[Net]:
    if not isinstance(value, list):
        raise ValueError("schematic.nets must be an array")
    nets = []
    for index, raw in enumerate(value):
        item = _expect_mapping(raw, f"schematic.nets[{index}]")
        _reject_unknown(item, _NET_FIELDS, f"schematic.nets[{index}]")
        nets.append(
            Net(
                name=_string(item.get("name"), f"schematic.nets[{index}].name"),
                kind=_string_field(item, "kind", f"schematic.nets[{index}].kind", "signal"),
                properties=_mapping(item.get("properties", {}), f"schematic.nets[{index}].properties"),
            )
        )
    return nets


def _instances(value: Any) -> list[Instance]:
    if not isinstance(value, list):
        raise ValueError("schematic.instances must be an array")
    instances = []
    for index, raw in enumerate(value):
        item = _expect_mapping(raw, f"schematic.instances[{index}]")
        _reject_unknown(item, _INSTANCE_FIELDS, f"schematic.instances[{index}]")
        ref = _ref(item.get("ref"), index)
        instances.append(
            Instance(
                name=_string(item.get("name"), f"schematic.instances[{index}].name"),
                ref=ref,
                connections=_text_mapping(item.get("connections", {}), f"schematic.instances[{index}].connections"),
                parameters=_mapping(item.get("parameters", {}), f"schematic.instances[{index}].parameters"),
                nodes=tuple(_string(node, f"schematic.instances[{index}].nodes[]") for node in _string_list(item.get("nodes", []), f"schematic.instances[{index}].nodes")),
                pin_order=tuple(_string(pin, f"schematic.instances[{index}].pin_order[]") for pin in _string_list(item.get("pin_order", []), f"schematic.instances[{index}].pin_order")),
                properties=_mapping(item.get("properties", {}), f"schematic.instances[{index}].properties"),
            )
        )
    return instances


def _ref(value: Any, instance_index: int) -> InstanceRef:
    item = _expect_mapping(value, f"schematic.instances[{instance_index}].ref")
    _reject_unknown(item, _REF_FIELDS, f"schematic.instances[{instance_index}].ref")
    return InstanceRef(
        kind=_string(item.get("kind"), f"schematic.instances[{instance_index}].ref.kind"),
        lib=_optional_string(item.get("lib")),
        cell=_optional_string(item.get("cell")),
        view=_optional_string(item.get("view")),
        device=_optional_string(item.get("device")),
        model=_optional_string(item.get("model")),
        subckt=_optional_string(item.get("subckt")),
    )


def _provenance(value: Any) -> list[Provenance]:
    if not isinstance(value, list):
        raise ValueError("schematic.provenance must be an array")
    result = []
    for index, raw in enumerate(value):
        item = _expect_mapping(raw, f"schematic.provenance[{index}]")
        _reject_unknown(item, _PROVENANCE_FIELDS, f"schematic.provenance[{index}]")
        result.append(
            Provenance(
                kind=_string(item.get("kind"), f"schematic.provenance[{index}].kind"),
                source=_string(item.get("source"), f"schematic.provenance[{index}].source"),
                sha256=_optional_string(item.get("sha256")),
                metadata=_mapping(item.get("metadata", {}), f"schematic.provenance[{index}].metadata"),
            )
        )
    return result


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _text_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return {str(key): _string(item, f"{label}.{key}") for key, item in value.items()}


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): " + ", ".join(unknown))


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_field(data: Mapping[str, Any], key: str, label: str, default: str) -> str:
    if key not in data:
        return default
    return _string(data[key], label)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional string fields must be non-empty strings when provided")
    return value


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [_string(item, f"{label}[]") for item in value]
