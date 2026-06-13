"""Technology-library TOML parsing helpers."""

from __future__ import annotations

from typing import Any

from monata._paths import validate_path_segment
from monata.corner import OperatingCorner
from monata.techlib.schema import (
    DeviceCell,
    DeviceView,
    ModelDeck,
    ParameterSpec,
    TechlibAttachment,
    TechlibError,
    _index_by_name,
    _validate_relative_path,
)


_CORNER_FIELDS = frozenset({
    "name",
    "model_deck",
    "section",
    "nominal_vdd",
    "process",
    "process_node",
    "flavor",
    "temperature",
    "voltages",
    "model_file",
    "device_defaults",
    "metadata",
})
_MODEL_DECK_FIELDS = frozenset({"name", "path", "description"})
_DEVICE_FIELDS = frozenset({"name", "kind", "pins", "params", "views"})
_PARAM_FIELDS = frozenset({"default", "type", "unit", "description"})
_VIEW_FIELDS = frozenset({
    "name",
    "primitive",
    "subckt",
    "model",
    "pin_order",
    "params",
    "model_deck",
    "corner_models",
})
_TECHLIB_ATTACHMENTS_FIELDS = frozenset({"techlibs", "default_corner", "corners"})
_TECHLIB_ATTACHMENT_FIELDS = frozenset({"name", "default_corner"})


def _reject_unknown_fields(
    data: dict[str, Any], allowed: frozenset[str], subject: str
) -> None:
    unknown = sorted(key for key in data if key not in allowed)
    if unknown:
        raise TechlibError(f"{subject} has unknown fields: {', '.join(unknown)}")


def parse_techlib_attachments(data: dict[str, Any] | None) -> list[TechlibAttachment]:
    if not data:
        return []
    _reject_unknown_fields(data, _TECHLIB_ATTACHMENTS_FIELDS, "techlib attachments")
    raw_techlibs = data.get("techlibs", [])
    if isinstance(raw_techlibs, str):
        raw_techlibs = [raw_techlibs]
    default_corner = data.get("default_corner")
    corners = data.get("corners", {})
    attachments = []
    for item in raw_techlibs:
        if isinstance(item, dict):
            name = item["name"]
            _reject_unknown_fields(
                item,
                _TECHLIB_ATTACHMENT_FIELDS,
                f"techlib attachment {name}",
            )
            corner = item.get("default_corner")
        else:
            name = str(item)
            corner = None
        attachments.append(
            TechlibAttachment(
                name=validate_path_segment(name, "techlib attachment"),
                default_corner=corner or corners.get(name) or default_corner,
            )
        )
    return attachments


def parse_model_deck(data: dict[str, Any]) -> ModelDeck:
    deck_name = data.get("name", "<unknown>")
    _reject_unknown_fields(data, _MODEL_DECK_FIELDS, f"model deck {deck_name}")
    return ModelDeck(
        name=validate_path_segment(data["name"], "model deck name"),
        path=_validate_relative_path(data["path"], "model deck path"),
        description=data.get("description"),
    )


def parse_corner(data: dict[str, Any]) -> OperatingCorner:
    corner_name = data.get("name", "<unknown>")
    _reject_unknown_fields(data, _CORNER_FIELDS, f"corner {corner_name}")
    return OperatingCorner(
        name=validate_path_segment(data["name"], "corner name"),
        model_deck=validate_path_segment(data["model_deck"], "corner model_deck"),
        section=data.get("section"),
        nominal_vdd=data.get("nominal_vdd"),
        process=data.get("process", data["name"]),
        process_node=data.get("process_node"),
        flavor=data.get("flavor"),
        temperature=data.get("temperature", 27),
        voltages=data.get("voltages", {}),
        model_file=data.get("model_file"),
        device_defaults=data.get("device_defaults", {}),
        metadata=data.get("metadata", {}),
    )


def parse_device(data: dict[str, Any]) -> DeviceCell:
    name = validate_path_segment(data["name"], "device name")
    _reject_unknown_fields(data, _DEVICE_FIELDS, f"device {name}")
    pins = tuple(str(pin) for pin in data.get("pins", ()))
    if not pins:
        raise TechlibError(f"device {name} must declare pins")
    params = {
        validate_path_segment(param_name, "parameter name"): _parse_param(param_name, param_data)
        for param_name, param_data in data.get("params", {}).items()
    }
    views = _index_by_name((_parse_view(view_data) for view_data in data.get("views", [])), "view")
    if not views:
        raise TechlibError(f"device {name} must declare at least one view")
    return DeviceCell(
        name=name,
        kind=str(data.get("kind", "device")),
        pins=pins,
        params=params,
        views=views,
    )


def _parse_param(name: str, data: dict[str, Any]) -> ParameterSpec:
    _reject_unknown_fields(data, _PARAM_FIELDS, f"parameter {name}")
    return ParameterSpec(
        name=validate_path_segment(name, "parameter name"),
        default=data.get("default"),
        type=data.get("type"),
        unit=data.get("unit"),
        description=data.get("description"),
    )


def _parse_view(data: dict[str, Any]) -> DeviceView:
    view_name = data.get("name", "<unknown>")
    _reject_unknown_fields(data, _VIEW_FIELDS, f"view {view_name}")
    primitive = str(data.get("primitive", "subckt"))
    pin_order = tuple(str(pin) for pin in data.get("pin_order", ()))
    if not pin_order:
        raise TechlibError("device view must declare pin_order")
    subckt = data.get("subckt")
    model = data.get("model")
    raw_corner_models = data.get("corner_models", {})
    if not isinstance(raw_corner_models, dict):
        raise TechlibError("corner_models must be a table")
    corner_models = {
        validate_path_segment(str(corner), "corner model name"): str(model_name)
        for corner, model_name in raw_corner_models.items()
    }
    if primitive == "subckt" and not subckt:
        raise TechlibError("subckt device view must declare subckt")
    if primitive == "mos" and not model and not corner_models:
        raise TechlibError("mos device view must declare model or corner_models")
    if primitive not in {"subckt", "mos", "symbol"}:
        raise TechlibError(f"unsupported device view primitive: {primitive}")
    return DeviceView(
        name=validate_path_segment(data["name"], "view name"),
        primitive=primitive,
        subckt=subckt,
        model=model,
        pin_order=pin_order,
        params=tuple(str(param) for param in data.get("params", ())),
        model_deck=data.get("model_deck"),
        corner_models=corner_models,
    )


__all__ = ["parse_corner", "parse_device", "parse_model_deck", "parse_techlib_attachments"]
