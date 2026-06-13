"""PDK instance validation and projection records."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from monata._types import ReferenceMode
from monata.corner import OperatingCorner
from monata.models.manifest import ModelSelection
from monata.netlist.ir import Element, _assert_single_line
from monata.techlib.schema import (
    DeviceCell,
    DeviceView,
    TechlibError,
    _assert_projectable_view,
)

if TYPE_CHECKING:
    from monata.techlib.registry import Techlib


@dataclass(frozen=True)
class PDKInstance:
    """Source-level PDK cell instance identity."""

    name: str
    lib: str
    cell: str
    view: str
    pins: OrderedDict[str, str]
    params: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        for label, value in (
            ("PDK instance name", self.name),
            ("PDK library name", self.lib),
            ("PDK cell name", self.cell),
            ("PDK view name", self.view),
        ):
            if not value:
                raise TechlibError(f"{label} is required")
            _assert_single_line(value, label)
        for pin, net in self.pins.items():
            _assert_single_line(pin, f"{self.name} pin name")
            _assert_single_line(net, f"{self.name}.{pin} net")
        for param, value in self.params.items():
            _assert_single_line(param, f"{self.name} parameter name")
            _assert_single_line(value, f"{self.name}.{param} value")


@dataclass(frozen=True)
class ValidatedPDKInstance:
    instance: PDKInstance
    techlib: Techlib
    device: DeviceCell
    view: DeviceView
    ordered_nets: tuple[str, ...]
    projected_params: OrderedDict[str, Any]
    corner: OperatingCorner | None = None

    def to_element(self) -> Element:
        _assert_projectable_view(self)
        if self.view.primitive == "subckt":
            model = self.view.subckt or self.device.name
            return Element(
                kind="X",
                name=self.instance.name,
                nodes=self.ordered_nets,
                model=model,
                params=self.projected_params,
            )
        if self.view.primitive == "mos":
            model = self.view.model_for_corner(self.corner)
            if model is None:
                corner_name = self.corner.name if self.corner is not None else "<none>"
                raise TechlibError(
                    f"view {self.device.name}/{self.view.name} has no model for corner: {corner_name}"
                )
            return Element(
                kind="M",
                name=self.instance.name,
                nodes=self.ordered_nets,
                model=model,
                params=self.projected_params,
            )
        raise TechlibError(f"unsupported projection primitive: {self.view.primitive}")

    def model_selection(self) -> ModelSelection | None:
        if self.corner is None:
            return None
        return self.techlib.model_selection(self.corner)

    def project(self) -> "PDKProjection":
        _assert_projectable_view(self)
        return PDKProjection(
            source=self.instance,
            validated=self,
            element=self.to_element(),
            model_selection=self.model_selection(),
        )


@dataclass(frozen=True)
class PDKProjection:
    source: PDKInstance
    validated: ValidatedPDKInstance
    element: Element
    model_selection: ModelSelection | None = None

    def apply_to(
        self,
        scope: Any,
        include_models: bool = True,
        reference_mode: ReferenceMode = "concrete",
    ) -> Element:
        if include_models and self.model_selection is not None:
            self.model_selection.apply_to_circuit(scope, reference_mode=reference_mode)
        return scope.add(self.element)


def pdk_instance(
    name: str,
    *,
    lib: str,
    cell: str,
    view: str,
    pins: dict[str, str],
    params: dict[str, Any] | None = None,
) -> PDKInstance:
    return PDKInstance(
        name=name,
        lib=lib,
        cell=cell,
        view=view,
        pins=OrderedDict((str(pin), str(net)) for pin, net in pins.items()),
        params=OrderedDict((str(param), value) for param, value in (params or {}).items()),
    )


def _project_params(
    device: DeviceCell,
    view: DeviceView,
    instance: PDKInstance,
    corner: OperatingCorner | None = None,
) -> OrderedDict[str, Any]:
    result: OrderedDict[str, Any] = OrderedDict()
    corner_defaults = corner.defaults_for_device(device.name) if corner is not None else {}
    for param in view.params:
        if param in instance.params:
            result[param] = instance.params[param]
        elif param in corner_defaults:
            result[param] = corner_defaults[param]
        elif param in device.params and device.params[param].default is not None:
            result[param] = device.params[param].default
    for param, value in instance.params.items():
        if param not in result:
            result[param] = value
    return result


__all__ = [
    "PDKInstance",
    "PDKProjection",
    "ValidatedPDKInstance",
    "pdk_instance",
]
