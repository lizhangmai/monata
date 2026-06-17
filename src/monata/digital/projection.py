"""Digital simulation projection helpers for source-level PDK instances."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from monata.corner import CornerLike
from monata.netlist import Circuit, Element
from monata.projection import projection_scopes
from monata._types import ReferenceMode


class PdkProjectionOwner(Protocol):
    def project_pdk_instances(
        self,
        netlist: Circuit,
        registry: object | None = None,
        corner: CornerLike = None,
        reference_mode: ReferenceMode = "concrete",
        include_models: bool = True,
    ) -> Circuit:
        ...


@dataclass(frozen=True)
class PdkDeviceProjection:
    model: str
    pins: tuple[str, ...]
    kind: str = "M"
    params: Mapping[str, Any] = field(default_factory=dict)


class PdkModelProjectionLibrary:
    """Project source PDK instances to static primitive model elements.

    This helper is for non-techlib model mappings such as toy CMOS. Use
    ``Library.project_pdk_instances`` for registry-aware or corner-aware
    technology-library projection.
    """

    def __init__(
        self,
        *,
        lib: str,
        view: str,
        devices: Mapping[str, PdkDeviceProjection],
    ):
        self.lib = lib
        self.view = view
        self.devices = dict(devices)

    def project_pdk_instances(
        self,
        netlist,
        registry: object | None = None,
        corner: CornerLike = None,
        reference_mode: ReferenceMode = "concrete",
        include_models: bool = True,
    ):
        if registry is not None or corner is not None:
            raise ValueError(
                "PdkModelProjectionLibrary does not support registry or corner "
                "projection; use Library.project_pdk_instances for techlib projection"
            )
        if reference_mode != "concrete":
            raise ValueError("PdkModelProjectionLibrary only supports concrete projection")
        for scope in projection_scopes(netlist):
            self._project_scope(scope)
        return netlist

    def _project_scope(self, scope) -> None:
        for instance in tuple(scope.pdk_instances):
            if instance.lib != self.lib or instance.view != self.view:
                raise ValueError(
                    f"PDK model projection does not support "
                    f"{instance.lib}/{instance.cell}/{instance.view}"
                )
            try:
                projection = self.devices[instance.cell]
            except KeyError as exc:
                raise ValueError(
                    f"PDK model projection does not support "
                    f"{instance.lib}/{instance.cell}/{instance.view}"
                ) from exc
            try:
                nodes = tuple(instance.pins[pin] for pin in projection.pins)
            except KeyError as exc:
                raise ValueError(
                    f"PDK model projection missing pin {exc.args[0]!r} "
                    f"on {instance.lib}/{instance.cell}/{instance.view}"
                ) from exc
            scope.add(
                Element(
                    kind=projection.kind,
                    name=instance.name,
                    nodes=nodes,
                    model=projection.model,
                    params=_projected_params(projection, instance),
                )
            )
        scope.pdk_instances.clear()


def _projected_params(projection: PdkDeviceProjection, instance) -> OrderedDict[str, Any]:
    params: OrderedDict[str, Any] = OrderedDict(
        (str(param), value) for param, value in projection.params.items()
    )
    params.update((str(param), value) for param, value in instance.params.items())
    return params
