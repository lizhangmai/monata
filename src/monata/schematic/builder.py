"""Python-friendly builder that emits structured schematic data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from monata.schematic.data import Instance, InstanceRef, Net, Pin, Provenance, SchematicData
from monata.schematic.json_io import write_schematic


class SchematicBuilder:
    """Chainable authoring helper for `monata-schematic-json` v2 data."""

    def __init__(self, cell: str):
        self.cell = cell
        self._pins: list[Pin] = []
        self._nets: dict[str, Net] = {}
        self._instances: list[Instance] = []
        self._provenance: list[Provenance] = []
        self._properties: dict[str, Any] = {}
        self._annotations: list[dict[str, Any]] = []

    def pin(
        self,
        name: str,
        *,
        direction: str = "inout",
        net: str | None = None,
        properties: Mapping[str, Any] | None = None,
    ) -> "SchematicBuilder":
        pin = Pin(name=name, direction=direction, net=net or name, order=len(self._pins), properties=dict(properties or {}))
        self._pins.append(pin)
        self.net(pin.net or pin.name)
        return self

    def net(
        self,
        name: str,
        *,
        kind: str = "signal",
        properties: Mapping[str, Any] | None = None,
    ) -> "SchematicBuilder":
        if name not in self._nets:
            self._nets[name] = Net(name=name, kind=kind, properties=dict(properties or {}))
        return self

    def pdk_instance(
        self,
        name: str,
        *,
        lib: str,
        cell: str,
        view: str,
        pins: Mapping[str, str],
        parameters: Mapping[str, Any] | None = None,
    ) -> "SchematicBuilder":
        return self.instance(
            name,
            ref=InstanceRef(kind="pdk", lib=lib, cell=cell, view=view),
            connections=pins,
            parameters=parameters,
        )

    def subckt_instance(
        self,
        name: str,
        subckt: str,
        *,
        pins: Mapping[str, str] | None = None,
        nodes: tuple[str, ...] | list[str] = (),
        pin_order: tuple[str, ...] | list[str] = (),
        parameters: Mapping[str, Any] | None = None,
    ) -> "SchematicBuilder":
        return self.instance(
            name,
            ref=InstanceRef(kind="subckt", subckt=subckt),
            connections=dict(pins or {}),
            nodes=tuple(nodes),
            pin_order=tuple(pin_order),
            parameters=parameters,
        )

    def primitive(
        self,
        name: str,
        kind: str,
        *,
        connections: Mapping[str, str],
        value: Any | None = None,
        model: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> "SchematicBuilder":
        params = dict(parameters or {})
        if value is not None:
            params.setdefault("value", value)
        return self.instance(
            name,
            ref=InstanceRef(kind=kind, model=model, device=kind),
            connections=connections,
            parameters=params,
        )

    def instance(
        self,
        name: str,
        *,
        ref: InstanceRef,
        connections: Mapping[str, str] | None = None,
        parameters: Mapping[str, Any] | None = None,
        nodes: tuple[str, ...] | list[str] = (),
        pin_order: tuple[str, ...] | list[str] = (),
        properties: Mapping[str, Any] | None = None,
    ) -> "SchematicBuilder":
        instance = Instance(
            name=name,
            ref=ref,
            connections=dict(connections or {}),
            parameters=dict(parameters or {}),
            nodes=tuple(nodes),
            pin_order=tuple(pin_order),
            properties=dict(properties or {}),
        )
        self._instances.append(instance)
        for net in instance.referenced_nets():
            self.net(net)
        return self

    def provenance(
        self,
        kind: str,
        source: str,
        *,
        sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SchematicBuilder":
        self._provenance.append(Provenance(kind=kind, source=source, sha256=sha256, metadata=dict(metadata or {})))
        return self

    def property(self, name: str, value: Any) -> "SchematicBuilder":
        self._properties[name] = value
        return self

    def annotation(self, **values: Any) -> "SchematicBuilder":
        self._annotations.append(dict(values))
        return self

    def build(self) -> SchematicData:
        return SchematicData(
            cell=self.cell,
            pins=tuple(self._pins),
            nets=tuple(self._nets.values()),
            instances=tuple(self._instances),
            provenance=tuple(self._provenance),
            properties=dict(self._properties),
            annotations=tuple(self._annotations),
        )

    def write(self, path: str | Path) -> Path:
        return write_schematic(path, self.build())
