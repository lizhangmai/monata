"""Projection from schematic data into native netlist IR."""

from __future__ import annotations

from typing import Any

from monata.netlist import SubCircuit
from monata.schematic.data import Instance, SchematicData


def schematic_to_subcircuit(schematic: SchematicData) -> SubCircuit:
    circuit = SubCircuit(name=schematic.cell, nodes=schematic.pin_names)
    for instance in schematic.instances:
        _add_instance(circuit, instance)
    circuit.ensure_built()
    return circuit


def _add_instance(scope: SubCircuit, instance: Instance) -> None:
    ref = instance.ref
    params = dict(instance.parameters)
    kind = ref.kind
    connections = dict(instance.connections)

    if kind == "pdk":
        scope.pdk_instance(
            instance.name,
            lib=str(ref.lib),
            cell=str(ref.cell),
            view=str(ref.view),
            pins=connections,
            params=params,
        )
        return
    if kind in {"subckt", "subcircuit", "instance", "x"}:
        if instance.nodes:
            scope.instance(instance.name, instance.nodes, ref.subckt_name, **params)
        else:
            scope.instance_pins(instance.name, ref.subckt_name, connections, pin_order=instance.pin_order or None, **params)
        return
    if kind in {"mos", "mosfet", "nmos", "pmos"}:
        scope.mos(
            instance.name,
            d=connections["d"],
            g=connections["g"],
            s=connections["s"],
            b=connections["b"],
            model=str(ref.model or ref.device or kind),
            **params,
        )
        return
    if kind in {"resistor", "res", "r"}:
        scope.resistor(instance.name, connections["n1"], connections["n2"], _pop_value(instance, params), **params)
        return
    if kind in {"capacitor", "cap", "c"}:
        scope.capacitor(instance.name, connections["n1"], connections["n2"], _pop_value(instance, params), **params)
        return
    if kind in {"inductor", "ind", "l"}:
        scope.inductor(instance.name, connections["n1"], connections["n2"], _pop_value(instance, params), **params)
        return
    if kind in {"voltage", "vsource", "v"}:
        p, n = _source_nodes(instance)
        scope.voltage(instance.name, p, n, _pop_value(instance, params), **params)
        return
    if kind in {"current", "isource", "i"}:
        p, n = _source_nodes(instance)
        scope.current(instance.name, p, n, _pop_value(instance, params), **params)
        return
    if kind in {"vpulse", "ipulse"}:
        p, n = _source_nodes(instance)
        values = _pulse_values(instance)
        if kind == "vpulse":
            scope.vpulse(instance.name, p, n, *values, **params)
        else:
            scope.ipulse(instance.name, p, n, *values, **params)
        return
    raise ValueError(f"unsupported schematic instance ref kind for {instance.name!r}: {kind}")


def _pop_value(instance: Instance, params: dict[str, Any]) -> Any:
    try:
        return params.pop("value")
    except KeyError as exc:
        raise ValueError(f"schematic instance {instance.name!r} is missing value") from exc


def _pulse_values(instance: Instance) -> list[Any]:
    value = instance.parameters.get("values", instance.parameters.get("value"))
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"schematic instance {instance.name!r} requires 7 pulse values")
    return list(value)


def _source_nodes(instance: Instance) -> tuple[str, str]:
    connections = instance.connections
    if "p" in connections and "n" in connections:
        return connections["p"], connections["n"]
    if "node" in connections and "ref" in connections:
        return connections["node"], connections["ref"]
    raise ValueError(f"schematic source instance {instance.name!r} requires p/n or node/ref")
