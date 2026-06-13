"""CMOS convenience builders for techlib-backed transistor cells."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class TransistorParams:
    techlib: str
    nmos_cell: str
    pmos_cell: str
    w_n: str
    l_n: str
    w_p: str
    l_p: str
    power_node: str
    ground_node: str
    view: str = "ngspice"

    def with_overrides(self, **overrides: Any) -> "TransistorParams":
        valid = {field.name for field in fields(self)}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"unknown transistor parameter(s): {', '.join(unknown)}")
        return replace(self, **overrides)


def transistor_params(params: TransistorParams | None = None, **overrides: Any) -> TransistorParams:
    if params is None:
        return TransistorParams(**overrides)
    base = params
    return base.with_overrides(**overrides) if overrides else base


def add_nmos(scope: Any, name: str, drain: str, gate: str, source: str, bulk: str, params: TransistorParams) -> Any:
    return scope.pdk_instance(
        name,
        lib=params.techlib,
        cell=params.nmos_cell,
        view=params.view,
        pins={"d": drain, "g": gate, "s": source, "b": bulk},
        params={"w": params.w_n, "l": params.l_n},
    )


def add_pmos(scope: Any, name: str, drain: str, gate: str, source: str, bulk: str, params: TransistorParams) -> Any:
    return scope.pdk_instance(
        name,
        lib=params.techlib,
        cell=params.pmos_cell,
        view=params.view,
        pins={"d": drain, "g": gate, "s": source, "b": bulk},
        params={"w": params.w_p, "l": params.l_p},
    )


def add_inverter(scope: Any, prefix: str, input_node: str, output_node: str, params: TransistorParams) -> None:
    add_pmos(scope, f"{prefix}_p", output_node, input_node, params.power_node, params.power_node, params)
    add_nmos(scope, f"{prefix}_n", output_node, input_node, params.ground_node, params.ground_node, params)


def add_transmission_gate(
    scope: Any,
    prefix: str,
    input_node: str,
    output_node: str,
    control_n: str,
    control_p: str,
    params: TransistorParams,
) -> None:
    add_nmos(scope, f"{prefix}_n", output_node, control_n, input_node, params.ground_node, params)
    add_pmos(scope, f"{prefix}_p", output_node, control_p, input_node, params.power_node, params)


def add_nand2(scope: Any, prefix: str, a: str, b: str, output_node: str, params: TransistorParams) -> None:
    add_pmos(scope, f"{prefix}_p1", output_node, a, params.power_node, params.power_node, params)
    add_pmos(scope, f"{prefix}_p2", output_node, b, params.power_node, params.power_node, params)
    mid = f"{prefix}_mid"
    add_nmos(scope, f"{prefix}_n1", output_node, a, mid, params.ground_node, params)
    add_nmos(scope, f"{prefix}_n2", mid, b, params.ground_node, params.ground_node, params)


def add_nor2(scope: Any, prefix: str, a: str, b: str, output_node: str, params: TransistorParams) -> None:
    mid = f"{prefix}_mid"
    add_pmos(scope, f"{prefix}_p1", output_node, a, mid, params.power_node, params)
    add_pmos(scope, f"{prefix}_p2", mid, b, params.power_node, params.power_node, params)
    add_nmos(scope, f"{prefix}_n1", output_node, a, params.ground_node, params.ground_node, params)
    add_nmos(scope, f"{prefix}_n2", output_node, b, params.ground_node, params.ground_node, params)


__all__ = [
    "TransistorParams",
    "add_inverter",
    "add_nand2",
    "add_nmos",
    "add_nor2",
    "add_pmos",
    "add_transmission_gate",
    "transistor_params",
]
