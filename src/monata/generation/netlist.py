"""Netlist view generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monata._types import NetlistProjectionMode
from monata.netlist import Circuit, SubCircuit, render_ngspice


def generate_netlist(
    cell,
    *,
    force: bool = False,
    format: str = "cir",
    projection: NetlistProjectionMode = "none",
    registry: Any = None,
    corner: Any = None,
) -> Path:
    schematic_view = cell["schematic"]
    factory_cls = schematic_view.load()

    if isinstance(factory_cls, type):
        native_netlist = factory_cls()
    else:
        native_netlist = factory_cls

    if not isinstance(native_netlist, (Circuit, SubCircuit)):
        raise TypeError(
            "schematic must define a monata.netlist Circuit or SubCircuit; "
            "legacy schematic factories are no longer supported"
        )
    _apply_generation_projection(
        cell,
        native_netlist,
        projection=projection,
        registry=registry,
        corner=corner,
    )

    suffix = format.lstrip(".") or "cir"
    entry = f"netlist.{suffix}"
    return cell.write_generated_view(
        "netlist",
        entry=entry,
        content=render_ngspice(native_netlist),
        force=force,
    )


def _apply_generation_projection(
    cell,
    native_netlist: Circuit | SubCircuit,
    *,
    projection: NetlistProjectionMode,
    registry: Any,
    corner: Any,
) -> None:
    if projection == "none":
        return
    if projection not in {"logical", "concrete"}:
        raise ValueError(f"unsupported netlist projection mode: {projection}")

    from monata.projection import projection_context_for

    projection_context_for(cell.library).project_pdk_instances(
        native_netlist,
        registry=registry,
        corner=corner,
        reference_mode=projection,
    )
