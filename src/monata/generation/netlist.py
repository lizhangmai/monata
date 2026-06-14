"""Netlist view generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

from monata._types import NetlistProjectionMode
from monata.netlist import Circuit, SubCircuit, render_ngspice
from monata.views.declarative import schematic_view_to_circuit


def generate_netlist(
    cell,
    *,
    force: bool = False,
    output_format: str = "cir",
    format: str | None = None,
    projection: NetlistProjectionMode = "none",
    registry: Any = None,
    corner: Any = None,
) -> Path:
    if format is not None:
        if output_format != "cir":
            raise ValueError("use only one of format or output_format")
        warnings.warn(
            "generate_netlist(format=...) is deprecated; use output_format=...",
            DeprecationWarning,
            stacklevel=2,
        )
        output_format = format

    schematic_view = cell["schematic"]
    native_netlist = schematic_view_to_circuit(
        schematic_view,
        allow_trusted_python=True,
        reason="generate_netlist",
    )
    _apply_generation_projection(
        cell,
        native_netlist,
        projection=projection,
        registry=registry,
        corner=corner,
    )

    suffix = output_format.lstrip(".") or "cir"
    entry = f"netlist.{suffix}"
    return cell.write_generated_view(
        "netlist",
        entry=entry,
        content=render_ngspice(native_netlist),
        force=force,
        metadata={"format": "spice"},
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
