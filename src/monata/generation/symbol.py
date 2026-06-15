"""Symbol view generation."""

from __future__ import annotations

import json
from pathlib import Path

from monata.views.symbol import infer_pin_direction
from monata.views.declarative import schematic_pin_names


def generate_symbol(cell, *, force: bool = False) -> Path:
    schematic_view = cell["schematic"]
    nodes = schematic_pin_names(
        schematic_view,
        reason="generate_symbol",
    )
    payload = {
        "schema_version": 1,
        "view_type": "symbol",
        "name": cell.name,
        "pins": [
            {
                "name": node_name,
                "side": _pin_side(node_name),
                "direction": infer_pin_direction(node_name),
            }
            for node_name in nodes
        ],
    }
    return cell.write_generated_view(
        "symbol",
        entry="symbol.monata.json",
        content=json.dumps(payload, indent=2) + "\n",
        force=force,
        metadata={"format": "monata-symbol-json", "schema_version": 1},
    )


def _pin_side(pin_name: str) -> str:
    direction = infer_pin_direction(pin_name)
    lower = pin_name.lower()
    if "vdd" in lower or "vcc" in lower:
        return "top"
    if "vss" in lower or "gnd" in lower:
        return "bottom"
    if direction == "output":
        return "right"
    return "left"
