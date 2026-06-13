"""Symbol view generation."""

from __future__ import annotations

from pathlib import Path

from monata._paths import toml_string
from monata.views.symbol import infer_pin_direction


def generate_symbol(cell, *, force: bool = False) -> Path:
    schematic_view = cell["schematic"]
    factory_cls = schematic_view.load()
    nodes = factory_cls.NODES

    lines = ['[symbol]\n', f'name = "{toml_string(cell.name)}"\n']
    for node_name in nodes:
        lines.append('\n[[pins]]\n')
        lines.append(f'name = "{toml_string(node_name)}"\n')
        lines.append(f'direction = "{toml_string(infer_pin_direction(node_name))}"\n')
    return cell.write_generated_view(
        "symbol",
        entry="symbol.toml",
        content="".join(lines),
        force=force,
    )
