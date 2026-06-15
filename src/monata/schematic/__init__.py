"""Structured schematic data view API."""

from __future__ import annotations

from monata.schematic.builder import SchematicBuilder
from monata.schematic.data import Instance, InstanceRef, Net, Pin, Provenance, SchematicData
from monata.schematic.json_io import SCHEMA_VERSION, dump_schematic, load_schematic, read_schematic, write_schematic
from monata.schematic.source_import import schematic_from_source_subcircuit
from monata.schematic.to_netlist import schematic_to_subcircuit

__all__ = [
    "SCHEMA_VERSION",
    "Instance",
    "InstanceRef",
    "Net",
    "Pin",
    "Provenance",
    "SchematicBuilder",
    "SchematicData",
    "dump_schematic",
    "load_schematic",
    "read_schematic",
    "schematic_from_source_subcircuit",
    "schematic_to_subcircuit",
    "write_schematic",
]
