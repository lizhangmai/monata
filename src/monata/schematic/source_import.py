"""Helpers for importing parsed source subcircuits as structured data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from monata.parser import parse_source_subcircuit
from monata.schematic.builder import SchematicBuilder
from monata.schematic.data import SchematicData


def schematic_from_source_subcircuit(
    source_file: str | Path,
    *,
    cell: str | None = None,
    expected_name: str | None = None,
    expected_ports: Iterable[str] | None = None,
    expected_count: int | None = None,
    allowed_kinds: Iterable[str] | None = None,
) -> SchematicData:
    source = parse_source_subcircuit(source_file)
    if expected_name is not None and source.name != expected_name:
        raise ValueError(f"{source.path} contains subckt {source.name!r}, expected {expected_name!r}")
    ports = tuple(expected_ports or source.ports)
    if tuple(source.ports) != ports:
        raise ValueError(f"{source.path} ports {source.ports!r} do not match expected {ports!r}")
    if expected_count is not None and len(source.instances) != expected_count:
        raise ValueError(f"{source.path} has {len(source.instances)} instances, expected {expected_count}")
    allowed = {kind for kind in allowed_kinds or ()}
    if allowed:
        bad = sorted({instance.kind for instance in source.instances if instance.kind not in allowed})
        if bad:
            raise ValueError(f"{source.path} references unsupported subcircuits: {', '.join(bad)}")

    path = Path(source.path)
    builder = SchematicBuilder(cell or source.name)
    for port in ports:
        builder.pin(port)
    for instance in source.instances:
        builder.subckt_instance(instance.name, instance.kind, nodes=instance.nodes)
    builder.provenance(
        "source-subcircuit",
        str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata={"source_name": source.name, "instance_count": len(source.instances)},
    )
    return builder.build()
