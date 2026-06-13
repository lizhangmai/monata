"""Source-netlist construction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from monata.parser.source_subcircuit import parse_source_subcircuit as _parse_source_subcircuit


def source_subcircuit_ports(
    source_file: str | Path | None = None,
    *,
    source_root: str | Path | None = None,
    filename: str | None = None,
) -> tuple[str, ...]:
    return _parse_source_subcircuit(_source_subcircuit_path(source_file, source_root=source_root, filename=filename)).ports


def build_source_subcircuit_instances(
    scope: Any,
    source_file: str | Path | None = None,
    *,
    source_root: str | Path | None = None,
    filename: str | None = None,
    expected_name: str,
    expected_ports: Iterable[str],
    expected_count: int,
    allowed_kinds: Iterable[str],
) -> None:
    path = _source_subcircuit_path(source_file, source_root=source_root, filename=filename)
    netlist = _parse_source_subcircuit(path)
    ports = tuple(expected_ports)
    allowed = set(allowed_kinds)

    if netlist.name != expected_name:
        raise ValueError(f"{path} defines {netlist.name}, expected {expected_name}")
    if netlist.ports != ports:
        raise ValueError(f"{path} ports do not match {expected_name}")
    if len(netlist.instances) != expected_count:
        raise ValueError(f"{path} has {len(netlist.instances)} instances, expected {expected_count}")

    unknown_kinds = sorted({instance.kind for instance in netlist.instances} - allowed)
    if unknown_kinds:
        raise ValueError(f"{path} references unsupported subcircuits: {unknown_kinds}")

    for instance in netlist.instances:
        scope.instance(instance.name, instance.nodes, instance.kind)


def _source_subcircuit_path(
    source_file: str | Path | None,
    *,
    source_root: str | Path | None,
    filename: str | None,
) -> Path:
    if source_file is not None:
        if source_root is not None or filename is not None:
            raise ValueError("source_file cannot be combined with source_root or filename")
        return Path(source_file)
    if source_root is None or filename is None:
        raise ValueError("source_file or source_root plus filename is required")
    return Path(source_root) / filename


__all__ = [
    "build_source_subcircuit_instances",
    "source_subcircuit_ports",
]
