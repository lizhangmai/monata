"""Source-subcircuit import helpers used by project-level construction code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class SourceSubcircuitInstance:
    name: str
    nodes: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class SourceSubcircuit:
    path: Path
    name: str
    ports: tuple[str, ...]
    instances: tuple[SourceSubcircuitInstance, ...]


def parse_source_subcircuit(source_file: str | Path) -> SourceSubcircuit:
    path = Path(source_file)
    text = path.read_text()

    subckt = re.search(r"^\.?subckt\s+(\S+)\s+(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if subckt is None:
        raise ValueError(f"{path} does not contain a subckt declaration")

    instances = []
    for line in text.splitlines():
        match = re.match(r"\s*(I\d+)\s+\(([^)]*)\)\s+(\S+)\s*$", line)
        if match:
            instances.append(
                SourceSubcircuitInstance(
                    name=match.group(1),
                    nodes=tuple(match.group(2).split()),
                    kind=match.group(3),
                )
            )

    if not instances:
        raise ValueError(f"{path} does not contain source instances")

    return SourceSubcircuit(
        path=path,
        name=subckt.group(1),
        ports=tuple(subckt.group(2).split()),
        instances=tuple(instances),
    )
