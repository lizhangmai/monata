"""Digital verification entry discovery for Monata libraries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import tomllib
from typing import cast

from monata.cell import Cell
from monata.library import Library
from monata.digital.spec import DigitalVerificationSpec


@dataclass(frozen=True)
class DigitalTestbenchEntry:
    cell: Cell
    view: object
    spec: DigitalVerificationSpec

    @property
    def testbench_cell(self) -> str:
        return cast(str, getattr(self.cell, "qualified_name", getattr(self.cell, "name", "<unknown>")))


def discover_digital_testbench_entries(library: Library) -> tuple[DigitalTestbenchEntry, ...]:
    entries = []
    for cell in library.iter_cells(recursive=True):
        if "verification" not in cell:
            continue
        view = cell["verification"]
        spec = view.spec()
        if not isinstance(spec, DigitalVerificationSpec):
            qualified = getattr(cell, "qualified_name", cell.name)
            raise TypeError(f"{qualified} verification view must load a DigitalVerificationSpec")
        if "simulation" not in cell:
            qualified = getattr(cell, "qualified_name", cell.name)
            raise RuntimeError(f"{qualified} must define a 'simulation' view")
        entries.append(DigitalTestbenchEntry(cell=cell, view=view, spec=spec))
    return tuple(sorted(entries, key=lambda entry: entry.spec.dut))


def validate_digital_testbench_coverage(
    library: Library,
    entries: Iterable[DigitalTestbenchEntry],
) -> None:
    expected = digital_schematic_cells(library)
    actual = {entry.spec.dut for entry in entries}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(
            "missing verification testbench views for digital cells: "
            + ", ".join(missing)
        )


def digital_schematic_cells(library: Library) -> set[str]:
    category_prefixes = digital_dut_category_prefixes(library)
    names: set[str] = set()
    for cell in library.iter_cells(recursive=True):
        category_path = getattr(cell, "category_path", None)
        if category_path not in category_prefixes and not any(
            str(category_path).startswith(f"{prefix}/") for prefix in category_prefixes
        ):
            continue
        if "schematic" in cell:
            names.add(cell.name)
    return names


def digital_dut_category_prefixes(library: Library) -> tuple[str, ...]:
    with (library.path / "lib.toml").open("rb") as file:
        config = tomllib.load(file)
    categories = config.get(library.name, {}).get("digital", {}).get("dut_categories", ())
    prefixes = tuple(str(category).strip("/") for category in categories if str(category).strip("/"))
    if not prefixes:
        raise RuntimeError(
            f"{library.path / 'lib.toml'} must define {library.name}.digital.dut_categories"
        )
    return prefixes


def select_digital_testbench_entries(
    entries: Iterable[DigitalTestbenchEntry],
    requested: Iterable[str],
    *,
    all_token: str = "all",
) -> tuple[DigitalTestbenchEntry, ...]:
    selected = tuple(entries)
    requested_names = {name.strip() for name in requested if name.strip()}
    if requested_names and requested_names != {all_token}:
        selected = tuple(entry for entry in selected if entry_matches_name(entry, requested_names))
        matched = {
            name
            for entry in selected
            for name in entry_names(entry)
        }
        missing = requested_names - matched
        if missing:
            raise ValueError(
                "requested cells did not match any digital verification views: "
                + ", ".join(sorted(missing))
            )
    if not selected:
        raise ValueError("no digital verification views selected")
    return selected


def entry_matches_name(entry: DigitalTestbenchEntry, requested: set[str]) -> bool:
    return any(name in requested for name in entry_names(entry))


def entry_names(entry: DigitalTestbenchEntry) -> tuple[str, ...]:
    return (
        entry.spec.dut,
        entry.testbench_cell,
        entry.testbench_cell.split("/")[-1],
    )


def parse_name_set(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}
