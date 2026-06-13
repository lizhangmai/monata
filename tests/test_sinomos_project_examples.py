from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
from typing import Callable, cast

import pytest

from monata.library import Library
from monata.netlist import SubCircuit, render_ngspice
from monata.sim.core import TranSpec
from monata.sim.digital_plan import digital_task_metadata
from monata.sim.digital_table import DigitalTruthTable
from monata.workspace.project import Project

pytestmark = pytest.mark.integration

SINOMOS_ROOT = Path(
    os.environ.get(
        "MONATA_SINOMOS_PROJECT",
        Path(__file__).resolve().parents[5] / "sinomos",
    )
)

SINOMOS_SMALL_CELLS = (
    "PT_NMOS",
    "PT_PMOS",
    "TG",
    "inv_lvt_mac",
    "Bit2FullAdder",
)
SINOMOS_LARGE_MULTIPLIER_CELLS = (
    "sinomos_187T_usertree_mul4",
    "sinomos_area221_mul4",
    "sinomos_step3_mul4",
    "sinomos_step4_mul4",
    "sinomos_step5_mul4",
    "sinomos_step6_mul4",
    "sinomos_step7_mul4",
)
PRIMITIVE_DEPENDENCIES = ("PT_NMOS", "PT_PMOS", "TG")
BIT2_DEPENDENCIES = (*PRIMITIVE_DEPENDENCIES, "inv_lvt_mac")

ExpectedFn = Callable[[tuple[int, ...]], tuple[int, ...]]
SubCircuitType = type[SubCircuit]


@dataclass(frozen=True)
class DigitalExample:
    cell_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    expected: ExpectedFn
    dependencies: tuple[str, ...] = ()
    rails: tuple[str, str] = ("vdd", "0")
    complement_inputs: tuple[str, ...] = ()


def _expected_bit2_full_adder(bits: tuple[int, ...]) -> tuple[int, int, int]:
    a1, a2, b1, b2, c0 = bits
    total = (a1 + (a2 << 1)) + (b1 + (b2 << 1)) + c0
    return (total >> 1) & 1, total & 1, (total >> 2) & 1


DIGITAL_EXAMPLES = (
    DigitalExample(
        "Bit2FullAdder",
        ("A1", "A2", "B1", "B2", "C0"),
        ("S2", "S1", "C2"),
        _expected_bit2_full_adder,
        dependencies=BIT2_DEPENDENCIES,
        rails=("VDD", "0"),
        complement_inputs=("A1_bar", "A2_bar", "B1_bar", "B2_bar", "C0_bar"),
    ),
)


def _sinomos_project() -> Project:
    if not (SINOMOS_ROOT / "project.toml").is_file():
        pytest.skip(f"SINOMOS project fixture is not available: {SINOMOS_ROOT}")
    return Project(SINOMOS_ROOT)


@pytest.fixture(scope="module")
def sinomos_library() -> Library:
    return _sinomos_project().get_library("sinomos")


def _load_schematic_class(library: Library, cell_name: str) -> SubCircuitType:
    schematic_cls = library[cell_name]["schematic"].load()
    assert isinstance(schematic_cls, type)
    assert issubclass(schematic_cls, SubCircuit)
    return cast(SubCircuitType, schematic_cls)


def test_sinomos_project_registers_non_multiplier_examples(sinomos_library: Library):
    project = _sinomos_project()
    observed = {cell.name for cell in sinomos_library.iter_cells(recursive=True)}

    assert project.name == "sinomos"
    assert "sinomos" in project.list_libraries()
    assert sinomos_library.name == "sinomos"
    assert set(sinomos_library.list_categories()) == {
        "adders",
        "multipliers",
        "primitives",
        "testbench",
    }
    assert set(SINOMOS_SMALL_CELLS).issubset(observed)
    assert set(SINOMOS_LARGE_MULTIPLIER_CELLS).issubset(observed)
    assert not (set(SINOMOS_SMALL_CELLS) & set(SINOMOS_LARGE_MULTIPLIER_CELLS))


@pytest.mark.parametrize("cell_name", SINOMOS_SMALL_CELLS)
def test_sinomos_non_multiplier_examples_render_checked_in_netlists(
    sinomos_library: Library,
    cell_name: str,
):
    cell = sinomos_library[cell_name]

    assert "schematic" in cell.list_views()
    assert "symbol" in cell.list_views()
    assert (cell.path / "symbol.toml").is_file()
    assert "sinomos_agent" not in (cell.path / "schematic.py").read_text()

    schematic_cls = _load_schematic_class(sinomos_library, cell_name)
    schematic = schematic_cls().ensure_built()
    expected_instance_count = getattr(schematic_cls, "EXPECTED_INSTANCE_COUNT", None)
    if expected_instance_count is not None:
        source_instances = [element for element in schematic.elements if element.kind == "X"]
        assert len(source_instances) == expected_instance_count

    projected = sinomos_library.project_pdk_instances(schematic, reference_mode="logical")
    rendered = render_ngspice(projected)

    assert rendered == (cell.path / "netlist.cir").read_text()


@pytest.mark.parametrize("example", DIGITAL_EXAMPLES, ids=lambda example: example.cell_name)
def test_sinomos_digital_examples_build_truth_table_tasks(
    sinomos_library: Library,
    example: DigitalExample,
):
    assert "mul4" not in example.cell_name
    for bits in product((0, 1), repeat=len(example.inputs)):
        assert len(example.expected(bits)) == len(example.outputs)

    table = DigitalTruthTable(
        _load_schematic_class(sinomos_library, example.cell_name),
        inputs=example.inputs,
        outputs=example.outputs,
        expected=example.expected,
        dependencies=tuple(
            _load_schematic_class(sinomos_library, dependency)
            for dependency in example.dependencies
        ),
        rails=example.rails,
        complement_inputs=example.complement_inputs,
        library=sinomos_library,
        metadata={"library": "sinomos", "cell": example.cell_name},
    )

    tasks = table.transient_tasks()
    vector_count = 2 ** len(example.inputs)
    arc_count = vector_count * len(example.inputs)
    assert len(tasks) == (arc_count + 127) // 128
    assert all(isinstance(task.analysis_spec, TranSpec) for task in tasks)
    assert all(task.output_names == (*example.inputs, *example.outputs) for task in tasks)
    task_payloads = [digital_task_metadata(task.metadata) for task in tasks]
    assert all(
        payload["digital_truth_table"]["task_kind"] == "digital-single-bit-arc-sequence"
        for payload in task_payloads
    )
    assert all(payload["measurements"] == ["truth_table"] for payload in task_payloads)
    assert all(
        payload["stimulus"]["kind"] == "digital_single_bit_arc_sequence"
        for payload in task_payloads
    )
    assert sum(payload["stimulus"]["arcs"] for payload in task_payloads) == arc_count
