import json
import tomllib

import pytest

from monata.cell import Cell
from monata.library import Library
from monata.netlist import SubCircuit, render_ngspice
from monata.schematic import SchematicData
from monata.sim.core import SimTask, TranSpec
from monata.views.declarative import (
    SchematicJsonView,
    SymbolJsonView,
    TestbenchJsonView,
    parse_metric_number,
    schematic_view_to_circuit,
)

LEGACY_SCHEMATIC_FORMAT = "python-" + "schematic"
REMOVED_SCHEMATIC_VIEW = "schematic" + "_py"


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _schematic_payload():
    return {
        "schema_version": 2,
        "view_type": "schematic",
        "cell": {"name": "inverter"},
        "interface": {
            "pins": [
                {"name": "vin", "direction": "input", "net": "vin", "order": 0},
                {"name": "vout", "direction": "output", "net": "vout", "order": 1},
                {"name": "vdd", "direction": "power", "net": "vdd", "order": 2},
                {"name": "vss", "direction": "ground", "net": "vss", "order": 3},
            ]
        },
        "instances": [
            {
                "name": "mn",
                "ref": {"kind": "nmos", "device": "nmos"},
                "connections": {"d": "vout", "g": "vin", "s": "vss", "b": "vss"},
                "parameters": {"w": "1u", "l": "45n"},
            }
        ],
        "nets": [
            {"name": "vin", "kind": "signal"},
            {"name": "vout", "kind": "signal"},
            {"name": "vdd", "kind": "power"},
            {"name": "vss", "kind": "ground"},
        ],
        "provenance": [],
        "properties": {},
        "annotations": [],
    }


def _make_cell(tmp_path, views_toml):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir()
    (cell_dir / "cell.toml").write_text(
        '[cell]\nname = "inverter"\ndescription = "test"\n\n'
        '[views]\n'
        f"{views_toml}"
    )
    lib = Library.create(tmp_path / "lib", name="lib")
    cell = Cell(cell_dir, lib)
    return cell


def test_schematic_json_view_reads_without_executing_neighbor_python(tmp_path):
    cell = _make_cell(
        tmp_path,
        'schematic = { entry = "schematic.monata.json", format = "monata-schematic-json" }\n',
    )
    _write_json(cell.path / "schematic.monata.json", _schematic_payload())
    (cell.path / "schematic.py").write_text("raise RuntimeError('python schematic executed')\n")

    view = cell["schematic"]
    payload = view.load()
    circuit = view.to_circuit()

    assert isinstance(view, SchematicJsonView)
    assert isinstance(payload, SchematicData)
    assert payload.cell == "inverter"
    assert view.pin_names() == ("vin", "vout", "vdd", "vss")
    assert isinstance(circuit, SubCircuit)
    assert "Mmn vout vin vss vss nmos" in render_ngspice(circuit)


def test_legacy_python_schematic_metadata_rejected_without_import(tmp_path):
    cell = _make_cell(
        tmp_path,
        f'schematic = {{ entry = "schematic.py", format = "{LEGACY_SCHEMATIC_FORMAT}", trusted = false, class = "Inv" }}\n',
    )
    (cell.path / "schematic.py").write_text("raise RuntimeError('should not import')\n")

    with pytest.raises(ValueError, match="legacy Python schematic format is no longer supported"):
        cell["schematic"]


def test_create_view_metadata_uses_data_schematic_and_rejects_removed_python_view(tmp_path):
    cell = _make_cell(tmp_path, "")

    schematic = cell.create_view("schematic")
    with pytest.raises(ValueError, match="removed Python schematic view type"):
        cell.create_view(REMOVED_SCHEMATIC_VIEW, cls_name="Inv")
    testbench = cell.create_view("testbench", entry="custom.monata.json")

    with open(cell.path / "cell.toml", "rb") as file:
        config = tomllib.load(file)

    assert schematic.entry == "schematic.monata.json"
    assert config["views"]["schematic"] == {
        "entry": "schematic.monata.json",
        "format": "monata-schematic-json",
        "schema_version": 2,
    }
    assert testbench.entry == "custom.monata.json"
    assert REMOVED_SCHEMATIC_VIEW not in config["views"]
    assert config["views"]["testbench"] == {
        "entry": "custom.monata.json",
        "format": "monata-testbench-json",
        "schema_version": 1,
    }


def test_create_view_data_format_rejects_python_metadata(tmp_path):
    cell = _make_cell(tmp_path, "")

    with pytest.raises(ValueError, match="Python schematic metadata is not supported"):
        cell.create_view("schematic", format="monata-schematic-json", cls_name="Inv")
    with pytest.raises(ValueError, match="use testbench_py"):
        cell.create_view("testbench", format="monata-testbench-json", function_name="main")

    with open(cell.path / "cell.toml", "rb") as file:
        config = tomllib.load(file)
    assert config["views"] == {}


def test_explicit_unknown_format_fails_closed(tmp_path):
    cell = _make_cell(
        tmp_path,
        'schematic = { entry = "schematic.monata.json", format = "monata-schematic-jsno" }\n',
    )

    with pytest.raises(ValueError, match="unknown view format"):
        cell.create_view("schematic", format="monata-schematic-jsno")
    with pytest.raises(ValueError, match="unknown view format"):
        cell["schematic"]


def test_schematic_conversion_refuses_python_without_explicit_allow(tmp_path):
    marker = tmp_path / "executed.txt"

    class LegacyPythonView:
        format = LEGACY_SCHEMATIC_FORMAT
        trusted = True

        def load_trusted(self):
            marker.write_text("executed")
            return SubCircuit("legacy", nodes=("a", "y"))

    with pytest.raises(TypeError, match="legacy Python schematic format is no longer supported"):
        schematic_view_to_circuit(LegacyPythonView(), allow_trusted_python=False, reason="unit test")
    assert not marker.exists()


def test_schematic_conversion_rejects_unregistered_to_circuit_object():
    class AdHocView:
        format = "adhoc-schematic"
        trusted = True

        def to_circuit(self):
            return SubCircuit("adhoc", nodes=("a", "b"))

    with pytest.raises(TypeError, match="unsupported schematic view format"):
        schematic_view_to_circuit(AdHocView(), allow_trusted_python=False, reason="unit test")


def test_symbol_json_view_loads_normalized_payload(tmp_path):
    cell = _make_cell(
        tmp_path,
        'symbol = { entry = "symbol.monata.json", format = "monata-symbol-json", generated = true, schema_version = 1 }\n',
    )
    _write_json(
        cell.path / "symbol.monata.json",
        {
            "schema_version": 1,
            "view_type": "symbol",
            "pins": [{"name": "vin", "side": "left"}, {"name": "vout", "side": "right"}],
        },
    )

    view = cell["symbol"]
    assert isinstance(view, SymbolJsonView)
    assert view.load() == {
        "name": "inverter",
        "pins": [{"name": "vin", "side": "left"}, {"name": "vout", "side": "right"}],
    }


def test_testbench_json_view_builds_sim_task_from_data_schematic(tmp_path):
    lib = Library.create(tmp_path / "lib", name="lib")
    cell = lib.create_cell("inverter")
    _write_json(cell.path / "schematic.monata.json", _schematic_payload())
    _write_json(
        cell.path / "testbench.monata.json",
        {
            "schema_version": 1,
            "view_type": "testbench",
            "dut": "inverter",
            "analysis": {"kind": "tran", "step": "1p", "stop": "2n"},
            "sources": [
                {
                    "kind": "vpulse",
                    "name": "vin",
                    "node": "vin",
                    "ref": "0",
                    "values": ["0", "1", "0", "10p", "10p", "500p", "1n"],
                }
            ],
            "outputs": ["vout"],
            "measurements": ["truth_table", "max_propagation_delay"],
        },
    )
    cell.create_view("schematic")
    cell.create_view("testbench")

    view = cell["testbench"]
    task = view.to_sim_task()

    assert isinstance(view, TestbenchJsonView)
    assert isinstance(task, SimTask)
    assert isinstance(task.analysis_spec, TranSpec)
    assert task.analysis_spec.step == pytest.approx(1e-12)
    assert task.analysis_spec.stop == pytest.approx(2e-9)
    assert task.output_names == ("vout",)
    assert task.metadata["measurements"] == ("truth_table", "max_propagation_delay")
    assert "PULSE(0 1 0 10p 10p 500p 1n)" in render_ngspice(task.circuit)


def test_testbench_json_view_does_not_execute_python_schematic_by_default(tmp_path):
    lib = Library.create(tmp_path / "lib", name="lib")
    cell = lib.create_cell("inverter")
    marker = tmp_path / "executed.txt"
    _write_json(cell.path / "schematic.monata.json", _schematic_payload())
    (cell.path / "schematic.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
    _write_json(
        cell.path / "testbench.monata.json",
        {
            "schema_version": 1,
            "view_type": "testbench",
            "dut": "inverter",
            "analysis": {"kind": "tran", "step": "1p", "stop": "2n"},
            "sources": [],
        },
    )
    cell.create_view("schematic")
    cell.create_view("testbench")

    cell["testbench"].to_sim_task()

    assert not marker.exists()


def test_testbench_json_view_refuses_legacy_python_schematic_by_default(tmp_path):
    lib = Library.create(tmp_path / "lib", name="lib")
    cell = lib.create_cell("legacy")
    marker = tmp_path / "executed.txt"
    (cell.path / "schematic.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "from monata.netlist import SubCircuit\n"
        "class Legacy(SubCircuit):\n"
        "    NAME = 'legacy'\n"
        "    NODES = ('a', 'z')\n"
    )
    _write_json(
        cell.path / "testbench.monata.json",
        {
            "schema_version": 1,
            "view_type": "testbench",
            "dut": "legacy",
            "analysis": {"kind": "tran", "step": "1p", "stop": "2n"},
            "sources": [],
        },
    )
    cell.create_view("testbench")
    (cell.path / "cell.toml").write_text(
        '[cell]\nname = "legacy"\n\n'
        '[views]\n'
        f'schematic = {{ entry = "schematic.py", format = "{LEGACY_SCHEMATIC_FORMAT}", trusted = true, class = "Legacy" }}\n'
        'testbench = { entry = "testbench.monata.json", format = "monata-testbench-json", schema_version = 1 }\n'
    )
    cell._config = None

    with pytest.raises(ValueError, match="legacy Python schematic format is no longer supported"):
        cell["testbench"].to_sim_task()
    assert not marker.exists()


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"kind": "vdc", "name": "vin", "value": "1"}, "requires node or p"),
        ({"kind": "vdc", "name": "vin", "node": "vin"}, "value is required"),
        ({"kind": "vpulse", "name": "vin", "node": "vin", "values": ["0", "1"]}, "7 values"),
        ({"kind": "behavioral", "name": "b1", "node": "vin", "value": "1"}, "unsupported"),
    ],
)
def test_testbench_json_view_rejects_incomplete_sources(tmp_path, source, message):
    cell = _make_cell(
        tmp_path,
        'testbench = { entry = "testbench.monata.json", format = "monata-testbench-json" }\n',
    )
    _write_json(
        cell.path / "testbench.monata.json",
        {
            "schema_version": 1,
            "view_type": "testbench",
            "dut": "inverter",
            "analysis": {"kind": "tran", "step": "1p", "stop": "2n"},
            "sources": [source],
        },
    )

    with pytest.raises(ValueError, match=message):
        cell["testbench"].read()


def test_metric_number_parser_accepts_spice_suffixes():
    assert parse_metric_number("45n", field="unit") == pytest.approx(45e-9)
    assert parse_metric_number("1meg", field="unit") == pytest.approx(1e6)
