import json

import pytest
import tomllib
from unittest.mock import MagicMock

from monata.cell import Cell
from monata.errors import ViewAlreadyModifiedError
from monata.library import Library
from monata.projection import PDKProjectionContext


def _make_cell_with_schematic(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir()

    (cell_dir / "schematic.py").write_text(
        "from monata.netlist import SubCircuit\n"
        "\n"
        "class Inverter(SubCircuit):\n"
        "    NAME = 'inverter'\n"
        "    NODES = ('vin', 'out', 'vdd', 'gnd')\n"
        "\n"
        "    def build(self):\n"
        "        pass\n"
    )

    (cell_dir / "cell.toml").write_text(
        '[cell]\nname = "inverter"\ndescription = "source schematic"\n\n'
        '[views]\n'
        'schematic = { entry = "schematic.py", class = "Inverter" }\n'
    )

    lib = MagicMock()
    lib.name = "testlib"
    return Cell(cell_dir, lib)


def _make_cell_with_pdk_schematic(tmp_path, library):
    cell_dir = tmp_path / "pdk_inv"
    cell_dir.mkdir()

    (cell_dir / "schematic.py").write_text(
        "from monata.netlist import SubCircuit\n"
        "\n"
        "class PdkInv(SubCircuit):\n"
        "    NAME = 'pdk_inv'\n"
        "    NODES = ('vin', 'out', 'vdd', 'gnd')\n"
        "\n"
        "    def build(self):\n"
        "        self.pdk_instance(\n"
        "            'mn',\n"
        "            lib='PTM_TEST',\n"
        "            cell='nfet',\n"
        "            view='ngspice',\n"
        "            pins={'d': 'out', 'g': 'vin', 's': 'gnd', 'b': 'gnd'},\n"
        "        )\n"
    )

    (cell_dir / "cell.toml").write_text(
        '[cell]\nname = "pdk_inv"\ndescription = "source schematic"\n\n'
        '[views]\n'
        'schematic = { entry = "schematic.py", class = "PdkInv" }\n'
    )

    return Cell(cell_dir, library)


class RecordingProjectionContext(PDKProjectionContext):
    def __init__(self):
        super().__init__()
        self.calls = []

    def project_pdk_instances(
        self,
        netlist,
        registry=None,
        corner=None,
        reference_mode="concrete",
        include_models=True,
    ):
        self.calls.append(
            {
                "registry": registry,
                "corner": corner,
                "reference_mode": reference_mode,
                "include_models": include_models,
            }
        )
        scope = netlist.ensure_built()
        for instance in tuple(scope.pdk_instances):
            scope.mos(
                instance.name,
                d=instance.pins["d"],
                g=instance.pins["g"],
                s=instance.pins["s"],
                b=instance.pins["b"],
                model="nfet_model",
            )
        scope.pdk_instances.clear()
        return netlist


class RecordingLibrary:
    name = "testlib"

    def __init__(self, context):
        self.context = context

    def pdk_projection_context(self):
        return self.context


def test_generate_symbol(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)
    result_path = cell.generate_symbol()

    assert result_path.exists()
    assert result_path.name == "symbol.monata.json"

    data = json.loads(result_path.read_text())

    assert data["name"] == "inverter"
    assert data["schema_version"] == 1
    assert data["view_type"] == "symbol"
    pins = data["pins"]
    assert len(pins) == 4

    pin_map = {p["name"]: p["direction"] for p in pins}
    assert pin_map["vin"] == "input"
    assert pin_map["out"] == "output"
    assert pin_map["vdd"] == "inout"
    assert pin_map["gnd"] == "inout"


def test_generate_symbol_updates_cell_toml(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)
    cell.generate_symbol()

    with open(cell.path / "cell.toml", "rb") as f:
        config = tomllib.load(f)

    assert "symbol" in config["views"]
    assert config["views"]["symbol"]["entry"] == "symbol.monata.json"
    assert config["views"]["symbol"]["format"] == "monata-symbol-json"
    assert config["views"]["symbol"]["schema_version"] == 1
    assert config["views"]["symbol"]["generated"] is True


def test_generate_symbol_refuses_when_not_generated(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)
    cell.generate_symbol()
    cell_toml = cell.path / "cell.toml"
    content = cell_toml.read_text().replace("generated = true", "generated = false")
    cell_toml.write_text(content)
    cell._config = None

    with pytest.raises(ViewAlreadyModifiedError):
        cell.generate_symbol()


def test_generate_symbol_force_overwrites(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)
    cell.generate_symbol()
    cell_toml = cell.path / "cell.toml"
    content = cell_toml.read_text().replace("generated = true", "generated = false")
    cell_toml.write_text(content)
    cell._config = None

    result_path = cell.generate_symbol(force=True)
    assert result_path.exists()


def test_generate_netlist(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)
    result_path = cell.generate_netlist()

    assert result_path.exists()
    assert result_path.name == "netlist.cir"

    content = result_path.read_text()
    assert ".subckt inverter" in content
    assert ".ends inverter" in content


def test_generate_views_for_category_owned_cell(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    cell = lib.create_category("logic").create_cell("inverter")
    (cell.path / "schematic.py").write_text(
        "from monata.netlist import SubCircuit\n"
        "\n"
        "class Inverter(SubCircuit):\n"
        "    NAME = 'inverter'\n"
        "    NODES = ('vin', 'out', 'vdd', 'gnd')\n"
        "\n"
        "    def build(self):\n"
        "        pass\n"
    )
    cell.create_view("schematic", cls_name="Inverter")

    symbol_path = cell.generate_symbol()
    netlist_path = cell.generate_netlist()

    assert symbol_path == cell.path / "symbol.monata.json"
    assert netlist_path == cell.path / "netlist.cir"
    assert lib["logic/inverter"].qualified_name == "logic/inverter"


def test_generate_netlist_does_not_project_pdk_instances_by_default(tmp_path):
    context = RecordingProjectionContext()
    cell = _make_cell_with_pdk_schematic(tmp_path, RecordingLibrary(context))

    result_path = cell.generate_netlist()

    assert context.calls == []
    assert "Mmn" not in result_path.read_text()


def test_generate_netlist_projects_only_when_requested(tmp_path):
    context = RecordingProjectionContext()
    registry = object()
    corner = object()
    cell = _make_cell_with_pdk_schematic(tmp_path, RecordingLibrary(context))

    result_path = cell.generate_netlist(projection="logical", registry=registry, corner=corner)

    assert context.calls == [
            {
                "registry": registry,
                "corner": corner,
                "reference_mode": "logical",
                "include_models": True,
            }
        ]
    assert "Mmn out vin gnd gnd nfet_model" in result_path.read_text()


def test_generate_netlist_updates_cell_toml(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)
    cell.generate_netlist()

    with open(cell.path / "cell.toml", "rb") as f:
        config = tomllib.load(f)

    assert "netlist" in config["views"]
    assert config["views"]["netlist"]["entry"] == "netlist.cir"
    assert config["views"]["netlist"]["format"] == "spice"
    assert config["views"]["netlist"]["generated"] is True
    assert config["cell"] == {"name": "inverter", "description": "source schematic"}
    assert config["views"]["schematic"] == {"entry": "schematic.py", "class": "Inverter"}


def test_generate_netlist_refuses_when_not_generated(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)
    cell.generate_netlist()
    cell_toml = cell.path / "cell.toml"
    content = cell_toml.read_text().replace("generated = true", "generated = false")
    cell_toml.write_text(content)
    cell._config = None

    with pytest.raises(ViewAlreadyModifiedError):
        cell.generate_netlist()


def test_generate_netlist_force_overwrites(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)
    cell.generate_netlist()
    cell_toml = cell.path / "cell.toml"
    content = cell_toml.read_text().replace("generated = true", "generated = false")
    cell_toml.write_text(content)
    cell._config = None

    result_path = cell.generate_netlist(force=True)
    assert result_path.exists()


def test_generate_view_dispatches_registered_generator(tmp_path):
    cell = _make_cell_with_schematic(tmp_path)

    result_path = cell.generate_view("symbol")

    assert result_path == cell.path / "symbol.monata.json"
    assert result_path.exists()
