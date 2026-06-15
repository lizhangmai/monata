import pytest

from monata import LibraryRegistry
from monata.schematic import SchematicBuilder

pytestmark = pytest.mark.integration


def test_full_workflow(tmp_path):
    """End-to-end: create library, add cell, generate views."""
    reg = LibraryRegistry()

    lib = reg.create_library(
        path=str(tmp_path / "testlib"),
        name="testlib",
        tech_model_paths=["/fake/nmos.lib", "/fake/pmos.lib"],
    )
    assert "testlib" in reg

    cell = lib.create_cell("buffer", description="unity gain buffer")
    assert "buffer" in lib

    (
        SchematicBuilder("buffer")
        .pin("inp", direction="input")
        .pin("out", direction="output")
        .pin("vdd", direction="power")
        .pin("gnd", direction="ground")
        .write(cell.path / "schematic.monata.json")
    )
    cell.create_view("schematic")

    sch = cell["schematic"]
    schematic = sch.load()
    assert schematic.cell == "buffer"
    assert schematic.pin_names == ("inp", "out", "vdd", "gnd")

    sym_path = cell.generate_symbol()
    assert sym_path.exists()
    sym_view = cell["symbol"]
    sym_data = sym_view.load()
    assert sym_data["name"] == "buffer"
    pin_names = [p["name"] for p in sym_data["pins"]]
    assert pin_names == ["inp", "out", "vdd", "gnd"]

    net_path = cell.generate_netlist()
    assert net_path.exists()
    content = net_path.read_text()
    assert ".subckt buffer" in content
    assert ".ends buffer" in content

    assert "schematic" in cell
    assert "symbol" in cell
    assert "netlist" in cell


def test_dict_like_access(tmp_path):
    """Verify the reg['lib']['cell']['view'] access pattern."""
    reg = LibraryRegistry()
    lib = reg.create_library(
        path=str(tmp_path / "mylib"),
        name="mylib",
        tech_model_paths=[],
    )
    cell = lib.create_cell("inv")
    (
        SchematicBuilder("inv")
        .pin("a", direction="input")
        .pin("y", direction="output")
        .pin("vdd", direction="power")
        .pin("gnd", direction="ground")
        .write(cell.path / "schematic.monata.json")
    )
    cell.create_view("schematic")

    schematic = reg["mylib"]["inv"]["schematic"].load()
    assert schematic.cell == "inv"
