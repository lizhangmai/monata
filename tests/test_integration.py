import pytest

from monata import LibraryRegistry

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

    (cell.path / "schematic.py").write_text(
        "from monata.netlist import SubCircuit\n"
        "\n"
        "class Buffer(SubCircuit):\n"
        "    NAME = 'buffer'\n"
        "    NODES = ('inp', 'out', 'vdd', 'gnd')\n"
        "\n"
        "    def build(self):\n"
        "        pass\n"
    )
    cell.create_view(
        "schematic",
        entry="schematic.py",
        format="python-schematic",
        trusted=True,
        cls_name="Buffer",
    )

    sch = cell["schematic"]
    cls = sch.load()
    assert cls.NAME == "buffer"
    assert cls.NODES == ('inp', 'out', 'vdd', 'gnd')

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
    (cell.path / "sch.py").write_text(
        "from monata.netlist import SubCircuit\n"
        "class Inv(SubCircuit):\n"
        "    NAME = 'inv'\n"
        "    NODES = ('a', 'y', 'vdd', 'gnd')\n"
        "    def build(self):\n"
        "        pass\n"
    )
    cell.create_view(
        "schematic",
        entry="sch.py",
        format="python-schematic",
        trusted=True,
        cls_name="Inv",
    )

    cls = reg["mylib"]["inv"]["schematic"].load()
    assert cls.NAME == "inv"
