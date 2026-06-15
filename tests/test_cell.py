import tomllib
from unittest.mock import MagicMock

import pytest

from monata.cell import Cell
from monata.errors import ViewAlreadyModifiedError, ViewNotFoundError
from monata.library import Library
from monata.schematic import SchematicBuilder
from monata.views import View
from monata.views.registry import register_view_type, unregister_view_type


def _make_cell(tmp_path, cell_name="inverter", views_toml=""):
    cell_dir = tmp_path / cell_name
    cell_dir.mkdir()
    (cell_dir / "cell.toml").write_text(
        f'[cell]\nname = "{cell_name}"\ndescription = "test"\n\n'
        f'[views]\n{views_toml}'
    )
    return cell_dir


def test_cell_name(tmp_path):
    cell_dir = _make_cell(tmp_path, "inverter")
    lib = MagicMock()
    lib.name = "mylib"
    cell = Cell(cell_dir, lib)
    assert cell.name == "inverter"


def test_cell_config_rejects_unknown_root_fields(tmp_path):
    cell_dir = _make_cell(tmp_path, "inverter")
    text = (cell_dir / "cell.toml").read_text()
    (cell_dir / "cell.toml").write_text(f"unexpected = true\n{text}")
    cell = Cell(cell_dir, MagicMock())

    with pytest.raises(ValueError, match="cell.toml has unknown fields: unexpected"):
        _ = cell.name


def test_cell_config_rejects_unknown_cell_fields(tmp_path):
    cell_dir = _make_cell(tmp_path, "inverter")
    text = (cell_dir / "cell.toml").read_text()
    (cell_dir / "cell.toml").write_text(
        text.replace(
            '[cell]\nname = "inverter"\n',
            '[cell]\nname = "inverter"\nunexpected = true\n',
        )
    )
    cell = Cell(cell_dir, MagicMock())

    with pytest.raises(ValueError, match="cell table has unknown fields: unexpected"):
        _ = cell.name


def test_cell_config_rejects_non_table_view_entries(tmp_path):
    cell_dir = _make_cell(tmp_path, "inverter", views_toml='schematic = "bad"\n')
    cell = Cell(cell_dir, MagicMock())

    with pytest.raises(ValueError, match="view schematic config must be a table"):
        cell.list_views()


def test_cell_library_backref(tmp_path):
    cell_dir = _make_cell(tmp_path)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    assert cell.library is lib


def test_cell_path(tmp_path):
    cell_dir = _make_cell(tmp_path)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    assert cell.path == cell_dir


def test_cell_list_views_empty(tmp_path):
    cell_dir = _make_cell(tmp_path)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    assert cell.list_views() == []


def test_cell_list_views(tmp_path):
    views_toml = 'schematic = { entry = "schematic.monata.json", format = "monata-schematic-json", schema_version = 2 }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    assert "schematic" in cell.list_views()


def test_cell_getitem_schematic(tmp_path):
    views_toml = 'schematic = { entry = "schematic.monata.json", format = "monata-schematic-json", schema_version = 2 }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    view = cell["schematic"]
    assert view.view_type == "schematic"
    assert view.entry == "schematic.monata.json"


def test_cell_getitem_testbench(tmp_path):
    views_toml = 'testbench = { entry = "testbench.monata.json", format = "monata-testbench-json", schema_version = 1 }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    view = cell["testbench"]
    assert view.view_type == "testbench"
    assert view.entry == "testbench.monata.json"


def test_cell_getitem_netlist(tmp_path):
    views_toml = 'netlist = { entry = "netlist.scs", generated = true }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    view = cell["netlist"]
    assert view.view_type == "netlist"
    assert view.generated is True


def test_cell_getitem_symbol(tmp_path):
    views_toml = 'symbol = { entry = "symbol.monata.json", format = "monata-symbol-json", generated = true }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    view = cell["symbol"]
    assert view.view_type == "symbol"


@pytest.mark.parametrize(
    ("views_toml", "view_type", "message"),
    [
        (
            'schematic = { entry = "schematic.py", class = "Inv" }\n',
            "schematic",
            "Python class metadata is no longer supported",
        ),
        (
            'testbench = { entry = "testbench.py", function = "main" }\n',
            "testbench",
            "executable Python metadata",
        ),
        (
            'testbench = { entry = "testbench.py", format = "python-testbench" }\n',
            "testbench",
            "Python testbench cellviews are no longer supported",
        ),
        (
            'testbench = { entry = "testbench.monata.json", format = "monata-testbench-json", trusted = false }\n',
            "testbench",
            "executable Python metadata",
        ),
        (
            'symbol = { entry = "symbol.toml", generated = true }\n',
            "symbol",
            "symbol.monata.json",
        ),
    ],
)
def test_cell_getitem_rejects_removed_view_metadata(tmp_path, views_toml, view_type, message):
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    cell = Cell(cell_dir, MagicMock())

    with pytest.raises(ValueError, match=message):
        cell[view_type]


def test_cell_getitem_not_found(tmp_path):
    cell_dir = _make_cell(tmp_path)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    with pytest.raises(ViewNotFoundError):
        cell["layout"]


def test_cell_getitem_uses_registered_custom_view(tmp_path):
    views_toml = 'layout = { entry = "layout.toml" }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)

    register_view_type("layout", lambda owner, cfg: View("layout", owner, str(cfg["entry"])))
    try:
        view = cell["layout"]
    finally:
        unregister_view_type("layout")

    assert view.view_type == "layout"
    assert view.entry == "layout.toml"


def test_cell_contains(tmp_path):
    views_toml = 'schematic = { entry = "schematic.monata.json", format = "monata-schematic-json", schema_version = 2 }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    assert "schematic" in cell
    assert "layout" not in cell


def test_cell_generated_view_writable_contract_honors_force(tmp_path):
    views_toml = 'netlist = { entry = "netlist.cir", generated = false }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)

    with pytest.raises(ViewAlreadyModifiedError):
        cell.write_generated_view("netlist", entry="netlist.cir", content="new", force=False)

    path = cell.write_generated_view("netlist", entry="netlist.cir", content="new", force=True)
    assert path.read_text() == "new"
    cell.write_generated_view("symbol", entry="symbol.monata.json", content="{}\n", force=False)


def test_cell_create_view(tmp_path):
    cell_dir = _make_cell(tmp_path)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)
    view = cell.create_view("schematic")
    assert view.view_type == "schematic"


def test_cell_create_view_preserves_existing_metadata(tmp_path):
    views_toml = 'schematic = { entry = "schematic.monata.json", format = "monata-schematic-json", schema_version = 2 }\n'
    cell_dir = _make_cell(tmp_path, views_toml=views_toml)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)

    view = cell.create_view("netlist", entry="netlist.cir", generated=True)

    with open(cell_dir / "cell.toml", "rb") as file:
        config = tomllib.load(file)
    assert view.view_type == "netlist"
    assert config["cell"] == {"name": "inverter", "description": "test"}
    assert config["views"]["schematic"] == {
        "entry": "schematic.monata.json",
        "format": "monata-schematic-json",
        "schema_version": 2,
    }
    assert config["views"]["netlist"] == {
        "entry": "netlist.cir",
        "format": "spice",
        "generated": True,
    }


@pytest.mark.parametrize("view_type", ["../layout", "bad view", "evil\nview", "tab\tview", ""])
def test_cell_create_view_rejects_unsafe_view_types_without_writing(tmp_path, view_type):
    cell_dir = _make_cell(tmp_path)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)

    with pytest.raises(ValueError, match="view type must be a single safe path segment"):
        cell.create_view(view_type, entry="layout.gds")

    with open(cell_dir / "cell.toml", "rb") as file:
        config = tomllib.load(file)
    assert config["views"] == {}


def test_cell_create_view_rejects_removed_implicit_python_metadata_without_writing(tmp_path):
    cell_dir = _make_cell(tmp_path)
    cell = Cell(cell_dir, MagicMock())

    with pytest.raises(ValueError, match="cannot include Python class metadata"):
        cell.create_view("schematic", entry="schematic.py", cls_name="Inv")

    with pytest.raises(ValueError, match="executable Python metadata"):
        cell.create_view("testbench", entry="testbench.py", function_name="main")

    with pytest.raises(ValueError, match="symbol.monata.json"):
        cell.create_view("symbol", entry="symbol.toml")

    with open(cell_dir / "cell.toml", "rb") as file:
        config = tomllib.load(file)
    assert config["views"] == {}


def test_cell_create_view_uses_registered_schema_for_custom_view(tmp_path):
    cell_dir = _make_cell(tmp_path)
    lib = MagicMock()
    cell = Cell(cell_dir, lib)

    register_view_type(
        "layout",
        lambda owner, cfg: View("layout", owner, str(cfg["entry"]), generated=bool(cfg["generated"])),
        default_entry="layout.gds",
        generated=True,
    )
    try:
        view = cell.create_view("layout")
    finally:
        unregister_view_type("layout")

    assert view.view_type == "layout"
    assert view.entry == "layout.gds"
    assert view.generated is True


def test_generate_netlist_projects_attached_pdk_instances(tmp_path, monkeypatch):
    techlib_dir = tmp_path / "PTM_BULK"
    model_dir = techlib_dir / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ptm_65nm_bulk.mod").write_text(".LIB ptm65\n.ENDL ptm65\n")
    (techlib_dir / "techlib.toml").write_text(
        """
[techlib]
name = "PTM_BULK"
default_corner = "ptm65"

[[model_decks]]
name = "ptm_bulk_65nm"
path = "models/ptm_65nm_bulk.mod"

[[corners]]
name = "ptm65"
model_deck = "ptm_bulk_65nm"
section = "ptm65"

[[devices]]
name = "nmos"
kind = "mosfet"
pins = ["d", "g", "s", "b"]

[devices.params.w]
default = "1.2u"

[devices.params.l]
default = "65n"

[[devices.views]]
name = "ngspice"
primitive = "mos"
pin_order = ["d", "g", "s", "b"]
params = ["w", "l"]

[devices.views.corner_models]
ptm65 = "ptm65nm_nmos"

[[devices]]
name = "pmos"
kind = "mosfet"
pins = ["d", "g", "s", "b"]

[devices.params.w]
default = "2.4u"

[devices.params.l]
default = "65n"

[[devices.views]]
name = "ngspice"
primitive = "mos"
pin_order = ["d", "g", "s", "b"]
params = ["w", "l"]

[devices.views.corner_models]
ptm65 = "ptm65nm_pmos"
"""
    )

    class EntryPoint:
        def load(self):
            return lambda: [techlib_dir]

    class EntryPoints(list):
        def select(self, group):
            assert group == "monata.techlibs"
            return self

    monkeypatch.setattr(
        "monata.techlib.registry.metadata.entry_points",
        lambda: EntryPoints([EntryPoint()]),
    )

    lib_dir = tmp_path / "mylib"
    cell_dir = lib_dir / "inv"
    cell_dir.mkdir(parents=True)
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "mylib"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_BULK"]\ndefault_corner = "ptm65"\n'
    )
    (cell_dir / "cell.toml").write_text(
        '[cell]\nname = "inv"\n\n'
        '[views]\nschematic = { entry = "schematic.monata.json", format = "monata-schematic-json", schema_version = 2 }\n'
    )
    (
        SchematicBuilder("inv")
        .pin("in", direction="input")
        .pin("out", direction="output")
        .pin("vdd", direction="power")
        .pin("gnd", direction="ground")
        .pdk_instance(
            "n",
            lib="PTM_BULK",
            cell="nmos",
            view="ngspice",
            pins={"d": "out", "g": "in", "s": "gnd", "b": "gnd"},
            parameters={"w": "1.2u", "l": "65n"},
        )
        .pdk_instance(
            "p",
            lib="PTM_BULK",
            cell="pmos",
            view="ngspice",
            pins={"d": "out", "g": "in", "s": "vdd", "b": "vdd"},
            parameters={"w": "2.4u", "l": "65n"},
        )
        .write(cell_dir / "schematic.monata.json")
    )

    netlist_path = Library(lib_dir)["inv"].generate_netlist(force=True, projection="logical")
    netlist = netlist_path.read_text()

    assert ".lib " not in netlist
    assert str(model_dir) not in netlist
    assert (
        ".monata_model_ref techlib=PTM_BULK corner=ptm65 "
        "deck=ptm_bulk_65nm section=ptm65 simulator=ngspice"
    ) in netlist
    assert "Mn out in gnd gnd ptm65nm_nmos w=1.2u l=65n" in netlist
    assert "Mp out in vdd vdd ptm65nm_pmos w=2.4u l=65n" in netlist
    assert " nmos " not in netlist
    assert " pmos " not in netlist
