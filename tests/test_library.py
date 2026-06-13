import tomllib

import pytest

from monata.library import Library
from monata.errors import CellNotFoundError
from monata.projection import PDKProjectionContext, projection_context_for


def _make_lib(tmp_path, name="mylib", model_paths=None):
    lib_dir = tmp_path / name
    lib_dir.mkdir()
    model_paths = model_paths or ["/path/to/nmos.lib"]
    paths_str = ", ".join(f'"{p}"' for p in model_paths)
    (lib_dir / "lib.toml").write_text(
        f'[library]\nname = "{name}"\ndescription = "test lib"\n\n'
        f'[technology]\nmodel_paths = [{paths_str}]\n'
    )
    return lib_dir


def test_library_load_from_path(tmp_path):
    lib_dir = _make_lib(tmp_path, "analog")
    lib = Library(lib_dir)
    assert lib.name == "analog"
    assert lib.tech_model_paths == ["/path/to/nmos.lib"]


def test_library_create_writes_metadata(tmp_path):
    lib = Library.create(
        tmp_path / "analog",
        name="analog",
        tech_model_paths=["models/nmos.lib"],
        description="frontend devices",
    )

    with open(lib.path / "lib.toml", "rb") as file:
        config = tomllib.load(file)
    assert lib.name == "analog"
    assert lib.tech_model_paths == ["models/nmos.lib"]
    assert config["library"] == {"name": "analog", "description": "frontend devices"}


def test_library_create_writes_techlib_attachments(tmp_path):
    lib = Library.create(
        tmp_path / "digital",
        name="digital",
        techlib_attachments=["PTM_BULK"],
        default_corner="ptm65",
    )

    with open(lib.path / "lib.toml", "rb") as file:
        config = tomllib.load(file)

    assert config["attachments"] == {"techlibs": ["PTM_BULK"], "default_corner": "ptm65"}
    assert lib.attached_techlibs == ["PTM_BULK"]
    assert lib.techlib_attachments[0].default_corner == "ptm65"


def test_library_path_property(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    assert lib.path == lib_dir


def test_library_tech_optional_fields(tmp_path):
    lib_dir = tmp_path / "finlib"
    lib_dir.mkdir()
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "finlib"\ndescription = ""\n\n'
        '[technology]\nmodel_paths = ["/m.lib"]\n'
        'node_type = "finfet"\nsimulator = "xyce"\n'
    )
    lib = Library(lib_dir)
    assert lib.node_type == "finfet"
    assert lib.simulator == "xyce"


def test_library_tech_optional_fields_absent(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    assert lib.node_type is None
    assert lib.simulator is None


def test_library_config_rejects_unknown_root_fields(tmp_path):
    lib_dir = _make_lib(tmp_path)
    text = (lib_dir / "lib.toml").read_text()
    (lib_dir / "lib.toml").write_text(f"unexpected = true\n{text}")

    with pytest.raises(ValueError, match="lib.toml has unknown fields: unexpected"):
        _ = Library(lib_dir).name


def test_library_config_rejects_unknown_library_fields(tmp_path):
    lib_dir = _make_lib(tmp_path)
    text = (lib_dir / "lib.toml").read_text()
    (lib_dir / "lib.toml").write_text(
        text.replace(
            '[library]\nname = "mylib"\n',
            '[library]\nname = "mylib"\nunexpected = true\n',
        )
    )

    with pytest.raises(ValueError, match="library table has unknown fields: unexpected"):
        _ = Library(lib_dir).name


def test_library_config_rejects_unknown_technology_fields(tmp_path):
    lib_dir = _make_lib(tmp_path)
    text = (lib_dir / "lib.toml").read_text()
    (lib_dir / "lib.toml").write_text(
        text.replace(
            '[technology]\nmodel_paths = ["/path/to/nmos.lib"]\n',
            '[technology]\nmodel_paths = ["/path/to/nmos.lib"]\nunexpected = true\n',
        )
    )

    with pytest.raises(ValueError, match="technology table has unknown fields: unexpected"):
        _ = Library(lib_dir).tech_model_paths


def test_library_config_allows_library_named_extension_table(tmp_path):
    lib_dir = _make_lib(tmp_path)
    with (lib_dir / "lib.toml").open("a") as file:
        file.write('\n[mylib.digital]\ndut_categories = ["adders"]\n')

    assert Library(lib_dir).name == "mylib"


def test_library_exposes_explicit_pdk_projection_context(tmp_path):
    lib_dir = _make_lib(tmp_path)
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "mylib"\ndescription = "test lib"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_BULK"]\ndefault_corner = "ptm65"\n'
    )

    context = Library(lib_dir).pdk_projection_context()

    assert isinstance(context, PDKProjectionContext)
    assert [attachment.name for attachment in context.techlib_attachments] == ["PTM_BULK"]
    assert context.techlib_attachments[0].default_corner == "ptm65"


def test_projection_context_for_uses_library_owner_boundary(tmp_path):
    lib_dir = _make_lib(tmp_path)
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "mylib"\ndescription = "test lib"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_BULK"]\ndefault_corner = "ptm65"\n'
    )

    context = projection_context_for(Library(lib_dir))

    assert isinstance(context, PDKProjectionContext)
    assert [attachment.name for attachment in context.techlib_attachments] == ["PTM_BULK"]
    assert PDKProjectionContext.from_owner(context) is context


def test_library_list_cells_empty(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    assert lib.list_cells() == []


def test_library_list_cells(tmp_path):
    lib_dir = _make_lib(tmp_path)
    cell_dir = lib_dir / "inverter"
    cell_dir.mkdir()
    (cell_dir / "cell.toml").write_text(
        '[cell]\nname = "inverter"\ndescription = ""\n\n[views]\n'
    )
    (lib_dir / "scratch").mkdir()
    (lib_dir / "notes.txt").write_text("not a cell")
    lib = Library(lib_dir)
    assert lib.list_cells() == ["inverter"]


def test_library_getitem(tmp_path):
    lib_dir = _make_lib(tmp_path)
    cell_dir = lib_dir / "inverter"
    cell_dir.mkdir()
    (cell_dir / "cell.toml").write_text(
        '[cell]\nname = "inverter"\ndescription = ""\n\n[views]\n'
    )
    lib = Library(lib_dir)
    cell = lib["inverter"]
    assert cell.name == "inverter"


def test_library_getitem_not_found(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    with pytest.raises(CellNotFoundError):
        lib["nonexistent"]


def test_library_contains(tmp_path):
    lib_dir = _make_lib(tmp_path)
    cell_dir = lib_dir / "inverter"
    cell_dir.mkdir()
    (cell_dir / "cell.toml").write_text(
        '[cell]\nname = "inverter"\ndescription = ""\n\n[views]\n'
    )
    lib = Library(lib_dir)
    assert "inverter" in lib
    assert "nand" not in lib


def test_library_iter(tmp_path):
    lib_dir = _make_lib(tmp_path)
    for name in ("inv", "buf"):
        d = lib_dir / name
        d.mkdir()
        (d / "cell.toml").write_text(f'[cell]\nname = "{name}"\n\n[views]\n')
    lib = Library(lib_dir)
    assert set(lib) == {"inv", "buf"}


def test_library_create_cell(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    cell = lib.create_cell("buffer", description="unity gain buffer")
    assert cell.name == "buffer"
    assert (lib_dir / "buffer" / "cell.toml").exists()


def test_library_create_category_and_category_cell(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)

    category = lib.create_category("logic", description="logic cells")
    cell = category.create_cell("inverter", description="logic inverter")

    assert category.name == "logic"
    assert category.qualified_name == "logic"
    assert (lib_dir / "logic" / "category.toml").exists()
    assert cell.name == "inverter"
    assert cell.category_path == "logic"
    assert cell.qualified_name == "logic/inverter"
    assert lib.list_categories() == ["logic"]
    assert category.list_cells() == ["inverter"]


def test_category_config_rejects_unknown_root_fields(tmp_path):
    lib_dir = _make_lib(tmp_path)
    category = Library(lib_dir).create_category("logic")
    text = (category.path / "category.toml").read_text()
    (category.path / "category.toml").write_text(f"unexpected = true\n{text}")

    with pytest.raises(ValueError, match="category.toml has unknown fields: unexpected"):
        _ = category.name


def test_category_config_rejects_unknown_category_fields(tmp_path):
    lib_dir = _make_lib(tmp_path)
    category = Library(lib_dir).create_category("logic")
    text = (category.path / "category.toml").read_text()
    (category.path / "category.toml").write_text(
        text.replace(
            '[category]\nname = "logic"\n',
            '[category]\nname = "logic"\nunexpected = true\n',
        )
    )

    with pytest.raises(ValueError, match="category table has unknown fields: unexpected"):
        _ = category.name


def test_category_create_category_invalidates_parent_category_cache(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    category = lib.create_category("logic")

    assert category.list_categories() == []

    child = category.create_category("gates")

    assert child.qualified_name == "logic/gates"
    assert category.list_categories() == ["gates"]


def test_library_recursive_cell_discovery_and_lookup(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    logic = lib.create_category("logic")
    adders = lib.create_category("adders")
    logic.create_cell("inverter")
    adders.create_cell("half_adder")

    assert lib.list_cells() == []
    assert lib.list_cells(recursive=True) == ["adders/half_adder", "logic/inverter"]
    assert [cell.qualified_name for cell in lib.iter_cells(recursive=True)] == [
        "adders/half_adder",
        "logic/inverter",
    ]
    assert lib["logic/inverter"].qualified_name == "logic/inverter"
    assert lib["inverter"].qualified_name == "logic/inverter"


def test_library_ambiguous_bare_cell_lookup_fails(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    lib.create_category("logic").create_cell("inverter")
    lib.create_category("examples").create_cell("inverter")

    with pytest.raises(CellNotFoundError, match="ambiguous"):
        lib["inverter"]


def test_library_create_cell_rejects_duplicates_without_overwriting(tmp_path):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)
    lib.create_cell("buffer", description="original")

    with pytest.raises(FileExistsError, match="Cell already exists: buffer"):
        lib.create_cell("buffer", description="replacement")

    with open(lib_dir / "buffer" / "cell.toml", "rb") as file:
        config = tomllib.load(file)
    assert config["cell"]["description"] == "original"


@pytest.mark.parametrize("name", ["../outside", "quote\"name", "evil\nname", "tab\tname", "space name"])
def test_library_create_cell_rejects_unsafe_names(tmp_path, name):
    lib_dir = _make_lib(tmp_path)
    lib = Library(lib_dir)

    with pytest.raises(ValueError, match="single safe path segment"):
        lib.create_cell(name)
