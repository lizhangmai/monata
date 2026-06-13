import pytest

from monata.registry import LibraryRegistry
from monata.errors import LibraryNotFoundError


def _make_lib_dir(base, name):
    lib_dir = base / name
    lib_dir.mkdir(parents=True)
    (lib_dir / "lib.toml").write_text(
        f'[library]\nname = "{name}"\ndescription = ""\n\n'
        f'[technology]\nmodel_paths = ["/m.lib"]\n'
    )
    return lib_dir


def test_registry_empty():
    reg = LibraryRegistry()
    assert reg.list_libraries() == []


def test_registry_search_paths(tmp_path):
    search = tmp_path / "libs"
    search.mkdir()
    _make_lib_dir(search, "analog")
    _make_lib_dir(search, "digital")

    reg = LibraryRegistry(search_paths=[str(search)])
    libs = reg.list_libraries()
    assert "analog" in libs
    assert "digital" in libs


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            'unexpected = true\n\n[library]\nname = "analog"\n\n[technology]\nmodel_paths = []\n',
            "lib.toml has unknown fields: unexpected",
        ),
        (
            '[library]\nname = "analog"\n',
            r"lib.toml is missing \[technology\]",
        ),
    ],
)
def test_registry_search_paths_validate_discovered_libraries(tmp_path, body, message):
    search = tmp_path / "libs"
    lib_dir = search / "analog"
    lib_dir.mkdir(parents=True)
    (lib_dir / "lib.toml").write_text(body)

    with pytest.raises(ValueError, match=message):
        LibraryRegistry(search_paths=[str(search)])


def test_registry_getitem(tmp_path):
    search = tmp_path / "libs"
    search.mkdir()
    _make_lib_dir(search, "mylib")

    reg = LibraryRegistry(search_paths=[str(search)])
    lib = reg["mylib"]
    assert lib.name == "mylib"


def test_registry_getitem_not_found(tmp_path):
    reg = LibraryRegistry(search_paths=[str(tmp_path)])
    with pytest.raises(LibraryNotFoundError):
        reg["nonexistent"]


def test_registry_contains(tmp_path):
    search = tmp_path / "libs"
    search.mkdir()
    _make_lib_dir(search, "mylib")

    reg = LibraryRegistry(search_paths=[str(search)])
    assert "mylib" in reg
    assert "other" not in reg


def test_registry_iter(tmp_path):
    search = tmp_path / "libs"
    search.mkdir()
    _make_lib_dir(search, "a")
    _make_lib_dir(search, "b")

    reg = LibraryRegistry(search_paths=[str(search)])
    assert set(reg) == {"a", "b"}


def test_registry_add_library(tmp_path):
    lib_dir = _make_lib_dir(tmp_path, "standalone")
    reg = LibraryRegistry()
    lib = reg.add_library(str(lib_dir))
    assert lib.name == "standalone"
    assert "standalone" in reg


def test_registry_create_library(tmp_path):
    reg = LibraryRegistry()
    lib = reg.create_library(
        path=str(tmp_path / "newlib"),
        name="newlib",
        tech_model_paths=["/path/to/model.lib"],
    )
    assert lib.name == "newlib"
    assert (tmp_path / "newlib" / "lib.toml").exists()
    assert "newlib" in reg


def test_registry_multiple_search_paths(tmp_path):
    s1 = tmp_path / "path1"
    s2 = tmp_path / "path2"
    s1.mkdir()
    s2.mkdir()
    _make_lib_dir(s1, "lib_a")
    _make_lib_dir(s2, "lib_b")

    reg = LibraryRegistry(search_paths=[str(s1), str(s2)])
    assert "lib_a" in reg
    assert "lib_b" in reg


def test_registry_ignores_non_library_dirs(tmp_path):
    search = tmp_path / "libs"
    search.mkdir()
    _make_lib_dir(search, "real_lib")
    (search / "not_a_lib").mkdir()

    reg = LibraryRegistry(search_paths=[str(search)])
    assert reg.list_libraries() == ["real_lib"]
