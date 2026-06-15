from pathlib import Path

import pytest

from monata._paths import expand_path, find_file, walk_files


def test_expand_path_expands_environment_and_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    monkeypatch.setenv("MONATA_TEST_ROOT", str(root))

    expanded = expand_path("$MONATA_TEST_ROOT/models")

    assert expanded.is_absolute()
    assert expanded == (root / "models").absolute()
    assert str(expand_path("~")).startswith(str(Path.home()))


def test_walk_files_yields_files_in_deterministic_order(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "z.mod").write_text("", encoding="utf-8")
    (tmp_path / "a" / "m.mod").write_text("", encoding="utf-8")
    (tmp_path / "a" / "a.mod").write_text("", encoding="utf-8")

    assert [path.relative_to(tmp_path).as_posix() for path in walk_files(tmp_path)] == [
        "a/a.mod",
        "a/m.mod",
        "b/z.mod",
    ]

    single = tmp_path / "single.lib"
    single.write_text("", encoding="utf-8")

    assert list(walk_files(single)) == [single]


def test_find_file_searches_directories_in_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "nested").mkdir(parents=True)
    second.mkdir()
    (first / "nested" / "device.mod").write_text("first", encoding="utf-8")
    (second / "device.mod").write_text("second", encoding="utf-8")

    assert find_file("device.mod", [first, second]) == first / "nested" / "device.mod"

    with pytest.raises(FileNotFoundError, match="missing.mod"):
        find_file("missing.mod", [first, second])
    with pytest.raises(ValueError, match="file_name"):
        find_file("../device.mod", [first])
