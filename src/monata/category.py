from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from monata._config import (
    category_config,
    cell_config,
    read_toml,
    reject_unknown_fields,
    write_category_config,
    write_cell_config,
)
from monata._paths import validate_path_segment
from monata.errors import CellNotFoundError

CATEGORY_CONFIG_FILENAME = "category.toml"
CELL_CONFIG_FILENAME = "cell.toml"
_CATEGORY_CONFIG_FIELDS = frozenset({"category"})
_CATEGORY_TABLE_FIELDS = frozenset({"name", "description"})


def _validate_category_config(config: Mapping[str, Any]) -> None:
    reject_unknown_fields(config, _CATEGORY_CONFIG_FIELDS, "category.toml")
    try:
        category = config["category"]
    except KeyError as exc:
        raise ValueError("category.toml is missing [category]") from exc
    if not isinstance(category, Mapping):
        raise ValueError("[category] must be a table")
    reject_unknown_fields(category, _CATEGORY_TABLE_FIELDS, "category table")


class Category:
    """Filesystem-backed library category containing cells and subcategories."""

    def __init__(self, path, library):
        self._path = Path(path)
        self._library = library
        self._config = None
        self._cells_cache = None
        self._categories_cache = None

    @classmethod
    def create(cls, path, library, *, name: str, description: str = "") -> "Category":
        safe_name = validate_path_segment(name, "category name")
        category_dir = Path(path)
        category_dir.mkdir(parents=True, exist_ok=False)
        write_category_config(
            category_dir / CATEGORY_CONFIG_FILENAME,
            category_config(safe_name, description=description),
        )
        return cls(category_dir, library)

    def _load_config(self):
        if self._config is None:
            self._config = read_toml(self._path / CATEGORY_CONFIG_FILENAME)
            _validate_category_config(self._config)
        return self._config

    @property
    def path(self) -> Path:
        return self._path

    @property
    def library(self):
        return self._library

    @property
    def name(self) -> str:
        return self._load_config()["category"]["name"]

    @property
    def qualified_name(self) -> str:
        return self.path_relative_to_library().as_posix()

    def path_relative_to_library(self) -> Path:
        return self._path.relative_to(self._library.path)

    def _cell_path(self, name: str) -> tuple[str, Path]:
        safe_name = validate_path_segment(name, "cell name")
        return safe_name, self._path / safe_name

    def _scan_cells(self) -> dict[str, Path]:
        if self._cells_cache is None:
            self._cells_cache = {}
            for entry in self._path.iterdir():
                if entry.is_dir() and (entry / CELL_CONFIG_FILENAME).exists():
                    self._cells_cache[entry.name] = entry
        return self._cells_cache

    def _scan_categories(self) -> dict[str, Path]:
        if self._categories_cache is None:
            self._categories_cache = {}
            for entry in self._path.iterdir():
                if entry.is_dir() and (entry / CATEGORY_CONFIG_FILENAME).exists():
                    self._categories_cache[entry.name] = entry
        return self._categories_cache

    def list_cells(self) -> list[str]:
        return sorted(self._scan_cells())

    def list_categories(self) -> list[str]:
        return sorted(self._scan_categories())

    def get_category(self, path: str) -> "Category":
        return self._library.get_category(f"{self.qualified_name}/{path}")

    def create_category(self, name: str, description: str = "") -> "Category":
        category = self._library.create_category(f"{self.qualified_name}/{name}", description=description)
        self._categories_cache = None
        return category

    def create_cell(self, name: str, description: str = ""):
        from monata.cell import Cell

        safe_name, cell_dir = self._cell_path(name)
        if cell_dir.exists():
            raise FileExistsError(f"Cell already exists: {self.qualified_name}/{safe_name}")
        cell_dir.mkdir()
        write_cell_config(cell_dir / CELL_CONFIG_FILENAME, cell_config(safe_name, description=description))
        self._cells_cache = None
        self._library._clear_cell_cache()
        return Cell(cell_dir, self._library, category=self)

    def __getitem__(self, name: str):
        from monata.cell import Cell

        cells = self._scan_cells()
        if name not in cells:
            raise CellNotFoundError(name, self._library.name)
        return Cell(cells[name], self._library, category=self)

    def __contains__(self, name: str) -> bool:
        return name in self._scan_cells()

    def __iter__(self):
        return iter(self._scan_cells())
