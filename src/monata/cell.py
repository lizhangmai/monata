from collections.abc import Mapping
from pathlib import Path
from typing import Any

from monata._config import read_toml, reject_unknown_fields, write_cell_config
from monata._paths import validate_path_segment
from monata._types import NetlistProjectionMode
from monata.errors import ViewAlreadyModifiedError, ViewNotFoundError
from monata.views.registry import (
    ViewConfig,
    create_registered_view,
    create_registered_view_config,
    generate_registered_view,
)

_CELL_CONFIG_FIELDS = frozenset({"cell", "views"})
_CELL_TABLE_FIELDS = frozenset({"name", "description"})


def _validate_cell_config(config: Mapping[str, Any]) -> None:
    reject_unknown_fields(config, _CELL_CONFIG_FIELDS, "cell.toml")
    try:
        cell = config["cell"]
    except KeyError as exc:
        raise ValueError("cell.toml is missing [cell]") from exc
    if not isinstance(cell, Mapping):
        raise ValueError("[cell] must be a table")
    reject_unknown_fields(cell, _CELL_TABLE_FIELDS, "cell table")
    views = config.get("views", {})
    if not isinstance(views, Mapping):
        raise ValueError("[views] must be a table")
    for view_type, view_config in views.items():
        safe_view_type = validate_path_segment(view_type, "view type")
        if not isinstance(view_config, Mapping):
            raise ValueError(f"view {safe_view_type} config must be a table")


class Cell:
    def __init__(self, path, library, *, category=None):
        self._path = Path(path)
        self._library = library
        self._category = category
        self._config = None

    def _load_config(self):
        if self._config is None:
            self._config = read_toml(self._path / "cell.toml")
            _validate_cell_config(self._config)
        return self._config

    @property
    def name(self) -> str:
        return self._load_config()["cell"]["name"]

    @property
    def library(self):
        return self._library

    @property
    def category(self):
        if self._category is not None:
            return self._category
        return self._infer_category()

    @property
    def category_path(self) -> str | None:
        category = self.category
        return category.qualified_name if category is not None else None

    @property
    def qualified_name(self) -> str:
        category_path = self.category_path
        return f"{category_path}/{self.name}" if category_path else self.name

    @property
    def path(self) -> Path:
        return self._path

    def _infer_category(self):
        library_path = getattr(self._library, "path", None)
        if library_path is None:
            return None
        try:
            relative_parent = self._path.parent.relative_to(Path(library_path))
        except ValueError:
            return None
        if relative_parent == Path("."):
            return None
        category_path = Path(library_path) / relative_parent
        if not (category_path / "category.toml").exists():
            return None
        from monata.category import Category

        self._category = Category(category_path, self._library)
        return self._category

    def _views_config(self) -> dict[str, ViewConfig]:
        return self._load_config().get("views", {})

    def list_views(self) -> list:
        return sorted(self._views_config().keys())

    def __getitem__(self, view_type: str):
        views = self._views_config()
        if view_type not in views:
            raise ViewNotFoundError(view_type, self.name)
        cfg = views[view_type]
        return create_registered_view(self, view_type, cfg)

    def __contains__(self, view_type: str) -> bool:
        return view_type in self._views_config()

    def _is_generated_view(self, view_type: str) -> bool:
        config = self._views_config().get(view_type)
        if config is None:
            return True
        return bool(config.get("generated", True))

    def _ensure_generated_view_writable(self, view_type: str, *, force: bool) -> None:
        if not self._is_generated_view(view_type) and not force:
            raise ViewAlreadyModifiedError(view_type, self.name)

    def _set_view_config(self, view_type: str, view_config: ViewConfig) -> None:
        config = self._load_config()
        config.setdefault("views", {})[view_type] = view_config
        write_cell_config(self._path / "cell.toml", config)
        self._config = None

    def _register_generated_view(self, view_type: str, *, entry: str) -> None:
        self._set_view_config(
            view_type,
            create_registered_view_config(view_type, entry=entry, generated=True),
        )

    def write_generated_view(
        self,
        view_type: str,
        *,
        entry: str,
        content: str,
        force: bool = False,
    ) -> Path:
        """Write a generated view artifact and commit its view metadata."""

        self._ensure_generated_view_writable(view_type, force=force)
        view_path = self._path / entry
        view_path.write_text(content)
        self._register_generated_view(view_type, entry=entry)
        return view_path

    def create_view(self, view_type: str, **kwargs):
        view_cfg = create_registered_view_config(view_type, **kwargs)
        self._set_view_config(view_type, view_cfg)

        return self[view_type]

    def generate_symbol(self, force: bool = False) -> Path:
        return self.generate_view("symbol", force=force)

    def generate_netlist(
        self,
        force: bool = False,
        format: str = "cir",
        *,
        projection: NetlistProjectionMode = "none",
        registry: Any = None,
        corner: Any = None,
    ) -> Path:
        return self.generate_view(
            "netlist",
            force=force,
            format=format,
            projection=projection,
            registry=registry,
            corner=corner,
        )

    def generate_view(self, view_type: str, **kwargs) -> Path:
        return generate_registered_view(self, view_type, **kwargs)
