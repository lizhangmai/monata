from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monata._config import (
    cell_config,
    read_toml,
    reject_unknown_fields,
    write_cell_config,
    write_library_config,
)
from monata._paths import validate_path_segment
from monata.category import CATEGORY_CONFIG_FILENAME, Category
from monata.errors import CellNotFoundError
from monata._types import ReferenceMode

if TYPE_CHECKING:
    from monata.corner import CornerLike, OperatingCorner
    from monata.projection import PDKProjectionContext

LIBRARY_CONFIG_FILENAME = "lib.toml"
CELL_CONFIG_FILENAME = "cell.toml"
_LIBRARY_CONFIG_FIELDS = frozenset({"library", "technology", "attachments"})
_LIBRARY_TABLE_FIELDS = frozenset({"name", "description"})
_TECHNOLOGY_TABLE_FIELDS = frozenset({"model_paths", "node_type", "simulator"})


def _validate_library_config(config: Mapping[str, Any]) -> None:
    try:
        library = config["library"]
        technology = config["technology"]
    except KeyError as exc:
        raise ValueError(f"lib.toml is missing [{exc.args[0]}]") from exc
    if not isinstance(library, Mapping):
        raise ValueError("[library] must be a table")
    if not isinstance(technology, Mapping):
        raise ValueError("[technology] must be a table")
    root_fields = set(_LIBRARY_CONFIG_FIELDS)
    library_name = library.get("name")
    if isinstance(library_name, str) and library_name:
        root_fields.add(library_name)
    reject_unknown_fields(config, frozenset(root_fields), "lib.toml")
    reject_unknown_fields(library, _LIBRARY_TABLE_FIELDS, "library table")
    reject_unknown_fields(technology, _TECHNOLOGY_TABLE_FIELDS, "technology table")


class Library:
    def __init__(self, path):
        self._path = Path(path)
        self._config = None
        self._cells_cache = None
        self._categories_cache = None
        self._recursive_cells_cache = None

    @classmethod
    def create(
        cls,
        path,
        *,
        name: str,
        tech_model_paths: list[str] | None = None,
        techlib_attachments: list[str] | None = None,
        default_corner: str | None = None,
        description: str = "",
    ):
        safe_name = validate_path_segment(name, "library name")
        lib_dir = Path(path)
        lib_dir.mkdir(parents=True, exist_ok=True)
        write_library_config(
            lib_dir / LIBRARY_CONFIG_FILENAME,
            name=safe_name,
            tech_model_paths=tech_model_paths or [],
            techlib_attachments=techlib_attachments or [],
            default_corner=default_corner,
            description=description,
        )
        return cls(lib_dir)

    def _config_path(self) -> Path:
        return self._path / LIBRARY_CONFIG_FILENAME

    def _cell_path(self, name: str) -> tuple[str, Path]:
        safe_name = validate_path_segment(name, "cell name")
        return safe_name, self._path / safe_name

    def _category_parts(self, path: str) -> tuple[str, ...]:
        parts = tuple(part for part in str(path).split("/") if part)
        if not parts:
            raise ValueError("category path must name at least one category")
        return tuple(validate_path_segment(part, "category path segment") for part in parts)

    def _load_config(self):
        if self._config is None:
            self._config = read_toml(self._config_path())
            _validate_library_config(self._config)
        return self._config

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._load_config()["library"]["name"]

    @property
    def tech_model_paths(self) -> list[str]:
        return self._load_config()["technology"]["model_paths"]

    @property
    def techlib_attachments(self):
        from monata.techlib.parse import parse_techlib_attachments

        return parse_techlib_attachments(self._load_config().get("attachments"))

    @property
    def attached_techlibs(self) -> list[str]:
        return [attachment.name for attachment in self.techlib_attachments]

    def pdk_projection_context(self) -> "PDKProjectionContext":
        """Return the explicit PDK projection boundary for this library."""

        from monata.projection import PDKProjectionContext

        return PDKProjectionContext(tuple(self.techlib_attachments))

    def validate_pdk_instance(
        self,
        instance: Any,
        registry: Any = None,
        corner: "CornerLike" = None,
        require_projectable: bool = False,
    ) -> Any:
        return self.pdk_projection_context().validate_pdk_instance(
            instance,
            registry=registry,
            corner=corner,
            require_projectable=require_projectable,
        )

    def project_pdk_instance(
        self,
        instance: Any,
        registry: Any = None,
        corner: "CornerLike" = None,
    ) -> Any:
        return self.pdk_projection_context().project_pdk_instance(
            instance,
            registry=registry,
            corner=corner,
        )

    def project_pdk_instances(
        self,
        netlist: Any,
        registry: Any = None,
        corner: "CornerLike" = None,
        reference_mode: ReferenceMode = "concrete",
        include_models: bool = True,
    ) -> Any:
        return self.pdk_projection_context().project_pdk_instances(
            netlist,
            registry=registry,
            corner=corner,
            reference_mode=reference_mode,
            include_models=include_models,
        )

    def resolve_pdk_corner(
        self,
        corner: "CornerLike" = None,
        registry: Any = None,
    ) -> "OperatingCorner | None":
        return self.pdk_projection_context().resolve_pdk_corner(corner=corner, registry=registry)

    @property
    def node_type(self):
        return self._load_config()["technology"].get("node_type")

    @property
    def simulator(self):
        return self._load_config()["technology"].get("simulator")

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

    def _scan_cells_recursive(self) -> dict[str, Path]:
        if self._recursive_cells_cache is None:
            cells = dict(self._scan_cells())
            for path in sorted(self._path.rglob(CELL_CONFIG_FILENAME)):
                cell_dir = path.parent
                if cell_dir.parent == self._path:
                    continue
                qualified_name = cell_dir.relative_to(self._path).as_posix()
                cells[qualified_name] = cell_dir
            self._recursive_cells_cache = cells
        return self._recursive_cells_cache

    def _clear_cell_cache(self) -> None:
        self._cells_cache = None
        self._categories_cache = None
        self._recursive_cells_cache = None

    def list_cells(self, *, recursive: bool = False) -> list[str]:
        cells = self._scan_cells_recursive() if recursive else self._scan_cells()
        return sorted(cells.keys())

    def iter_cells(self, *, recursive: bool = False):
        from monata.cell import Cell

        cells = self._scan_cells_recursive() if recursive else self._scan_cells()
        for qualified_name in sorted(cells):
            yield Cell(cells[qualified_name], self, category=self._category_for_cell_path(cells[qualified_name]))

    def list_categories(self) -> list[str]:
        return sorted(self._scan_categories().keys())

    def iter_categories(self, *, recursive: bool = False):
        categories = self._scan_categories()
        for name in sorted(categories):
            category = Category(categories[name], self)
            yield category
            if recursive:
                yield from self._iter_subcategories(category)

    def _iter_subcategories(self, category: Category):
        for name in category.list_categories():
            child = category.get_category(name)
            yield child
            yield from self._iter_subcategories(child)

    def get_category(self, path: str) -> Category:
        parts = self._category_parts(path)
        category_path = self._path.joinpath(*parts)
        if not (category_path / CATEGORY_CONFIG_FILENAME).exists():
            raise KeyError(f"category not found: {'/'.join(parts)}")
        return Category(category_path, self)

    def create_category(self, path: str, description: str = "") -> Category:
        parts = self._category_parts(path)
        category_path = self._path.joinpath(*parts)
        if category_path.exists():
            raise FileExistsError(f"Category already exists: {'/'.join(parts)}")
        if len(parts) > 1 and not (category_path.parent / CATEGORY_CONFIG_FILENAME).exists():
            raise KeyError(f"parent category not found: {'/'.join(parts[:-1])}")
        category = Category.create(category_path, self, name=category_path.name, description=description)
        self._categories_cache = None
        return category

    def get_cell(self, name: str):
        from monata.cell import Cell

        if "/" in str(name):
            cells = self._scan_cells_recursive()
            if name not in cells:
                raise CellNotFoundError(name, self.name)
            return Cell(cells[name], self, category=self._category_for_cell_path(cells[name]))

        cells = self._scan_cells()
        if name in cells:
            return Cell(cells[name], self)

        matches = [
            (qualified_name, path)
            for qualified_name, path in self._scan_cells_recursive().items()
            if qualified_name.split("/")[-1] == name
        ]
        if not matches:
            raise CellNotFoundError(name, self.name)
        if len(matches) > 1:
            choices = ", ".join(qualified_name for qualified_name, _path in sorted(matches))
            raise CellNotFoundError(f"{name} is ambiguous; use one of: {choices}", self.name)
        _qualified_name, path = matches[0]
        return Cell(path, self, category=self._category_for_cell_path(path))

    def _category_for_cell_path(self, cell_path: Path) -> Category | None:
        try:
            relative_parent = cell_path.parent.relative_to(self._path)
        except ValueError:
            return None
        if relative_parent == Path("."):
            return None
        category_path = self._path / relative_parent
        if not (category_path / CATEGORY_CONFIG_FILENAME).exists():
            return None
        return Category(category_path, self)

    def __getitem__(self, name: str):
        return self.get_cell(name)

    def __contains__(self, name: str) -> bool:
        try:
            self.get_cell(name)
        except CellNotFoundError:
            return False
        return True

    def __iter__(self):
        return iter(self._scan_cells().keys())

    def create_cell(self, name: str, description: str = ""):
        from monata.cell import Cell

        safe_name, cell_dir = self._cell_path(name)
        if cell_dir.exists():
            raise FileExistsError(f"Cell already exists: {safe_name}")
        cell_dir.mkdir()
        write_cell_config(cell_dir / CELL_CONFIG_FILENAME, cell_config(safe_name, description=description))
        self._clear_cell_cache()
        return Cell(cell_dir, self)
