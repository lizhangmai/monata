from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import importlib.util
import keyword
from os import PathLike
from pathlib import Path
import re
import sys
from typing import Any


class View:
    def __init__(self, view_type: str, cell, entry: str, generated: bool = False):
        self._view_type = view_type
        self._cell = cell
        self._entry = entry
        self._generated = generated

    @property
    def view_type(self) -> str:
        return self._view_type

    @property
    def cell(self):
        return self._cell

    @property
    def entry(self) -> str:
        return self._entry

    @property
    def generated(self) -> bool:
        return self._generated

    def path(self) -> Path:
        return self._cell.path

    def load_python_entry(self, module_name: str):
        file_path = self.path() / self._entry
        if not file_path.exists():
            raise FileNotFoundError(f"{self._view_type.capitalize()} file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {self._view_type} file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        previous_module = sys.modules.get(module_name) if module_name in sys.modules else None

        with _temporary_import_paths(self._python_import_paths(file_path)):
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                if previous_module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous_module
                raise

        return module

    def load_python_attribute(self, module_prefix: str, attribute_name: str):
        module_name = self.python_module_name(module_prefix)
        had_previous_module = module_name in sys.modules
        previous_module = sys.modules.get(module_name)
        module = self.load_python_entry(module_name)
        try:
            return getattr(module, attribute_name)
        except AttributeError:
            if had_previous_module and previous_module is not None:
                sys.modules[module_name] = previous_module
            else:
                sys.modules.pop(module_name, None)
            raise

    def _python_import_paths(self, file_path: Path) -> list[Path]:
        paths = [file_path.parent]

        library = getattr(self._cell, "library", None)
        library_path = _path_from(getattr(library, "path", None))
        if library_path is not None:
            paths.append(library_path.parent)

        return paths

    def python_module_name(self, prefix: str) -> str:
        library = getattr(self._cell, "library", None)
        library_name = getattr(library, "name", None)
        cell_name = getattr(self._cell, "name", None)
        category_path = getattr(self._cell, "category_path", None)
        category_parts = tuple(str(category_path).split("/")) if category_path else ()
        entry_stem = Path(self._entry).stem

        parts = tuple(str(part) for part in (library_name, *category_parts, cell_name, entry_stem))
        if all(_is_identifier(part) for part in parts):
            return ".".join(parts)

        digest = hashlib.sha1("/".join(parts).encode("utf-8")).hexdigest()[:12]
        safe_parts = "_".join(_identifier_segment(part) for part in parts)
        return f"{prefix}_{safe_parts}_{digest}"

    def load(self):
        raise NotImplementedError(
            "load() not implemented for base View. Use a specific view subclass."
        )

    def run(self) -> Any:
        raise TypeError(
            f"run() is only valid on testbench views, not '{self._view_type}'"
        )


def _path_from(value) -> Path | None:
    if isinstance(value, (str, PathLike)):
        return Path(value)
    return None


def _is_identifier(value) -> bool:
    return isinstance(value, str) and value.isidentifier() and not keyword.iskeyword(value)


def _identifier_segment(value: str) -> str:
    candidate = re.sub(r"\W+", "_", value).strip("_")
    if not candidate:
        candidate = "part"
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_"
    return candidate


@contextmanager
def _temporary_import_paths(paths: list[Path]) -> Iterator[None]:
    original = list(sys.path)
    seen = set()
    normalized = []

    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)

    for path in reversed(normalized):
        sys.path.insert(0, path)

    try:
        yield
    finally:
        sys.path[:] = original
