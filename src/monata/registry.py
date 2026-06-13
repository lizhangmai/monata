from pathlib import Path

from monata.errors import LibraryNotFoundError
from monata.library import Library


class LibraryRegistry:
    def __init__(self, search_paths=None):
        self._search_paths = [Path(p) for p in (search_paths or [])]
        self._libraries: dict[str, Path] = {}
        self._scan()

    def _scan(self):
        for search_path in self._search_paths:
            if not search_path.is_dir():
                continue
            for entry in search_path.iterdir():
                if entry.is_dir() and (entry / "lib.toml").exists():
                    lib = Library(entry)
                    self._libraries[lib.name] = lib.path

    def add_library(self, path) -> Library:
        lib = Library(path)
        self._libraries[lib.name] = Path(path)
        return lib

    def create_library(
        self,
        path: str,
        name: str,
        tech_model_paths: list[str] | None = None,
        techlib_attachments: list[str] | None = None,
        default_corner: str | None = None,
        description: str = "",
    ) -> Library:
        lib = Library.create(
            path,
            name=name,
            tech_model_paths=tech_model_paths,
            techlib_attachments=techlib_attachments,
            default_corner=default_corner,
            description=description,
        )
        self._libraries[lib.name] = lib.path
        return lib

    def list_libraries(self) -> list:
        return sorted(self._libraries.keys())

    def __getitem__(self, name: str) -> Library:
        if name not in self._libraries:
            raise LibraryNotFoundError(name)
        return Library(self._libraries[name])

    def __contains__(self, name: str) -> bool:
        return name in self._libraries

    def __iter__(self):
        return iter(self._libraries.keys())
