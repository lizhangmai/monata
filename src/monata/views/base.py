from pathlib import Path
from typing import Any


class View:
    def __init__(
        self,
        view_type: str,
        cell,
        entry: str,
        generated: bool = False,
        *,
        format: str | None = None,
        schema_version: int | None = None,
    ):
        self._view_type = view_type
        self._cell = cell
        self._entry = entry
        self._generated = generated
        self._format = format
        self._schema_version = schema_version

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

    @property
    def format(self) -> str | None:
        return self._format

    @property
    def schema_version(self) -> int | None:
        return self._schema_version

    def path(self) -> Path:
        return self._cell.path

    def load(self):
        raise NotImplementedError(
            "load() not implemented for base View. Use a specific view subclass."
        )

    def read(self) -> Any:
        raise TypeError(
            f"read() is only valid on declarative data views, not '{self._view_type}'"
        )

    def run(self) -> Any:
        raise TypeError(
            f"run() is only valid on testbench views, not '{self._view_type}'"
        )
