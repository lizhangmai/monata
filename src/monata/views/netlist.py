from pathlib import Path

from monata.views.base import View
from monata.errors import ViewNotGeneratedError
from monata.views.path_safety import resolve_cell_relative_path


class NetlistView(View):
    def __init__(self, cell, entry: str):
        super().__init__(
            view_type="netlist",
            cell=cell,
            entry=entry,
            generated=True,
            format="spice",
        )

    def load(self) -> Path:
        file_path = resolve_cell_relative_path(
            self.path(),
            self._entry,
            label="netlist.entry",
        )
        if not file_path.exists():
            raise ViewNotGeneratedError("netlist", self._cell.name)
        return file_path
