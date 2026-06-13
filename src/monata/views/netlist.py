from pathlib import Path

from monata.views.base import View
from monata.errors import ViewNotGeneratedError


class NetlistView(View):
    def __init__(self, cell, entry: str):
        super().__init__(view_type="netlist", cell=cell, entry=entry, generated=True)

    def load(self) -> Path:
        file_path = self.path() / self._entry
        if not file_path.exists():
            raise ViewNotGeneratedError("netlist", self._cell.name)
        return file_path
