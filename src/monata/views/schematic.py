import warnings

from monata.views.base import View


class SchematicView(View):
    def __init__(
        self,
        cell,
        entry: str,
        cls_name: str,
        *,
        view_type: str = "schematic",
        trusted: bool = True,
    ):
        super().__init__(
            view_type=view_type,
            cell=cell,
            entry=entry,
            generated=False,
            format="python-schematic",
            trusted=trusted,
        )
        self._cls_name = cls_name

    def load_trusted(self):
        return self.load_python_attribute("schematic", self._cls_name)

    def load(self):
        warnings.warn(
            "SchematicView.load() executes trusted Python code; use load_trusted() "
            "for Python views or a monata-schematic-json data view for safe parsing.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_trusted()
