from monata.views.base import View


class SchematicView(View):
    def __init__(self, cell, entry: str, cls_name: str):
        super().__init__(view_type="schematic", cell=cell, entry=entry, generated=False)
        self._cls_name = cls_name

    def load(self):
        return self.load_python_attribute("schematic", self._cls_name)
