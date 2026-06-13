from monata.views.base import View


class TestbenchView(View):
    __test__ = False

    def __init__(self, cell, entry: str, function_name: str):
        super().__init__(view_type="testbench", cell=cell, entry=entry, generated=False)
        self._function_name = function_name

    def load(self):
        return self.load_python_attribute("testbench", self._function_name)

    def run(self):
        func = self.load()
        return func(self._cell)
