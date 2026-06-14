import warnings

from monata.views.base import View


class TestbenchView(View):
    __test__ = False

    def __init__(
        self,
        cell,
        entry: str,
        function_name: str,
        *,
        view_type: str = "testbench",
        trusted: bool = True,
        legacy_trusted: bool = False,
    ):
        super().__init__(
            view_type=view_type,
            cell=cell,
            entry=entry,
            generated=False,
            format="python-testbench",
            trusted=trusted,
            legacy_trusted=legacy_trusted,
        )
        self._function_name = function_name

    def load_trusted(self):
        return self.load_python_attribute("testbench", self._function_name)

    def load(self):
        warnings.warn(
            "TestbenchView.load() executes trusted Python code; use load_trusted() "
            "for Python views or a monata-testbench-json data view for safe parsing.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_trusted()

    def run_trusted(self):
        func = self.load_trusted()
        return func(self._cell)

    def run(self):
        warnings.warn(
            "TestbenchView.run() executes trusted Python code; use run_trusted() "
            "for Python testbench views.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_trusted()
