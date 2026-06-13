import json
from pathlib import Path
import sys
from concurrent.futures import Future
from types import MappingProxyType, ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from monata.errors import ViewNotGeneratedError
from monata.library import Library
from monata.netlist import SubCircuit
from monata.sim.core import SimResult, SimTask, TranSpec
from monata.sim.digital_plan import digital_task_metadata
from monata.views.base import View
from monata.views.digital_truth_table import DigitalTruthTableView
from monata.views.netlist import NetlistView
from monata.views.schematic import SchematicView
from monata.views.simulation import SimulationView
from monata.views.symbol import SymbolView, infer_pin_direction
from monata.views.testbench import TestbenchView
from monata.views.registry import ViewRegistry, get_view_factory

def test_view_path(tmp_path):
    cell = MagicMock()
    cell.path = tmp_path / "inverter"
    view = View(view_type="schematic", cell=cell, entry="schematic.py", generated=False)
    assert view.path() == tmp_path / "inverter"


def test_view_properties():
    cell = MagicMock()
    cell.path = Path("/fake/lib/inverter")
    view = View(view_type="netlist", cell=cell, entry="netlist.scs", generated=True)
    assert view.view_type == "netlist"
    assert view.entry == "netlist.scs"
    assert view.generated is True
    assert view.cell is cell


def test_view_python_module_name_includes_category_path(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    cell = lib.create_category("logic").create_cell("inverter")
    view = View(view_type="schematic", cell=cell, entry="schematic.py", generated=False)

    assert view.python_module_name("schematic") == "mylib.logic.inverter.schematic"


def test_view_python_module_name_uses_full_path_for_non_identifier_categories(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    first = lib.create_category("logic-v1").create_cell("inverter")
    second = lib.create_category("logic.v2").create_cell("inverter")

    first_name = View(
        view_type="schematic",
        cell=first,
        entry="schematic.py",
        generated=False,
    ).python_module_name("schematic")
    second_name = View(
        view_type="schematic",
        cell=second,
        entry="schematic.py",
        generated=False,
    ).python_module_name("schematic")

    assert first_name != second_name
    assert first_name.startswith("schematic_mylib_logic_v1_inverter_schematic_")
    assert second_name.startswith("schematic_mylib_logic_v2_inverter_schematic_")


def test_view_load_raises_not_implemented():
    cell = MagicMock()
    cell.path = Path("/fake/lib/inverter")
    view = View(view_type="schematic", cell=cell, entry="schematic.py", generated=False)
    with pytest.raises(NotImplementedError):
        view.load()


def test_view_run_raises_type_error():
    cell = MagicMock()
    cell.path = Path("/fake/lib/inverter")
    view = View(view_type="schematic", cell=cell, entry="schematic.py", generated=False)
    with pytest.raises(TypeError, match="only valid on testbench"):
        view.run()

def test_schematic_view_load(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    (cell_dir / "schematic.py").write_text(
        "from monata.netlist import SubCircuit\n"
        "\n"
        "class Inverter(SubCircuit):\n"
        "    NAME = 'inverter'\n"
        "    NODES = ('vin', 'out', 'vdd', 'gnd')\n"
        "\n"
        "    def build(self):\n"
        "        pass\n"
    )

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SchematicView(cell=cell, entry="schematic.py", cls_name="Inverter")
    loaded_cls = view.load()

    assert loaded_cls.NAME == "inverter"
    assert loaded_cls.NODES == ('vin', 'out', 'vdd', 'gnd')


def test_schematic_view_load_exposes_library_helpers(tmp_path):
    lib_dir = tmp_path / "mylib"
    cell_dir = lib_dir / "inverter"
    cell_dir.mkdir(parents=True)

    (lib_dir / "_devices.py").write_text("DEVICE_NAME = 'nmos'\n")
    (cell_dir / "schematic.py").write_text(
        "from monata.netlist import SubCircuit\n"
        "from mylib._devices import DEVICE_NAME\n"
        "\n"
        "class Inverter(SubCircuit):\n"
        "    NAME = DEVICE_NAME\n"
        "    NODES = ('vin', 'out', 'vdd', 'gnd')\n"
        "\n"
        "    def build(self):\n"
        "        pass\n"
    )

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"
    cell.library = MagicMock()
    cell.library.name = "mylib"
    cell.library.path = lib_dir
    original_sys_path = list(sys.path)

    view = SchematicView(cell=cell, entry="schematic.py", cls_name="Inverter")
    loaded_cls = view.load()

    assert loaded_cls.NAME == "nmos"
    assert sys.path == original_sys_path


def test_schematic_view_load_file_not_found(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SchematicView(cell=cell, entry="schematic.py", cls_name="Inverter")
    with pytest.raises(FileNotFoundError):
        view.load()


def test_python_attribute_load_failure_restores_module_and_import_paths(tmp_path):
    lib_dir = tmp_path / "mylib"
    cell_dir = lib_dir / "inverter"
    cell_dir.mkdir(parents=True)
    (cell_dir / "schematic.py").write_text("VALUE = 1\n")

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"
    cell.library = MagicMock()
    cell.library.name = "mylib"
    cell.library.path = lib_dir
    view = SchematicView(cell=cell, entry="schematic.py", cls_name="MissingClass")
    module_name = view.python_module_name("schematic")
    previous_module = ModuleType(module_name)
    old_module = sys.modules.get(module_name)
    sys.modules[module_name] = previous_module
    original_sys_path = list(sys.path)

    try:
        with pytest.raises(AttributeError):
            view.load()

        assert sys.modules[module_name] is previous_module
        assert sys.path == original_sys_path
    finally:
        if old_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_module


def test_testbench_view_load(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    (cell_dir / "testbench.py").write_text(
        "def main(cell):\n"
        "    return f'ran testbench for {cell.name}'\n"
    )

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = TestbenchView(cell=cell, entry="testbench.py", function_name="main")
    func = view.load()
    assert callable(func)
    assert func(cell) == "ran testbench for inverter"


def test_testbench_view_load_exposes_cell_helpers(tmp_path):
    lib_dir = tmp_path / "mylib"
    cell_dir = lib_dir / "inverter"
    cell_dir.mkdir(parents=True)

    (cell_dir / "helpers.py").write_text("def message(cell):\n    return cell.name\n")
    (cell_dir / "testbench.py").write_text(
        "from helpers import message\n"
        "\n"
        "def main(cell):\n"
        "    return message(cell)\n"
    )

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"
    cell.library = MagicMock()
    cell.library.path = lib_dir
    original_sys_path = list(sys.path)

    view = TestbenchView(cell=cell, entry="testbench.py", function_name="main")
    func = view.load()

    assert func(cell) == "inverter"
    assert sys.path == original_sys_path


def test_testbench_view_run(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    (cell_dir / "testbench.py").write_text(
        "results = []\n"
        "def main(cell):\n"
        "    results.append(cell.name)\n"
        "    return {'cell': cell.name, 'results': results}\n"
    )

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = TestbenchView(cell=cell, entry="testbench.py", function_name="main")
    result = view.run()

    assert result == {"cell": "inverter", "results": ["inverter"]}


def test_testbench_view_load_file_not_found(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = TestbenchView(cell=cell, entry="testbench.py", function_name="main")
    with pytest.raises(FileNotFoundError):
        view.load()


class ViewTestAnd2(SubCircuit):
    NAME = "and2"
    NODES = ("a", "b", "out", "vdd", "gnd")

    def build(self):
        pass


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _RecordingExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, task):
        self.tasks.append(task)
        return _ImmediateFuture(_op_result_for_task(task))

    def map(self, tasks):
        self.tasks.extend(tasks)
        return [_ImmediateFuture(_op_result_for_task(task)) for task in tasks]


class _DelayRetryRecordingExecutor(_RecordingExecutor):
    def __init__(self):
        super().__init__()
        self._delay_attempts = 0

    def submit(self, task):
        self.tasks.append(task)
        return _ImmediateFuture(self._result_for_task(task))

    def map(self, tasks):
        self.tasks.extend(tasks)
        return [_ImmediateFuture(self._result_for_task(task)) for task in tasks]

    def _result_for_task(self, task):
        payload = _task_digital_metadata(task)
        task_kind = payload["digital_truth_table"]["task_kind"]
        if task_kind == "digital-single-bit-arc-sequence":
            self._delay_attempts += 1
            if self._delay_attempts == 1:
                return _unmeasurable_timing_result_for_task(task)
        return _op_result_for_task(task)


def _task_digital_metadata(task):
    return digital_task_metadata(task.metadata)


def _op_result_for_task(task):
    payload = _task_digital_metadata(task)
    task_kind = payload["digital_truth_table"]["task_kind"]
    if task_kind == "digital-single-bit-arc-sequence":
        return _digital_sequence_result_for_task(task)
    values = {}
    bits = tuple(int(bit) for bit in payload["stimulus"]["bits"])
    for output in task.output_names:
        values[output] = [float(bits[0] & bits[1])]
    return SimResult(status="ok", waveforms=values, sweep_var=None, corner=None, metadata={"task_metadata": task.metadata})


def _digital_sequence_result_for_task(task):
    payload = _task_digital_metadata(task)
    stimulus = payload["stimulus"]
    inputs = tuple(payload["digital_truth_table"]["inputs"])
    outputs = tuple(payload["digital_truth_table"]["outputs"])
    state_sequence = tuple(_bits_from_text(text) for text in stimulus["state_sequence"])
    initial_settle = float(stimulus["initial_settle"])
    slot_duration = float(stimulus["slot_duration"])
    transition = float(stimulus["transition"])
    stop = float(task.analysis_spec.stop)
    time = np.linspace(0.0, stop, int(stop / 0.01) + 1)
    waveforms: dict[str, np.ndarray] = {}
    for input_index, input_name in enumerate(inputs):
        points: list[tuple[float, float]] = []
        previous = float(state_sequence[0][input_index])
        _append_test_point(points, 0.0, previous)
        _append_test_point(points, initial_settle, previous)
        _append_test_point(points, initial_settle + slot_duration, previous)
        for state_index, state in enumerate(state_sequence[1:], start=1):
            boundary = initial_settle + state_index * slot_duration
            level = float(state[input_index])
            _append_test_point(points, boundary, previous)
            _append_test_point(points, boundary + transition, level)
            _append_test_point(points, initial_settle + (state_index + 1) * slot_duration, level)
            previous = level
        waveforms[input_name] = np.interp(time, [point[0] for point in points], [point[1] for point in points])
    for output_name in outputs:
        points = []
        previous = float(state_sequence[0][0] & state_sequence[0][1])
        _append_test_point(points, 0.0, previous)
        _append_test_point(points, initial_settle, previous)
        _append_test_point(points, initial_settle + slot_duration, previous)
        for state_index, state in enumerate(state_sequence[1:], start=1):
            boundary = initial_settle + state_index * slot_duration
            level = float(state[0] & state[1])
            if level == previous:
                _append_test_point(points, initial_settle + (state_index + 1) * slot_duration, level)
            else:
                crossing = boundary + transition / 2.0 + 0.2
                _append_test_point(points, boundary, previous)
                _append_test_point(points, crossing - transition / 2.0, previous)
                _append_test_point(points, crossing + transition / 2.0, level)
                _append_test_point(points, initial_settle + (state_index + 1) * slot_duration, level)
            previous = level
        waveforms[output_name] = np.interp(time, [point[0] for point in points], [point[1] for point in points])
    return SimResult(status="ok", waveforms=waveforms, sweep_var=time, corner=None, metadata={"task_metadata": task.metadata})


def _timing_result_for_task(task):
    payload = _task_digital_metadata(task)
    stimulus = payload["stimulus"]
    inputs = tuple(payload["digital_truth_table"]["inputs"])
    outputs = tuple(payload["digital_truth_table"]["outputs"])
    period = float(task.analysis_spec.stop) / float(stimulus["arcs"])
    transition = period * 0.1
    trigger_fraction = float(stimulus["trigger_fraction"])
    time = np.linspace(0.0, float(task.analysis_spec.stop), int(float(task.analysis_spec.stop) / 0.01) + 1)
    arcs = _and2_timing_arcs(period=period, transition=transition, trigger_fraction=trigger_fraction)
    waveforms: dict[str, np.ndarray] = {}
    for input_index, input_name in enumerate(inputs):
        waveforms[input_name] = _interpolate_timing_input(time, arcs, input_index)
    for output_name in outputs:
        waveforms[output_name] = _interpolate_and2_output(time, arcs, transition=transition, delay=0.2 * period)
    return SimResult(status="ok", waveforms=waveforms, sweep_var=time, corner=None, metadata={"task_metadata": task.metadata})


def _unmeasurable_timing_result_for_task(task):
    time = np.linspace(0.0, float(task.analysis_spec.stop), int(float(task.analysis_spec.stop) / 0.01) + 1)
    waveforms = {name: np.zeros_like(time) for name in task.output_names}
    return SimResult(status="ok", waveforms=waveforms, sweep_var=time, corner=None, metadata={"task_metadata": task.metadata})


def _and2_timing_arcs(*, period: float, transition: float, trigger_fraction: float):
    rows = []
    for from_inputs, input_index in (
        ((0, 1), 0),
        ((1, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0),
    ):
        to_inputs = tuple(1 - bit if index == input_index else bit for index, bit in enumerate(from_inputs))
        index = len(rows)
        start = index * period
        rows.append(
            {
                "from_inputs": from_inputs,
                "to_inputs": to_inputs,
                "from_output": from_inputs[0] & from_inputs[1],
                "to_output": to_inputs[0] & to_inputs[1],
                "input_index": input_index,
                "start": start,
                "reset_end": start + transition,
                "trigger_start": start + period * trigger_fraction,
                "trigger_end": start + period * trigger_fraction + transition,
                "stop": start + period,
            }
        )
    return rows


def _interpolate_timing_input(time, arcs, input_index):
    points: list[tuple[float, float]] = []
    previous_final: float | None = None
    for arc in arcs:
        initial = float(arc["from_inputs"][input_index])
        final = float(arc["to_inputs"][input_index])
        if previous_final is None:
            previous_final = initial
        _append_test_point(points, arc["start"], previous_final)
        _append_test_point(points, arc["reset_end"], initial)
        _append_test_point(points, arc["trigger_start"], initial)
        _append_test_point(points, arc["trigger_end"], final)
        _append_test_point(points, arc["stop"], final)
        previous_final = final
    return np.interp(time, [point[0] for point in points], [point[1] for point in points])


def _interpolate_and2_output(time, arcs, *, transition: float, delay: float):
    points: list[tuple[float, float]] = []
    previous_final: float | None = None
    for arc in arcs:
        initial = float(arc["from_output"])
        final = float(arc["to_output"])
        if previous_final is None:
            previous_final = initial
        crossing = (arc["trigger_start"] + arc["trigger_end"]) / 2.0 + delay
        _append_test_point(points, arc["start"], previous_final)
        _append_test_point(points, arc["reset_end"], initial)
        _append_test_point(points, crossing - transition / 2.0, initial)
        _append_test_point(points, crossing + transition / 2.0, final)
        _append_test_point(points, arc["stop"], final)
        previous_final = final
    return np.interp(time, [point[0] for point in points], [point[1] for point in points])


def _append_test_point(points: list[tuple[float, float]], time: float, value: float) -> None:
    if points and points[-1][0] == time and points[-1][1] == value:
        return
    points.append((float(time), float(value)))


def _bits_from_text(text: str) -> tuple[int, ...]:
    return tuple(int(bit) for bit in text)


def test_simulation_view_runs_tasks_through_executor(tmp_path):
    cell_dir = tmp_path / "sim_cell"
    cell_dir.mkdir()
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "sim_cell"
    task = MagicMock()
    executor = MagicMock()
    expected = SimResult(status="ok", waveforms={}, sweep_var=None, corner=None)
    executor.submit.return_value.result.return_value = expected

    view = SimulationView(cell=cell, entry="simulation.py")
    result = view.run(task=task, executor=executor, mode="op")

    assert result.status == "ok"
    assert result.mode == "op"
    assert result.results == (expected,)
    executor.submit.assert_called_once_with(task)


def test_simulation_view_assigns_artifact_dirs_from_task_options(tmp_path):
    cell_dir = tmp_path / "sim_cell"
    cell_dir.mkdir()
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "sim_cell"
    task = SimTask(
        circuit=MagicMock(),
        analysis_spec=TranSpec(step=1e-9, stop=2e-9),
        output_names=["out"],
        metadata={
            "owner": "unit-test",
            "case": "artifact-dir-assignment",
        },
    )
    executor = MagicMock()
    expected = SimResult(status="ok", waveforms={}, sweep_var=None, corner=None)
    executor.submit.return_value.result.return_value = expected

    view = SimulationView(cell=cell, entry="simulation.py")
    result = view.run(task=task, executor=executor, artifact_dir=tmp_path / "artifacts")

    submitted = executor.submit.call_args.args[0]
    assert result.results == (expected,)
    assert submitted.artifacts.directory == tmp_path / "artifacts" / "tasks" / "task-0000"
    assert submitted.metadata["simulation_artifact_index"] == 0
    assert (tmp_path / "artifacts" / "tasks" / "task-0000").is_dir()


def test_simulation_view_reports_task_progress_without_reordering_results(tmp_path):
    cell_dir = tmp_path / "sim_cell"
    cell_dir.mkdir()
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "sim_cell"
    tasks = [MagicMock(), MagicMock()]
    expected = (
        SimResult(
            status="ok",
            waveforms={},
            sweep_var=None,
            corner=None,
            metadata={"task_metadata": {"progress_sample": {"chunk_index": 0}}},
        ),
        SimResult(
            status="ok",
            waveforms={},
            sweep_var=None,
            corner=None,
            metadata={"task_metadata": {"progress_sample": {"chunk_index": 1}}},
        ),
    )

    class RecordingExecutor:
        def map(self, submitted_tasks):
            assert list(submitted_tasks) == tasks
            futures = []
            for result in reversed(expected):
                future = Future()
                future.set_result(result)
                futures.append(future)
            return list(reversed(futures))

    events = []
    view = SimulationView(cell=cell, entry="simulation.py")
    result = view.run_tasks(tasks, executor=RecordingExecutor(), progress=events.append)

    assert result == expected
    assert [event["event"] for event in events] == [
        "tasks_start",
        "task_done",
        "task_done",
        "tasks_done",
    ]
    assert events[0]["total"] == 2
    assert events[-1]["completed"] == 2
    assert sorted(event["task_index"] for event in events if event["event"] == "task_done") == [0, 1]


def test_digital_truth_table_view_uses_simulation_boundary(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    cell = lib.create_cell("verify_and2")
    (cell.path / "digital_truth_table.py").write_text(
        "from monata.sim.digital_table import DigitalTruthTableSpec\n"
        "\n"
        "SPEC = DigitalTruthTableSpec(\n"
        "    dut='and2',\n"
        "    inputs=('a', 'b'),\n"
        "    outputs=('out',),\n"
        "    expected=lambda bits: (bits[0] & bits[1],),\n"
        ")\n"
    )
    (cell.path / "simulation.py").write_text(
        "from monata.sim.digital_table import DigitalTruthTable\n"
        "from test_views import ViewTestAnd2\n"
        "\n"
        "def build_truth_table(cell, view, spec, mode):\n"
        "    assert view.simulation_view_type == 'simulation'\n"
        "    return DigitalTruthTable(\n"
        "        ViewTestAnd2,\n"
        "        inputs=spec.inputs,\n"
        "        outputs=spec.outputs,\n"
        "        expected=spec.expected,\n"
        "        period=1.0,\n"
        "        step=0.01,\n"
        "        transition=0.1,\n"
        "    )\n"
    )
    cell.create_view("digital_truth_table", mode="transient")
    cell.create_view("simulation", function_name="build_truth_table")
    view = cell["digital_truth_table"]
    executor = _RecordingExecutor()

    assert isinstance(view, DigitalTruthTableView)
    artifact_dir = tmp_path / "artifacts" / "verify_and2"
    result = view.run(executor=executor, artifact_dir=artifact_dir)

    assert result.mode == "transient"
    assert [row.as_dict()["status"] for row in result] == ["PASS", "PASS", "PASS", "PASS"]
    assert result.max_propagation_delay == pytest.approx(0.2)
    assert len(executor.tasks) == 1
    assert (artifact_dir / "tasks" / "task-0000").is_dir()
    measurements = json.loads((artifact_dir / "measurements.json").read_text())
    assert measurements["truth_table"]["status"] == "PASS"
    assert measurements["max_propagation_delay"]["value"] == pytest.approx(0.2)
    assert measurements["max_propagation_delay"]["coverage"]["kind"] == "directed_single_bit_exhaustive"
    run = json.loads((artifact_dir / "run.json").read_text())
    assert run["measurements"] == ["max_propagation_delay", "truth_table"]
    assert [task["index"] for task in run["tasks"]] == [0]
    assert run["tasks"][0]["stimulus"]["kind"] == "digital_single_bit_arc_sequence"
    assert run["tasks"][0]["measurements"] == ["truth_table", "max_propagation_delay"]


def test_digital_truth_table_view_does_not_retry_unmeasurable_delay_chunks(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    cell = lib.create_cell("verify_and2")
    (cell.path / "digital_truth_table.py").write_text(
        "from monata.sim.digital_table import DigitalTruthTableSpec\n"
        "\n"
        "SPEC = DigitalTruthTableSpec(\n"
        "    dut='and2',\n"
        "    inputs=('a', 'b'),\n"
        "    outputs=('out',),\n"
        "    expected=lambda bits: (bits[0] & bits[1],),\n"
        ")\n"
    )
    (cell.path / "simulation.py").write_text(
        "from monata.sim.digital_table import DigitalTruthTable\n"
        "from test_views import ViewTestAnd2\n"
        "\n"
        "def build_truth_table(cell, spec, mode):\n"
        "    return DigitalTruthTable(\n"
        "        ViewTestAnd2,\n"
        "        inputs=spec.inputs,\n"
        "        outputs=spec.outputs,\n"
        "        expected=spec.expected,\n"
        "        period=1.0,\n"
        "        step=0.01,\n"
        "        transition=0.1,\n"
        "    )\n"
    )
    cell.create_view("digital_truth_table", mode="transient")
    cell.create_view("simulation", function_name="build_truth_table")
    view = cell["digital_truth_table"]
    executor = _DelayRetryRecordingExecutor()
    artifact_dir = tmp_path / "artifacts" / "verify_and2"

    with pytest.raises(RuntimeError, match="did not cross threshold"):
        view.run(executor=executor, artifact_dir=artifact_dir)

    assert len(executor.tasks) == 1
    assert (artifact_dir / "tasks" / "task-0000").is_dir()
    assert not (artifact_dir / "run.json").exists()


def test_digital_truth_table_view_mapping_requires_dut(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    cell = lib.create_cell("verify")
    (cell.path / "digital_truth_table.py").write_text(
        "from monata.sim.digital_table import DigitalTruthTableSpec\n"
        "SPEC = DigitalTruthTableSpec(dut='and2', inputs=('a',), outputs=('out',))\n"
    )
    (cell.path / "simulation.py").write_text(
        "def build_truth_table(cell, spec):\n"
        "    return {'inputs': ('a',), 'outputs': ('out',)}\n"
    )
    cell.create_view("simulation", function_name="build_truth_table")
    view = DigitalTruthTableView(cell=cell, entry="digital_truth_table.py")

    with pytest.raises(ValueError, match="requires string 'dut'"):
        view.load()


def test_digital_truth_table_view_mapping_accepts_expected_rows(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    dut = lib.create_cell("and2")
    (dut.path / "schematic.py").write_text(
        "from test_views import ViewTestAnd2\n"
        "main = ViewTestAnd2\n"
    )
    dut.create_view("schematic", cls_name="main")
    cell = lib.create_cell("and2_tb")
    (cell.path / "verification.py").write_text(
        "from monata.sim.digital_table import DigitalTruthTableSpec\n"
        "SPEC = DigitalTruthTableSpec(dut='and2', inputs=('a', 'b'), outputs=('out',))\n"
    )
    (cell.path / "simulation.py").write_text(
        "def build_truth_table(cell, spec):\n"
        "    return {\n"
        "        'dut': 'and2',\n"
        "        'inputs': ('a', 'b'),\n"
        "        'outputs': ('out',),\n"
        "        'expected': [\n"
        "            {'inputs': '00', 'expected': '0'},\n"
        "            {'inputs': '01', 'expected': '0'},\n"
        "            {'inputs': '10', 'expected': '0'},\n"
        "            {'inputs': '11', 'expected': '1'},\n"
        "        ],\n"
        "    }\n"
    )
    cell.create_view("simulation", function_name="build_truth_table")
    view = DigitalTruthTableView(cell=cell, entry="verification.py")

    table = view.load()

    assert table.expected_for((1, 1)) == (1,)
    assert table.expected_for((0, 1)) == (0,)


def test_view_registry_object_isolated_from_default_registry():
    registry = ViewRegistry()

    registry.register("layout", lambda owner, cfg: View("layout", owner, str(cfg["entry"])))

    assert registry.list_view_types() == ["layout"]
    assert registry.get_factory("layout") is not None
    assert get_view_factory("layout") is None


def test_default_registry_includes_simulation_and_digital_truth_table():
    assert get_view_factory("simulation") is not None
    assert get_view_factory("digital_truth_table") is not None


@pytest.mark.parametrize("view_type", ["../layout", "bad view", "evil\nview", "tab\tview", ""])
def test_view_registry_rejects_unsafe_view_types(view_type):
    registry = ViewRegistry()

    with pytest.raises(ValueError, match="view type must be a single safe path segment"):
        registry.register(view_type, lambda owner, cfg: View(str(view_type), owner, str(cfg["entry"])))

    assert registry.list_view_types() == []


def test_view_registry_normalizes_config_factory_mapping():
    registry = ViewRegistry()
    registry.register(
        "layout",
        lambda owner, cfg: View("layout", owner, str(cfg["entry"])),
        config_factory=lambda view_type, options, schema: MappingProxyType({"entry": "layout.gds"}),
    )

    config = registry.create_config("layout", {})

    assert type(config) is dict
    assert config == {"entry": "layout.gds"}


def test_infer_pin_direction_power():
    assert infer_pin_direction("vdd") == "inout"
    assert infer_pin_direction("vcc") == "inout"
    assert infer_pin_direction("gnd") == "inout"
    assert infer_pin_direction("vss") == "inout"


def test_infer_pin_direction_power_takes_priority():
    assert infer_pin_direction("vdd_in") == "inout"
    assert infer_pin_direction("gnd_out") == "inout"


def test_infer_pin_direction_output():
    assert infer_pin_direction("out") == "output"
    assert infer_pin_direction("output") == "output"
    assert infer_pin_direction("dout") == "output"


def test_infer_pin_direction_input():
    assert infer_pin_direction("in") == "input"
    assert infer_pin_direction("inp") == "input"
    assert infer_pin_direction("inn") == "input"
    assert infer_pin_direction("vin") == "input"


def test_infer_pin_direction_default():
    assert infer_pin_direction("clk") == "inout"
    assert infer_pin_direction("data") == "inout"


def test_symbol_view_load(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    (cell_dir / "symbol.toml").write_text(
        '[symbol]\nname = "inverter"\n\n'
        '[[pins]]\nname = "vin"\ndirection = "input"\n\n'
        '[[pins]]\nname = "out"\ndirection = "output"\n'
    )

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SymbolView(cell=cell, entry="symbol.toml")
    result = view.load()
    assert result["name"] == "inverter"
    assert len(result["pins"]) == 2
    assert result["pins"][0]["name"] == "vin"
    assert result["pins"][0]["direction"] == "input"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            'unexpected = true\n\n[symbol]\nname = "inverter"\n',
            "symbol.toml has unknown fields: unexpected",
        ),
        (
            '[symbol]\nname = "inverter"\nunexpected = true\n',
            "symbol table has unknown fields: unexpected",
        ),
        (
            '[symbol]\nname = "inverter"\n\n'
            '[[pins]]\nname = "vin"\ndirection = "input"\nunexpected = true\n',
            r"symbol pins\[0\] has unknown fields: unexpected",
        ),
        (
            'pins = "vin"\n\n[symbol]\nname = "inverter"\n',
            "symbol pins must be an array of tables",
        ),
    ],
)
def test_symbol_view_load_rejects_invalid_schema(tmp_path, body, message):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)
    (cell_dir / "symbol.toml").write_text(body)
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SymbolView(cell=cell, entry="symbol.toml")

    with pytest.raises(ValueError, match=message):
        view.load()


def test_symbol_view_load_not_generated(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SymbolView(cell=cell, entry="symbol.toml")
    with pytest.raises(ViewNotGeneratedError):
        view.load()


def test_netlist_view_load(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    scs_content = "subckt inverter(vin out vdd gnd)\n  // body\nends\n"
    (cell_dir / "netlist.scs").write_text(scs_content)

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = NetlistView(cell=cell, entry="netlist.scs")
    result = view.load()
    assert result == cell_dir / "netlist.scs"
    assert result.exists()


def test_netlist_view_load_not_generated(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = NetlistView(cell=cell, entry="netlist.scs")
    with pytest.raises(ViewNotGeneratedError):
        view.load()
