import json
from pathlib import Path
from collections.abc import Mapping
from concurrent.futures import Future
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from monata.errors import ViewNotGeneratedError
from monata.library import Library
from monata.netlist import SubCircuit
from monata.schematic import SchematicBuilder
from monata.sim.core import SimResult, SimTask, TranSpec
from monata.sim.digital_recipe import DigitalSimulationRecipe
from monata.sim.digital_table import DigitalTruthTableSpec, ExpectedTable, build_digital_truth_table_from_spec
from monata.sim.digital_plan import digital_task_metadata
from monata.views.declarative import SchematicJsonView, SymbolJsonView, TestbenchJsonView
from monata.views.base import View
from monata.views.digital_truth_table import DigitalTruthTableView
from monata.views.netlist import NetlistView
from monata.views.simulation import SimulationRecipeView, SimulationView
from monata.views.symbol import infer_pin_direction
from monata.views.registry import ViewRegistry, create_registered_view_config, get_view_factory

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


def test_view_has_no_python_execution_helpers():
    for name in ("load_python_entry", "load_python_attribute", "load_trusted", "python_module_name"):
        assert not hasattr(View, name)


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
    return SimResult(status="ok", waveforms=values, sweep_var=None, corner=None, metadata=task.metadata)


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
    return SimResult(status="ok", waveforms=waveforms, sweep_var=time, corner=None, metadata=task.metadata)


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
    return SimResult(status="ok", waveforms=waveforms, sweep_var=time, corner=None, metadata=task.metadata)


def _unmeasurable_timing_result_for_task(task):
    time = np.linspace(0.0, float(task.analysis_spec.stop), int(float(task.analysis_spec.stop) / 0.01) + 1)
    waveforms = {name: np.zeros_like(time) for name in task.output_names}
    return SimResult(status="ok", waveforms=waveforms, sweep_var=time, corner=None, metadata=task.metadata)


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


AND2_EXPECTED_ROWS = [
    {"inputs": "00", "expected": "0"},
    {"inputs": "01", "expected": "0"},
    {"inputs": "10", "expected": "0"},
    {"inputs": "11", "expected": "1"},
]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_config(**overrides):
    values = {
        "model": "toy",
        "vdd": 1.0,
        "threshold": None,
        "corner": None,
        "model_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _create_and2_dut(lib: Library, *, marker: Path | None = None):
    dut = lib.create_cell("and2")
    (
        SchematicBuilder("and2")
        .pin("a", direction="input")
        .pin("b", direction="input")
        .pin("out", direction="output")
        .pin("vdd", direction="power")
        .pin("0", direction="ground")
        .write(dut.path / "schematic.monata.json")
    )
    if marker is not None:
        (dut.path / "schematic.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
            encoding="utf-8",
        )
    dut.create_view("schematic")
    return dut


def _write_and2_truth_table_views(cell) -> None:
    _write_json(cell.path / "expected.json", {"rows": AND2_EXPECTED_ROWS})
    _write_json(
        cell.path / "digital_truth_table.monata.json",
        {
            "schema_version": 1,
            "view_type": "monata-digital-truth-table",
            "dut": "and2",
            "stage": "unit",
            "simulation_mode": "transient",
            "inputs": ["a", "b"],
            "outputs": ["out"],
            "dependencies": [],
            "rails": {"vdd": "vdd", "vss": "0"},
            "complement_inputs": {},
            "expected": {
                "entry": "expected.json",
                "format": "monata-expected-table-json",
            },
        },
    )
    _write_json(
        cell.path / "simulation.monata.json",
        {
            "schema_version": 1,
            "view_type": "monata-simulation",
            "recipe_kind": "digital_truth_table_transient",
            "timing": {
                "period": 1.0,
                "step": 0.01,
                "transition": 0.1,
            },
            "observation": {
                "cycles_per_vector": 2,
                "slots_per_task": 64,
                "uic": True,
            },
            "model_profiles": {
                "toy": {
                    "metadata": {"profile_marker": "toy"},
                },
            },
        },
    )


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
            metadata={"progress_sample": {"chunk_index": 0}},
        ),
        SimResult(
            status="ok",
            waveforms={},
            sweep_var=None,
            corner=None,
            metadata={"progress_sample": {"chunk_index": 1}},
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


def test_simulation_view_run_requires_explicit_task(tmp_path):
    cell_dir = tmp_path / "sim_cell"
    cell_dir.mkdir()
    (cell_dir / "simulation.py").write_text("raise RuntimeError('should not import')\n", encoding="utf-8")
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "sim_cell"

    view = SimulationView(cell=cell, entry="simulation.py")

    with pytest.raises(TypeError, match="requires explicit task or tasks"):
        view.run()


def test_simulation_recipe_view_loads_recipe_without_python_fallback(tmp_path):
    cell_dir = tmp_path / "sim_cell"
    cell_dir.mkdir()
    _write_json(
        cell_dir / "simulation.monata.json",
        {
            "schema_version": 1,
            "view_type": "monata-simulation",
            "recipe_kind": "digital_truth_table_transient",
            "timing": {"period": 1.0},
            "model_profiles": {"toy": {}},
        },
    )
    (cell_dir / "simulation.py").write_text("raise RuntimeError('should not import')\n", encoding="utf-8")
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "sim_cell"

    view = SimulationRecipeView(cell=cell, entry="simulation.monata.json")
    recipe = view.load()

    assert isinstance(recipe, DigitalSimulationRecipe)
    with pytest.raises(TypeError, match="monata-simulation-json views are recipes"):
        view.run()


def test_digital_truth_table_view_uses_simulation_boundary(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    _create_and2_dut(lib)
    cell = lib.create_cell("verify_and2")
    (cell.path / "verification.py").write_text("raise RuntimeError('verification imported')\n", encoding="utf-8")
    (cell.path / "simulation.py").write_text("raise RuntimeError('simulation imported')\n", encoding="utf-8")
    _write_and2_truth_table_views(cell)
    cell.create_view("digital_truth_table")
    cell.create_view("simulation")
    view = cell["digital_truth_table"]
    executor = _RecordingExecutor()

    assert isinstance(view, DigitalTruthTableView)
    assert view.simulation_view_type == "simulation"
    artifact_dir = tmp_path / "artifacts" / "verify_and2"
    result = view.run(executor=executor, artifact_dir=artifact_dir, run_config=_run_config())

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
    _create_and2_dut(lib)
    cell = lib.create_cell("verify_and2")
    _write_and2_truth_table_views(cell)
    cell.create_view("digital_truth_table")
    cell.create_view("simulation")
    view = cell["digital_truth_table"]
    executor = _DelayRetryRecordingExecutor()
    artifact_dir = tmp_path / "artifacts" / "verify_and2"

    with pytest.raises(RuntimeError, match="did not cross threshold"):
        view.run(executor=executor, artifact_dir=artifact_dir, run_config=_run_config())

    assert len(executor.tasks) == 1
    assert (artifact_dir / "tasks" / "task-0000").is_dir()
    assert not (artifact_dir / "run.json").exists()


def test_digital_truth_table_json_rejects_unknown_fields(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    _create_and2_dut(lib)
    cell = lib.create_cell("verify_and2")
    _write_and2_truth_table_views(cell)
    payload = json.loads((cell.path / "digital_truth_table.monata.json").read_text())
    payload["transient_observation"] = {"cycles_per_vector": 1}
    _write_json(cell.path / "digital_truth_table.monata.json", payload)
    cell.create_view("digital_truth_table")
    cell.create_view("simulation")

    with pytest.raises(ValueError, match="unknown digital truth-table spec fields: transient_observation"):
        cell["digital_truth_table"].spec()


def test_digital_truth_table_json_rejects_simulation_view_dispatch_metadata(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    _create_and2_dut(lib)
    cell = lib.create_cell("verify_and2")
    _write_and2_truth_table_views(cell)
    payload = json.loads((cell.path / "digital_truth_table.monata.json").read_text())
    payload["simulation_view"] = "custom_runner"
    _write_json(cell.path / "digital_truth_table.monata.json", payload)
    cell.create_view("digital_truth_table")

    with pytest.raises(ValueError, match="unknown digital truth-table spec fields: simulation_view"):
        cell["digital_truth_table"].spec()


def test_digital_truth_table_config_rejects_simulation_view_dispatch_metadata():
    with pytest.raises(ValueError, match="unknown view config fields: simulation_view"):
        create_registered_view_config("digital_truth_table", simulation_view="custom_runner")


def test_digital_truth_table_json_rejects_simulation_view_metadata(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    _create_and2_dut(lib)
    cell = lib.create_cell("verify_and2")
    _write_and2_truth_table_views(cell)
    payload = json.loads((cell.path / "digital_truth_table.monata.json").read_text())
    payload["metadata"] = {"simulation_view": "custom_runner"}
    _write_json(cell.path / "digital_truth_table.monata.json", payload)
    cell.create_view("digital_truth_table")

    with pytest.raises(ValueError, match="metadata cannot include simulation_view"):
        cell["digital_truth_table"].spec()


def test_digital_truth_table_json_loads_expected_rows_without_neighbor_python(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    marker = tmp_path / "executed.txt"
    _create_and2_dut(lib, marker=marker)
    cell = lib.create_cell("and2_tb")
    (cell.path / "verification.py").write_text("raise RuntimeError('verification imported')\n", encoding="utf-8")
    (cell.path / "simulation.py").write_text("raise RuntimeError('simulation imported')\n", encoding="utf-8")
    _write_and2_truth_table_views(cell)
    cell.create_view("digital_truth_table")
    cell.create_view("simulation")
    view = cell["digital_truth_table"]

    table = view.load(run_config=_run_config())

    assert table.expected_for((1, 1)) == (1,)
    assert table.expected_for((0, 1)) == (0,)
    assert not marker.exists()


@pytest.mark.parametrize("entry", ["../expected.json", "/tmp/expected.json"])
def test_digital_truth_table_expected_reference_rejects_unsafe_paths(tmp_path, entry):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    _create_and2_dut(lib)
    cell = lib.create_cell("and2_tb")
    _write_and2_truth_table_views(cell)
    payload = json.loads((cell.path / "digital_truth_table.monata.json").read_text())
    payload["expected"] = {"entry": entry, "format": "monata-expected-table-json"}
    _write_json(cell.path / "digital_truth_table.monata.json", payload)
    cell.create_view("digital_truth_table")
    cell.create_view("simulation")

    with pytest.raises(ValueError, match="expected.entry must be relative"):
        cell["digital_truth_table"].spec()


@pytest.mark.parametrize("entry", ["../simulation.monata.json", "/tmp/simulation.monata.json"])
def test_simulation_json_entry_rejects_unsafe_paths(tmp_path, entry):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    cell = lib.create_cell("and2_tb")
    cell.create_view("simulation", entry=entry)

    with pytest.raises(ValueError, match="simulation.entry must be relative"):
        cell["simulation"].load()


def test_digital_truth_table_spec_mapping_normalizes_json_objects():
    expected = ExpectedTable.from_rows(AND2_EXPECTED_ROWS)
    spec = DigitalTruthTableSpec.from_mapping(
        {
            "schema_version": 1,
            "view_type": "monata-digital-truth-table",
            "dut": "and2",
            "stage": "unit",
            "simulation_mode": "transient",
            "inputs": ["a", "b"],
            "outputs": ["out"],
            "dependencies": [],
            "rails": {"vdd": "VDD", "vss": "VSS"},
            "complement_inputs": {"a": "a_bar", "b": "b_bar"},
            "expected": {"entry": "expected.json", "format": "monata-expected-table-json"},
        },
        expected=expected,
    )

    assert spec.rails == ("VDD", "VSS")
    assert spec.complement_inputs == ("a_bar", "b_bar")
    assert spec.to_mapping()["rails"] == {"vdd": "VDD", "vss": "VSS"}
    assert "simulation_view" not in spec.to_mapping()


def test_digital_simulation_recipe_rejects_unknown_profile_fields():
    with pytest.raises(ValueError, match="unknown digital model profile fields: python"):
        DigitalSimulationRecipe.from_mapping(
            {
                "schema_version": 1,
                "view_type": "monata-simulation",
                "recipe_kind": "digital_truth_table_transient",
                "timing": {"period": 1.0},
                "model_profiles": {"toy": {"python": "not allowed"}},
            }
        )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"model_config": {"openvaf_bin": "/bin/echo"}}, "unknown digital simulation recipe fields: model_config"),
        ({"openvaf_bin": "/bin/echo"}, "unknown digital simulation recipe fields: openvaf_bin"),
        ({"source_paths": {"bsimcmg": "model.va"}}, "unknown digital simulation recipe fields: source_paths"),
        ({"external_osdi_paths": ["artifact.osdi"]}, "unknown digital simulation recipe fields: external_osdi_paths"),
        ({"cache_dir": "/tmp/monata-cache"}, "unknown digital simulation recipe fields: cache_dir"),
        ({"artifacts": {"directory": "artifacts"}}, "unknown digital simulation recipe fields: artifacts"),
    ],
)
def test_digital_simulation_recipe_rejects_strong_top_level_fields(extra, message):
    payload = {
        "schema_version": 1,
        "view_type": "monata-simulation",
        "recipe_kind": "digital_truth_table_transient",
        "timing": {"period": 1.0},
        "model_profiles": {"toy": {}},
        **extra,
    }

    with pytest.raises(ValueError, match=message):
        DigitalSimulationRecipe.from_mapping(payload)


@pytest.mark.parametrize(
    ("profile_extra", "message"),
    [
        (
            {"model_config": {"cache_dir": "/tmp/monata-cache"}},
            "unknown digital model profile fields: model_config",
        ),
        (
            {"model_config": {"source_paths": {"bsimcmg": "model.va"}}},
            "unknown digital model profile fields: model_config",
        ),
        (
            {"model_config": {"external_osdi_paths": ["artifact.osdi"]}},
            "unknown digital model profile fields: model_config",
        ),
        (
            {"artifacts": {"directory": "artifacts"}},
            "unknown digital model profile fields: artifacts",
        ),
    ],
)
def test_digital_simulation_recipe_rejects_nested_strong_profile_fields(profile_extra, message):
    with pytest.raises(ValueError, match=message):
        DigitalSimulationRecipe.from_mapping(
            {
                "schema_version": 1,
                "view_type": "monata-simulation",
                "recipe_kind": "digital_truth_table_transient",
                "timing": {"period": 1.0},
                "model_profiles": {"toy": profile_extra},
            }
        )


def test_digital_truth_table_spec_helper_uses_data_schematic_without_neighbor_python(tmp_path):
    lib = Library.create(tmp_path / "mylib", name="mylib")
    dut = lib.create_cell("and2")
    marker = tmp_path / "executed.txt"
    (
        SchematicBuilder("and2")
        .pin("a", direction="input")
        .pin("b", direction="input")
        .pin("out", direction="output")
        .write(dut.path / "schematic.monata.json")
    )
    (dut.path / "schematic.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    dut.create_view("schematic")

    table = build_digital_truth_table_from_spec(
        lib,
        DigitalTruthTableSpec(
            dut="and2",
            inputs=("a", "b"),
            outputs=("out",),
            expected=lambda bits: (bits[0] & bits[1],),
        ),
        run_config=SimpleNamespace(vdd=1.0, threshold=None, corner=None, model_config=None),
        mode="transient",
    )

    assert table.dut_name == "and2"
    assert not marker.exists()


def test_view_registry_object_isolated_from_default_registry():
    registry = ViewRegistry()

    registry.register("layout", lambda owner, cfg: View("layout", owner, str(cfg["entry"])))

    assert registry.list_view_types() == ["layout"]
    assert registry.get_factory("layout") is not None
    assert get_view_factory("layout") is None


def test_default_registry_includes_simulation_and_digital_truth_table():
    assert get_view_factory("simulation") is not None
    assert get_view_factory("digital_truth_table") is not None
    assert get_view_factory("testbench_py") is None
    assert create_registered_view_config("simulation") == {
        "entry": "simulation.monata.json",
        "format": "monata-simulation-json",
        "schema_version": 1,
    }
    assert create_registered_view_config("digital_truth_table") == {
        "entry": "digital_truth_table.monata.json",
        "format": "monata-digital-truth-table-json",
        "schema_version": 1,
    }
    with pytest.raises(ValueError, match="Python testbench cellviews are no longer supported"):
        create_registered_view_config("testbench_py")
    with pytest.raises(ValueError, match="Python testbench cellviews are no longer supported"):
        create_registered_view_config("testbench", format="python-testbench")
    with pytest.raises(ValueError, match="unknown view config fields: mode"):
        create_registered_view_config("digital_truth_table", mode="transient")


@pytest.mark.parametrize(
    ("view_type", "metadata"),
    [
        ("simulation", {"trusted": False}),
        ("simulation", {"function": ""}),
        ("simulation", {"function_name": ""}),
        ("digital_truth_table", {"trusted": False}),
        ("digital_truth_table", {"function": ""}),
        ("digital_truth_table", {"function_name": ""}),
        ("testbench", {"trusted": False}),
        ("testbench", {"function": ""}),
        ("testbench", {"function_name": ""}),
    ],
)
def test_default_registry_rejects_executable_metadata_shapes(view_type, metadata):
    with pytest.raises(ValueError, match="cannot include executable Python metadata"):
        create_registered_view_config(view_type, **metadata)


def test_canonical_digital_view_sources_have_no_python_or_smoke_residue():
    source_root = Path(__file__).resolve().parents[1] / "src" / "monata"
    source_text = "\n".join(
        (source_root / relative).read_text(encoding="utf-8")
        for relative in (
            "views/base.py",
            "views/digital_truth_table.py",
            "views/simulation.py",
            "views/registry.py",
        )
    )

    for token in (
        "load_python_attribute",
        "load_python_entry",
        "load_trusted",
        "run_trusted",
        "testbench_py",
        "python-testbench",
        "trusted = true",
        "verification.py",
        "simulation.py",
        "SPEC",
        "smoke",
    ):
        assert token not in source_text


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


def test_symbol_json_view_load(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    (cell_dir / "symbol.monata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "view_type": "symbol",
                "name": "inverter",
                "pins": [
                    {"name": "vin", "direction": "input"},
                    {"name": "out", "direction": "output"},
                ],
            }
        )
    )

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SymbolJsonView(cell=cell, entry="symbol.monata.json")
    result = view.load()
    assert result["name"] == "inverter"
    assert len(result["pins"]) == 2
    assert result["pins"][0]["name"] == "vin"
    assert result["pins"][0]["direction"] == "input"


@pytest.mark.parametrize("entry", ["../symbol.monata.json", "/tmp/symbol.monata.json"])
def test_symbol_json_view_rejects_unsafe_entry_paths(tmp_path, entry):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SymbolJsonView(cell=cell, entry=entry)

    with pytest.raises(ValueError, match="symbol.entry must be relative"):
        view.load()


@pytest.mark.parametrize("entry", ["../schematic.monata.json", "/tmp/schematic.monata.json"])
def test_schematic_json_view_rejects_unsafe_entry_paths(tmp_path, entry):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SchematicJsonView(cell=cell, entry=entry)

    with pytest.raises(ValueError, match="schematic.entry must be relative"):
        view.load()


@pytest.mark.parametrize("entry", ["../testbench.monata.json", "/tmp/testbench.monata.json"])
def test_testbench_json_view_rejects_unsafe_entry_paths(tmp_path, entry):
    cell_dir = tmp_path / "inverter_tb"
    cell_dir.mkdir(parents=True)
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter_tb"

    view = TestbenchJsonView(cell=cell, entry=entry)

    with pytest.raises(ValueError, match="testbench.entry must be relative"):
        view.load()


def test_symbol_json_view_load_rejects_invalid_schema(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)
    (cell_dir / "symbol.monata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "view_type": "symbol",
                "pins": "vin",
            }
        )
    )
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SymbolJsonView(cell=cell, entry="symbol.monata.json")

    with pytest.raises(ValueError, match="expected an array"):
        view.load()


def test_symbol_json_view_load_not_generated(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = SymbolJsonView(cell=cell, entry="symbol.monata.json")
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


@pytest.mark.parametrize("entry", ["../netlist.scs", "/tmp/netlist.scs"])
def test_netlist_view_rejects_unsafe_entry_paths(tmp_path, entry):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)
    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = NetlistView(cell=cell, entry=entry)

    with pytest.raises(ValueError, match="netlist.entry must be relative"):
        view.load()


def test_netlist_view_load_not_generated(tmp_path):
    cell_dir = tmp_path / "inverter"
    cell_dir.mkdir(parents=True)

    cell = MagicMock()
    cell.path = cell_dir
    cell.name = "inverter"

    view = NetlistView(cell=cell, entry="netlist.scs")
    with pytest.raises(ViewNotGeneratedError):
        view.load()
