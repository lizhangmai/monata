import numpy as np
import pytest
from threading import Event
import time

from monata.sim.executor import Executor, LocalExecutor
from monata.sim.results import SimResult
from monata.sim.task import SimTask
from monata.sim.analysis_spec import TranSpec


def test_executor_is_abstract():
    with pytest.raises(TypeError):
        Executor()  # type: ignore[abstract]


def test_local_executor_creation():
    ex = LocalExecutor(max_workers=2, backend="ngspice-subprocess")
    assert ex._max_workers == 2
    assert ex._backend == "ngspice-subprocess"


def test_local_executor_defaults():
    ex = LocalExecutor()
    assert ex._backend is None
    assert ex._max_workers is None


def test_local_executor_backend_sets_default_simulator(monkeypatch):
    ex = LocalExecutor(backend="custom-backend")
    spec = TranSpec(stop=1e-6)
    task = SimTask(circuit=None, analysis_spec=spec)
    seen = {}

    def mock_execute_task(t):
        seen["simulator"] = t.simulator
        return SimResult(
            status="ok",
            waveforms={"out": np.ones(1)},
            sweep_var=np.zeros(1),
            corner=t.corner,
            metadata={"simulator": t.simulator},
        )

    import monata.sim._backend as backend_module

    monkeypatch.setattr(backend_module, "execute_task", mock_execute_task)
    result = ex.submit(task).result()

    assert result.metadata["simulator"] == "custom-backend"
    assert seen["simulator"] == "custom-backend"
    assert task.simulator == "ngspice-subprocess"


def test_local_executor_submit_mock(monkeypatch):
    """Test submit with a mock execution function."""
    ex = LocalExecutor(max_workers=1)

    spec = TranSpec(stop=1e-6)
    task = SimTask(circuit=None, analysis_spec=spec)

    def mock_execute(t):
        return SimResult(
            status="ok",
            waveforms={"out": np.ones(10)},
            sweep_var=np.linspace(0, 1e-6, 10),
            corner=t.corner,
            metadata={"simulator": "mock"},
        )

    monkeypatch.setattr(ex, "_execute", mock_execute)
    future = ex.submit(task)
    result = future.result()
    assert result.status == "ok"
    assert len(result.waveforms["out"]) == 10


def test_local_executor_submit_reuses_configured_worker_pool(monkeypatch):
    ex = LocalExecutor(max_workers=1)
    spec = TranSpec(stop=1e-6)
    tasks = [SimTask(circuit=None, analysis_spec=spec) for _ in range(2)]
    first_started = Event()
    release_first = Event()
    calls = []

    def mock_execute(task):
        calls.append(task)
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(timeout=1)
        return SimResult(
            status="ok",
            waveforms={"out": np.ones(1)},
            sweep_var=np.zeros(1),
            corner=task.corner,
            metadata={},
        )

    monkeypatch.setattr(ex, "_execute", mock_execute)
    first = ex.submit(tasks[0])
    assert first_started.wait(timeout=1)
    second = ex.submit(tasks[1])
    time.sleep(0.05)

    assert len(calls) == 1
    release_first.set()
    assert first.result(timeout=1).status == "ok"
    assert second.result(timeout=1).status == "ok"
    assert calls == tasks


def test_local_executor_map_mock(monkeypatch):
    """Test map with multiple tasks."""
    ex = LocalExecutor(max_workers=2)

    spec = TranSpec(stop=1e-6)
    tasks = [SimTask(circuit=None, analysis_spec=spec) for _ in range(3)]

    def mock_execute(t):
        return SimResult(
            status="ok",
            waveforms={"out": np.ones(5)},
            sweep_var=np.linspace(0, 1e-6, 5),
            corner=t.corner,
            metadata={},
        )

    monkeypatch.setattr(ex, "_execute", mock_execute)
    futures = ex.map(tasks)
    assert len(futures) == 3
    for f in futures:
        assert f.result().status == "ok"


def test_local_executor_map_empty():
    ex = LocalExecutor(max_workers=2)

    assert ex.map([]) == []
