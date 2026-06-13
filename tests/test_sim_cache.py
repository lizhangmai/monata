import numpy as np
import pytest

from monata.netlist import Circuit
from monata.sim.analysis_spec import TranSpec
from monata.sim.cache import SimulationCache, task_fingerprint
from monata.sim.results import SimResult
from monata.sim.task import SimTask


def _circuit() -> Circuit:
    circuit = Circuit("cache rc")
    circuit.vdc("1", "in", "0", 1.0)
    circuit.resistor("1", "in", "out", "1k")
    circuit.capacitor("1", "out", "0", "1p")
    circuit.save("v(out)")
    return circuit


def _task(circuit: Circuit | None = None, **overrides) -> SimTask:
    values = {
        "circuit": circuit or _circuit(),
        "analysis_spec": TranSpec(stop=1e-9, step=1e-10),
        "simulator": "ngspice-subprocess",
        "output_names": ("out",),
    }
    values.update(overrides)
    return SimTask(**values)


def _result() -> SimResult:
    return SimResult(
        status="ok",
        waveforms={"out": np.array([0.0, 1.0])},
        sweep_var=np.array([0.0, 1e-9]),
        corner={"name": "tt", "temperature": 27},
        metadata={"analysis": "tran", "simulator": "mock"},
    )


def test_task_fingerprint_tracks_metadata_but_ignores_timeout():
    first = _task(metadata={"run": 1}, timeout=1)
    second = _task(metadata={"run": 1}, timeout=2)
    changed_metadata = _task(metadata={"run": 2}, timeout=1)
    changed_input = _task(param_overrides={"rload": "2k"})

    assert task_fingerprint(first).key == task_fingerprint(second).key
    assert task_fingerprint(first).key != task_fingerprint(changed_metadata).key
    assert task_fingerprint(first).key != task_fingerprint(changed_input).key


def test_task_fingerprint_tracks_backend_options_but_ignores_artifact_destination(tmp_path):
    first = _task(backend_options={"rawfile_format": "ascii"}, artifacts=tmp_path / "first")
    same = _task(backend_options={"rawfile_format": "ascii"}, artifacts=tmp_path / "second")
    changed_backend_options = _task(backend_options={"rawfile_format": "binary"}, artifacts=tmp_path / "first")

    assert task_fingerprint(first).key == task_fingerprint(same).key
    assert task_fingerprint(first).key != task_fingerprint(changed_backend_options).key


def test_task_fingerprint_tracks_recursive_include_content(tmp_path):
    nested = tmp_path / "nested.inc"
    root = tmp_path / "root.inc"
    nested.write_text(".model d D\n")
    root.write_text(f'.include "{nested.name}"\n')
    circuit = _circuit()
    circuit.include(root)

    first = task_fingerprint(_task(circuit))
    nested.write_text(".model d D (is=1e-12)\n")
    second = task_fingerprint(_task(circuit))

    assert first.key != second.key
    assert [artifact.path for artifact in first.artifacts] == [str(root), nested.name]
    assert all(artifact.exists for artifact in first.artifacts)


def test_simulation_cache_round_trips_results(tmp_path):
    cache = SimulationCache(tmp_path / "cache")
    task = _task()
    result = _result()

    entry = cache.store(task, result)
    loaded = cache.load(task)

    assert entry.is_dir()
    assert loaded is not None
    assert loaded.status == "ok"
    assert loaded.metadata["analysis"] == "tran"
    np.testing.assert_allclose(loaded.waveforms["out"], np.array([0.0, 1.0]))
    assert loaded.sweep_var is not None
    np.testing.assert_allclose(loaded.sweep_var, np.array([0.0, 1e-9]))


def test_simulation_cache_run_uses_cache_hit(tmp_path):
    cache = SimulationCache(tmp_path / "cache")
    task = _task()
    calls = 0

    def runner(received: SimTask) -> SimResult:
        nonlocal calls
        calls += 1
        assert received is task
        return _result()

    first = cache.run(task, runner)
    second = cache.run(task, runner)

    assert calls == 1
    assert first.status == "ok"
    assert second.status == "ok"
    np.testing.assert_allclose(second.waveforms["out"], np.array([0.0, 1.0]))


def test_simulation_cache_does_not_store_failed_results_by_default(tmp_path):
    cache = SimulationCache(tmp_path / "cache")
    task = _task()
    calls = 0

    def runner(_: SimTask) -> SimResult:
        nonlocal calls
        calls += 1
        return SimResult(status="failed", waveforms={}, sweep_var=None, corner=None, error_message="failed")

    assert cache.run(task, runner).status == "failed"
    assert cache.run(task, runner).status == "failed"
    assert calls == 2


def test_simulation_cache_rejects_external_cache_keys(tmp_path):
    cache = SimulationCache(tmp_path / "cache")

    with pytest.raises(ValueError, match="sha256"):
        cache.load_key("../escape")
