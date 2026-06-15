import numpy as np

from monata import examples
from monata.netlist import render_ngspice
from monata.sim.core import SimResult


def test_native_rc_dc_task_example_is_ngspice_ready():
    task = examples.native_rc_dc_task()

    assert task.simulator == "ngspice-subprocess"
    assert task.output_names == ("out",)
    assert task.metadata["example"] == "native-rc-dc"
    deck = render_ngspice(task.circuit)
    assert "Rload in out 1k" in deck
    assert ".end" in deck


def test_imported_roundtrip_deck_example_renders_normalized_deck():
    rendered = examples.imported_roundtrip_deck()

    assert "R1 in out 1k" in rendered
    assert ".op" in rendered
    assert ".end" in rendered


def test_synthetic_model_registry_example_uses_local_fixture(tmp_path):
    registry = examples.synthetic_model_registry(tmp_path)

    resolved = registry.resolve("mos", level=1, version="example")

    assert resolved is not None
    assert resolved.module_name == "synthetic_mos"
    assert registry.osdi_paths("mos", level=1, version="example") == [str(tmp_path / "synthetic.osdi")]


def test_experiment_persistence_example_round_trips_result(tmp_path):
    loaded = examples.experiment_persistence_example(tmp_path)

    assert isinstance(loaded, SimResult)
    assert loaded.status == "ok"
    assert loaded.metadata["example"] == "experiment-persistence"
    np.testing.assert_allclose(loaded.waveforms["out"], np.array([0.0, 0.5, 1.0]))
