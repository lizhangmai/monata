"""Small tested workflow examples for Monata users."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from monata.measure.spec import Spec
from monata.models import ModelRegistry
from monata.netlist import Circuit, render_ngspice
from monata.parser import parse_spice_to_circuit
from monata.sim.core import DCSpec, SimResult, SimTask
from monata.workspace.project import Project


def native_rc_dc_task() -> SimTask:
    """Build a native RC circuit and a DC task for the ngspice backend."""

    circuit = Circuit("native rc example")
    circuit.voltage("1", "in", "0", "0")
    circuit.resistor("load", "in", "out", "1k")
    circuit.resistor("sense", "out", "0", "1g")
    return SimTask(
        circuit=circuit,
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["out"],
        metadata={"example": "native-rc-dc"},
    )


def imported_roundtrip_deck() -> str:
    """Parse and render a small license-compatible SPICE deck."""

    circuit = parse_spice_to_circuit(
        """
rc import example
R1 in out 1k
C1 out 0 1n
V1 in 0 DC 1
.op
.end
"""
    )
    return render_ngspice(circuit)


def synthetic_model_registry(root: str | Path) -> ModelRegistry:
    """Create a synthetic model registry without third-party model assets."""

    root_path = Path(root)
    model_path = root_path / "synthetic.osdi"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("synthetic model placeholder")
    registry = ModelRegistry(auto_discover=False)
    registry.register("mos", model_path, module_name="synthetic_mos", level=1, version="example")
    return registry


def experiment_persistence_example(root: str | Path) -> SimResult:
    """Save and reload a synthetic simulation result through Project experiments."""

    project = Project.create(Path(root) / "example_project")
    result = SimResult(
        status="ok",
        waveforms={"out": np.array([0.0, 0.5, 1.0])},
        sweep_var=np.array([0.0, 0.5, 1.0]),
        corner=None,
        metadata={"analysis": "dc", "example": "experiment-persistence"},
    )
    spec = Spec("final_vout", lambda sim: float(sim.waveforms["out"][-1]), min=0.99, max=1.01)
    if not spec.evaluate(result).passed:
        raise RuntimeError("synthetic example result failed its final_vout spec")
    experiment = project.new_experiment("dc_example")
    experiment.save_results("nominal", result)
    loaded = experiment.load_results("nominal")
    if not isinstance(loaded, SimResult):
        raise TypeError("example expected a single SimResult")
    return loaded
