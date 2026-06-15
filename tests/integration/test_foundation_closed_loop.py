import numpy as np
import pytest

from monata.measure.spec import Spec
from monata.models import ModelRegistry
from monata.netlist import Circuit
from monata.schematic import SchematicBuilder
from monata.sim.core import DCSpec, LocalExecutor, SimResult, SimTask
from monata.workspace.project import Project

pytestmark = [pytest.mark.integration, pytest.mark.native]


def test_foundation_project_to_ngspice_result_closed_loop(tmp_path, require_ngspice):
    project = Project.create(tmp_path / "closed_loop")
    lib = project.create_library("analog", tech_model_paths=[])
    cell = lib.create_cell("rc_probe", description="foundation closed-loop cell")
    (
        SchematicBuilder("rc_probe")
        .pin("inp", direction="input")
        .pin("out", direction="output")
        .pin("gnd", direction="ground")
        .primitive("load", "resistor", connections={"n1": "inp", "n2": "out"}, value="1k")
        .primitive("hold", "capacitor", connections={"n1": "out", "n2": "gnd"}, value="1n")
        .write(cell.path / "schematic.monata.json")
    )
    cell.create_view("schematic")

    symbol_path = cell.generate_symbol()
    netlist_path = cell.generate_netlist()

    assert symbol_path.exists()
    assert netlist_path.exists()
    assert ".subckt rc_probe inp out gnd" in netlist_path.read_text()
    assert "rc_probe" in Project(project.path).get_library("analog")

    model_path = tmp_path / "models" / "placeholder.osdi"
    model_path.parent.mkdir()
    model_path.write_text("placeholder")
    models = ModelRegistry(auto_discover=False)
    models.register("mos", model_path, module_name="placeholder")
    assert models.osdi_paths("mos") == [str(model_path)]

    subcircuit = cell["schematic"].to_circuit()
    circuit = Circuit("foundation dc sanity")
    circuit.subckt(subcircuit)
    circuit.voltage("1", "in", "0", "0")
    circuit.instance("probe", ("in", "out", "0"), subcircuit)
    circuit.resistor("sense", "out", "0", "1g")
    task = SimTask(
        circuit=circuit,
        analysis_spec=DCSpec(source="V1", start=0, stop=1, step=0.5),
        output_names=["out"],
        metadata={"project": project.name, "library": lib.name, "cell": cell.name},
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok", result.error_message
    assert result.sweep_var is not None
    np.testing.assert_allclose(result.waveforms["out"], result.sweep_var, rtol=0, atol=2e-6)
    spec = Spec("final_vout", lambda sim: float(sim.waveforms["out"][-1]), min=0.99, max=1.01)
    spec_result = spec.evaluate(result)
    assert spec_result.passed is True

    experiment = project.new_experiment("dc_validation")
    experiment.save_results("nominal", result)
    loaded = experiment.load_results("nominal")
    assert isinstance(loaded, SimResult)
    assert loaded.status == "ok"
    np.testing.assert_allclose(loaded.waveforms["out"], result.waveforms["out"])
