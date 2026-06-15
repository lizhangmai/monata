import numpy as np
import pytest

from monata.measure import SpecTable
from monata.measure.summary import tran_summary
from monata.netlist import Circuit
from monata.sim.core import LocalExecutor, SimTask, TranSpec
from monata.sim.results import SimResult
from monata.workspace import Experiment

pytestmark = [pytest.mark.integration, pytest.mark.native]


def test_p4_measure_summary_specs_and_bundle_round_trip(tmp_path, require_ngspice):
    circuit = Circuit("p4 workflow")
    circuit.voltage("1", "in", "0", "pulse(0 1 0 1n 1n 5n 10n)")
    circuit.resistor("1", "in", "out", "1k")
    circuit.capacitor("1", "out", "0", "1n")
    circuit.measure("tran", "vout_2n", "FIND v(out) AT=2n")
    task = SimTask(
        circuit=circuit,
        analysis_spec=TranSpec(step=1e-9, stop=5e-9),
        output_names=["out"],
    )

    result = LocalExecutor(max_workers=1).submit(task).result()
    assert result.status == "ok", result.error_message
    result = result.with_summary("tran", tran_summary(result, "out"))

    specs = SpecTable()
    specs.add_measure("vout_2n", min=0.0, max=1.0, unit="V")
    specs.add_summary("out_swing", "tran", "peak_to_peak", min=0.0, unit="V")
    specs.add("final_out", lambda sim: float(sim.waveforms["out"][-1]), min=0.0, max=1.0, unit="V")
    rows = specs.evaluate_rows([result])

    assert {row["name"] for row in rows} == {"vout_2n", "out_swing", "final_out"}
    assert all(row["passed"] is True for row in rows)
    assert result.measures.value("vout_2n") >= 0.0
    assert result.summaries["tran"].value("peak_to_peak") > 0.0

    experiment = Experiment(tmp_path / "p4")
    experiment.save_results("tran_nom", result, specs=rows)
    loaded = experiment.load_results("tran_nom")
    bundle = experiment.load_result_bundle("tran_nom")

    assert isinstance(loaded, SimResult)
    np.testing.assert_allclose(loaded.waveforms["out"], result.waveforms["out"])
    assert loaded.measures.value("vout_2n") == result.measures.value("vout_2n")
    assert loaded.summaries["tran"].value("peak_to_peak") == result.summaries["tran"].value("peak_to_peak")
    assert bundle.measures.value("vout_2n") == result.measures.value("vout_2n")
    assert len(bundle.specs) == 3
