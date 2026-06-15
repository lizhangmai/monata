import json

import numpy as np
import pytest

from monata.corner import OperatingCorner
from monata.measure.result import MeasureResult
from monata.measure.summary import AnalysisSummary
from monata.sim.results import AnalysisResult, Waveform
from monata.workspace.experiment import Experiment, ExperimentResultBundle
from monata.sim.results import SimResult
from monata.sim.corner import CornerResults


def _make_sim_result():
    corner = OperatingCorner("nom_27C", 27, voltages={"vdd": 1.0})
    return SimResult(
        status="ok",
        waveforms={"out": np.linspace(0, 1, 100), "in": np.ones(100)},
        sweep_var=np.linspace(0, 1e-6, 100),
        corner=corner,
        metadata={"simulator": "ngspice", "elapsed_time": 0.3},
    )


def _make_corner_results():
    c1 = OperatingCorner("nom", 27, voltages={"vdd": 1.0})
    c2 = OperatingCorner("hot", 125, voltages={"vdd": 0.9})
    return CornerResults([
        SimResult(status="ok", waveforms={"out": np.ones(50)}, sweep_var=np.zeros(50), corner=c1, metadata={}),
        SimResult(status="ok", waveforms={"out": np.ones(50) * 0.8}, sweep_var=np.zeros(50), corner=c2, metadata={}),
    ])


class TestExperiment:
    def test_create(self, tmp_path):
        exp = Experiment(tmp_path / "exp1", description="test experiment")
        assert exp.name == "exp1"
        assert (tmp_path / "exp1" / "experiment.toml").exists()

    def test_save_and_load_sim_result(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        result = _make_sim_result()
        exp.save_results("tran_nom", result)
        npz_path = tmp_path / "exp1" / "results" / "tran_nom.npz"
        meta = json.loads((tmp_path / "exp1" / "results" / "tran_nom.json").read_text())
        waveform_key = meta["payload"]["waveforms"]["out"]["key"]
        sweep_key = meta["payload"]["sweep_var"]["key"]
        with np.load(npz_path) as arrays:
            assert waveform_key in arrays.files
            assert sweep_key in arrays.files
            np.testing.assert_array_almost_equal(arrays[waveform_key], result.waveforms["out"])
            assert result.sweep_var is not None
            np.testing.assert_array_almost_equal(arrays[sweep_key], result.sweep_var)

        loaded = exp.load_results("tran_nom")
        assert isinstance(loaded, SimResult)
        assert isinstance(loaded.corner, OperatingCorner)
        assert "payload" in meta
        assert meta["payload"]["waveforms"]["out"]["storage"] == "npz"
        assert meta["corner"]["schema"] == "monata.operating_corner.v1"
        assert meta["corner"]["voltages"] == {"vdd": 1.0}
        assert loaded.status == "ok"
        assert "out" in loaded.waveforms
        np.testing.assert_array_almost_equal(loaded.waveforms["out"], result.waveforms["out"])
        assert loaded.sweep_var is not None
        assert result.sweep_var is not None
        np.testing.assert_array_almost_equal(loaded.sweep_var, result.sweep_var)

    def test_save_and_load_sim_result_preserves_colliding_array_storage_names(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        result = SimResult(
            status="ok",
            waveforms={
                "v/out": np.array([1.0, 2.0]),
                "v__out": np.array([3.0, 4.0]),
            },
            sweep_var=np.array([0.0, 1e-9]),
            corner=None,
        )

        exp.save_results("colliding_names", result)
        loaded = exp.load_results("colliding_names")

        assert isinstance(loaded, SimResult)
        np.testing.assert_allclose(loaded.waveforms["v/out"], np.array([1.0, 2.0]))
        np.testing.assert_allclose(loaded.waveforms["v__out"], np.array([3.0, 4.0]))

    def test_save_and_load_failed_sim_result_preserves_error_message(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        result = SimResult(
            status="failed",
            waveforms={},
            sweep_var=None,
            corner=OperatingCorner("nom", 27, voltages={"vdd": 1.0}),
            metadata={"simulator": "ngspice"},
            error_message="no convergence at 12ns",
        )

        exp.save_results("failed", result)
        loaded = exp.load_results("failed")

        assert isinstance(loaded, SimResult)
        assert loaded.status == "failed"
        assert loaded.error_message == "no convergence at 12ns"

    def test_save_and_load_sim_result_preserves_explicit_analysis_result(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        analysis = AnalysisResult(
            analysis="ac",
            waveforms={
                "out": Waveform(
                    "out",
                    np.array([1 + 0j, 0.5 - 0.5j], dtype=np.complex64),
                    unit="V",
                    quantity="voltage",
                    raw_vector_name="v(out)",
                    title="Output voltage",
                ),
                "derived": Waveform(
                    "derived",
                    np.array([3, 4], dtype=np.int16),
                    unit="dB",
                    quantity="gain_db",
                ),
            },
            abscissa=Waveform("frequency", np.array([1.0, 10.0], dtype=np.float32), unit="Hz"),
            metadata={"source": "rawfile"},
            source="rawfile",
        )
        result = SimResult(
            status="ok",
            waveforms={"out": np.array([1 + 0j, 0.5 - 0.5j], dtype=np.complex64)},
            sweep_var=np.array([1.0, 10.0], dtype=np.float32),
            corner=OperatingCorner("nom", 27, voltages={"vdd": 1.0}),
            metadata={"analysis": "ac"},
            analysis_result=analysis,
        )

        exp.save_results("typed", result)
        npz_path = tmp_path / "exp1" / "results" / "typed.npz"
        meta = json.loads((tmp_path / "exp1" / "results" / "typed.json").read_text())
        analysis_payload = meta["payload"]["analysis_result"]
        derived_key = analysis_payload["waveforms"]["derived"]["data"]["key"]
        abscissa_key = analysis_payload["abscissa"]["data"]["key"]
        with np.load(npz_path) as arrays:
            assert derived_key in arrays.files
            assert abscissa_key in arrays.files
            np.testing.assert_array_equal(arrays[derived_key], np.array([3, 4], dtype=np.int16))

        loaded = exp.load_results("typed")

        assert isinstance(loaded, SimResult)
        assert loaded.analysis_result is not None
        assert loaded.analysis_result.analysis == "ac"
        assert loaded.analysis_result.source == "rawfile"
        assert loaded.analysis_result.abscissa is not None
        assert loaded.analysis_result.abscissa.name == "frequency"
        assert loaded.analysis_result.waveform("out").raw_vector_name == "v(out)"
        assert loaded.analysis_result.waveform("out").title == "Output voltage"
        np.testing.assert_array_equal(
            loaded.analysis_result.waveform("derived").data,
            np.array([3, 4], dtype=np.int16),
        )

    @pytest.mark.parametrize("name", ["../outside", "quote\"name", "evil\nname", "tab\tname", "space name"])
    def test_result_names_reject_unsafe_path_segments(self, tmp_path, name):
        exp = Experiment(tmp_path / "exp1")

        with pytest.raises(ValueError, match="single safe path segment"):
            exp.save_results(name, _make_sim_result())
        with pytest.raises(ValueError, match="single safe path segment"):
            exp.load_results(name)
        with pytest.raises(ValueError, match="single safe path segment"):
            exp.load_result_bundle(name)

    def test_save_results_rejects_unrecoverable_generic_payloads(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")

        with pytest.raises(TypeError, match="SimResult or CornerResults"):
            exp.save_results("generic", {"a": 1})

    def test_save_and_load_corner_results(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        cr = _make_corner_results()
        exp.save_results("corners", cr)
        npz_path = tmp_path / "exp1" / "results" / "corners.npz"
        meta = json.loads((tmp_path / "exp1" / "results" / "corners.json").read_text())
        first_waveform_key = meta["results"][0]["payload"]["waveforms"]["out"]["key"]
        second_waveform_key = meta["results"][1]["payload"]["waveforms"]["out"]["key"]
        with np.load(npz_path) as arrays:
            assert first_waveform_key == "r0_waveforms__out"
            assert second_waveform_key == "r1_waveforms__out"
            assert first_waveform_key in arrays.files
            assert second_waveform_key in arrays.files

        loaded = exp.load_results("corners")
        assert isinstance(loaded, CornerResults)
        assert meta["results"][0]["corner"]["schema"] == "monata.operating_corner.v1"
        assert len(loaded) == 2
        assert loaded["nom"].status == "ok"

    def test_load_results_requires_canonical_sim_result_payload(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        result = _make_sim_result()
        exp.save_results("missing_payload", result)
        meta_path = tmp_path / "exp1" / "results" / "missing_payload.json"
        meta = json.loads(meta_path.read_text())
        meta.pop("payload")
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="missing canonical payload"):
            exp.load_results("missing_payload")

    def test_load_corner_results_requires_canonical_item_payload(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        exp.save_results("missing_payload", _make_corner_results())
        meta_path = tmp_path / "exp1" / "results" / "missing_payload.json"
        meta = json.loads(meta_path.read_text())
        meta["results"][0].pop("payload")
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="missing canonical payload"):
            exp.load_results("missing_payload")

    def test_overwrite_protection(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        exp.save_results("data", _make_sim_result())
        with pytest.raises(FileExistsError):
            exp.save_results("data", _make_sim_result())

    def test_overwrite_allowed(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        exp.save_results("data", _make_sim_result())
        exp.save_results("data", _make_sim_result(), overwrite=True)

    def test_note(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        exp.note("GBW achieved 150MHz")
        notes_path = tmp_path / "exp1" / "notes.md"
        assert "GBW achieved 150MHz" in notes_path.read_text()

    def test_summary(self, tmp_path):
        exp = Experiment(tmp_path / "exp1", description="my desc")
        exp.save_results("corners", _make_corner_results())
        s = exp.summary
        assert s["description"] == "my desc"
        assert "corners" in s["results"]

    def test_summary_lists_results_in_stable_order(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        exp.save_results("z_last", _make_sim_result())
        exp.save_results("a_first", _make_sim_result())

        assert exp.summary["results"] == ["a_first", "z_last"]

    def test_summary_ignores_incomplete_result_sidecars(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        exp.save_results("complete", _make_sim_result())
        results_dir = tmp_path / "exp1" / "results"
        np.savez_compressed(results_dir / "orphan_npz.npz", values=np.array([1.0]))
        (results_dir / "orphan_json.json").write_text('{"type": "SimResult"}')

        assert exp.summary["results"] == ["complete"]

    def test_load_results_reports_missing_metadata_sidecar(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        np.savez_compressed(tmp_path / "exp1" / "results" / "orphan_npz.npz", values=np.array([1.0]))

        with pytest.raises(FileNotFoundError, match=r"incomplete.*orphan_npz\.json"):
            exp.load_results("orphan_npz")

    def test_load_result_bundle_reports_missing_array_sidecar(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        (tmp_path / "exp1" / "results" / "orphan_json.json").write_text('{"type": "SimResult"}')

        with pytest.raises(FileNotFoundError, match=r"incomplete.*orphan_json\.npz"):
            exp.load_result_bundle("orphan_json")

    def test_existing_experiment_loads_persisted_description(self, tmp_path):
        Experiment(tmp_path / "exp1", description="persisted desc")

        reopened = Experiment(tmp_path / "exp1")

        assert reopened.summary["description"] == "persisted desc"

    def test_save_load_result_bundle_preserves_measures_summaries_and_specs(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        base = _make_sim_result()
        result = SimResult(
            status=base.status,
            waveforms=base.waveforms,
            sweep_var=base.sweep_var,
            corner=base.corner,
            metadata=base.metadata,
            measures={"tdelay": MeasureResult("tdelay", np.float64(2e-9), unit="s")},
            summaries={
                "tran": AnalysisSummary(
                    "tran",
                    {"rise_time": np.float64(1e-9)},
                    units={"rise_time": "s"},
                    metadata={"samples": np.array([1, 2])},
                )
            },
        )
        specs = [{"name": "tdelay", "corner": "nom_27C", "value": np.float64(2e-9), "passed": True}]

        exp.save_results("tran_nom", result, specs=specs)
        loaded = exp.load_results("tran_nom")
        bundle = exp.load_result_bundle("tran_nom")

        assert isinstance(loaded, SimResult)
        assert isinstance(bundle, ExperimentResultBundle)
        assert loaded.measures.value("tdelay") == 2e-9
        assert loaded.summaries["tran"].value("rise_time") == 1e-9
        assert bundle.measures.value("tdelay") == 2e-9
        assert bundle.summaries["tran"].to_dict()["metadata"]["samples"] == [1, 2]
        assert bundle.specs == [{"name": "tdelay", "corner": "nom_27C", "value": 2e-9, "passed": True}]

    def test_corner_result_bundle_preserves_per_corner_sidecars(self, tmp_path):
        exp = Experiment(tmp_path / "exp1")
        c1 = OperatingCorner("nom", 27, voltages={"vdd": 1.0})
        c2 = OperatingCorner("hot", 125, voltages={"vdd": 0.9})
        cr = CornerResults([
            SimResult(
                status="ok",
                waveforms={"out": np.ones(50)},
                sweep_var=np.zeros(50),
                corner=c1,
                metadata={},
                measures={"vmax": MeasureResult("vmax", 1.0, unit="V")},
            ),
            SimResult(
                status="ok",
                waveforms={"out": np.ones(50) * 0.8},
                sweep_var=np.zeros(50),
                corner=c2,
                metadata={},
                summaries={"tran": AnalysisSummary("tran", {"rise_time": 2e-9})},
            ),
        ])

        exp.save_results("corners", cr, specs={"worst": "hot"})
        loaded = exp.load_results("corners")
        bundle = exp.load_result_bundle("corners")

        assert isinstance(loaded, CornerResults)
        assert loaded["nom"].measures.value("vmax") == 1.0
        assert loaded["hot"].summaries["tran"].value("rise_time") == 2e-9
        assert bundle.measures["nom"]["vmax"]["value"] == 1.0
        assert bundle.summaries["hot"]["tran"]["values"]["rise_time"] == 2e-9
        assert bundle.specs == {"worst": "hot"}
