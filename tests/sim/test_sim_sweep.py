import numpy as np
import pytest
from monata.corner import OperatingCorner
from monata.sim.sweep import ParameterSweep, SweepResults
from monata.sim.results import SimResult
from monata.sim.analysis_spec import ACSpec


class TestParameterSweep:
    def test_single_param_tasks(self):
        spec = ACSpec(start=1, stop=1e9, points=50)
        sweep = ParameterSweep(circuit=None, analysis_spec=spec)
        sweep.sweep("M1.W", np.linspace(1e-6, 10e-6, 5))
        tasks = sweep.tasks()
        assert len(tasks) == 5
        assert tasks[0].param_overrides == {"M1.W": pytest.approx(1e-6)}
        assert tasks[4].param_overrides == {"M1.W": pytest.approx(10e-6)}

    def test_single_param_with_corner(self):
        spec = ACSpec(start=1, stop=1e9, points=50)
        corner = OperatingCorner("hot", 125, voltages={"vdd": 0.9})
        sweep = ParameterSweep(circuit=None, analysis_spec=spec, corner=corner)
        sweep.sweep("R1.R", [1e3, 2e3, 5e3])
        tasks = sweep.tasks()
        assert len(tasks) == 3
        for t in tasks:
            assert t.corner is corner

    def test_2d_sweep_tasks(self):
        spec = ACSpec(start=1, stop=1e9, points=50)
        sweep = ParameterSweep(circuit=None, analysis_spec=spec)
        sweep.sweep("Vgs", [0.3, 0.4, 0.5], param2="Vds", values2=np.array([0.1, 0.3, 0.6]))
        tasks = sweep.tasks()
        assert len(tasks) == 9  # 3 x 3
        assert tasks[0].param_overrides == {"Vgs": 0.3, "Vds": 0.1}

    def test_2d_sweep_requires_paired_second_axis(self):
        spec = ACSpec(start=1, stop=1e9, points=50)
        sweep = ParameterSweep(circuit=None, analysis_spec=spec)

        with pytest.raises(ValueError, match="param2 and values2"):
            sweep.sweep("Vgs", [0.3, 0.4], param2="Vds")
        with pytest.raises(ValueError, match="param2 and values2"):
            sweep.sweep("Vgs", [0.3, 0.4], values2=[0.1, 0.6])

    def test_2d_sweep_results_preserve_second_axis(self):
        spec = ACSpec(start=1, stop=1e9, points=50)
        sweep = ParameterSweep(circuit=None, analysis_spec=spec)
        sweep.sweep("Vgs", [0.3, 0.4], param2="Vds", values2=[0.1, 0.6])

        class Executor:
            def map(self, tasks):
                class Future:
                    def __init__(self, task):
                        self.task = task

                    def result(self):
                        return SimResult(
                            status="ok",
                            waveforms={"out": np.array([self.task.param_overrides["Vgs"] + self.task.param_overrides["Vds"]])},
                            sweep_var=np.array([0.0]),
                            corner=None,
                            metadata=self.task.metadata,
                        )

                return [Future(task) for task in tasks]

        results = sweep.run(Executor())

        assert results.values2().tolist() == [0.1, 0.6]
        assert results.result_at(0.4, 0.6).waveforms["out"][0] == pytest.approx(1.0)
        np.testing.assert_allclose(
            results.extract_grid(lambda r: r.waveforms["out"][0]),
            [[0.4, 0.9], [0.5, 1.0]],
        )
        with pytest.raises(ValueError, match="value2"):
            results.result_at(0.4)

    def test_unconfigured_sweep_fails_explicitly(self):
        sweep = ParameterSweep(circuit=None, analysis_spec=ACSpec(start=1, stop=1e9, points=50))

        with pytest.raises(ValueError, match="must be configured"):
            sweep.tasks()

    def test_reconfigured_sweep_clears_previous_second_axis(self):
        spec = ACSpec(start=1, stop=1e9, points=50)
        sweep = ParameterSweep(circuit=None, analysis_spec=spec)
        sweep.sweep("Vgs", [0.3, 0.4], param2="Vds", values2=[0.1, 0.6])
        assert len(sweep.tasks()) == 4

        sweep.sweep("gain", [1, 2, 3])
        tasks = sweep.tasks()

        assert len(tasks) == 3
        assert tasks[0].param_overrides == {"gain": 1}
        assert tasks[-1].param_overrides == {"gain": 3}

    def test_tasks_preserve_ingress_fields_and_metadata(self):
        spec = ACSpec(start=1, stop=1e9, points=50)
        sweep = ParameterSweep(
            circuit=None,
            analysis_spec=spec,
            output_names=["out"],
            osdi_paths=["model.osdi"],
            metadata={"run": "sweep"},
            simulator="custom-backend",
            timeout=None,
            backend_options={"rawfile_format": "binary"},
            artifacts="artifacts",
        )
        sweep.sweep("gain", [1, 2])

        tasks = sweep.tasks()

        assert tasks[0].output_names == ("out",)
        assert [str(path) for path in tasks[0].osdi_paths] == ["model.osdi"]
        assert tasks[0].metadata["run"] == "sweep"
        assert tasks[0].metadata["sweep_overrides"] == {"gain": 1}
        assert tasks[0].simulator == "custom-backend"
        assert tasks[0].timeout is None
        assert tasks[0].backend_options == {"rawfile_format": "binary"}
        assert str(tasks[0].artifacts.directory) == "artifacts"


class TestSweepResults:
    def test_values(self):
        results = []
        sweep_vals = [1e-6, 2e-6, 5e-6]
        for v in sweep_vals:
            results.append(SimResult(
                status="ok",
                waveforms={"out": np.ones(10) * v},
                sweep_var=np.zeros(10),
                corner=None,
                metadata={"param_value": v},
            ))
        sr = SweepResults(results, param_name="M1.W", param_values=np.array(sweep_vals))
        assert len(sr.values()) == 3
        assert sr.values()[0] == 1e-6

    def test_result_at(self):
        sweep_vals = [1e-6, 2e-6, 5e-6]
        results = []
        for v in sweep_vals:
            results.append(SimResult(
                status="ok",
                waveforms={"out": np.ones(10) * v * 1e6},
                sweep_var=np.zeros(10),
                corner=None,
                metadata={},
            ))
        sr = SweepResults(results, param_name="W", param_values=np.array(sweep_vals))
        r = sr.result_at(2e-6)
        assert r.waveforms["out"][0] == pytest.approx(2.0)

    def test_result_at_requires_exact_value_by_default(self):
        sweep_vals = [1e-6, 2e-6, 5e-6]
        results = [
            SimResult(
                status="ok",
                waveforms={"out": np.array([value])},
                sweep_var=np.zeros(1),
                corner=None,
                metadata={},
            )
            for value in sweep_vals
        ]
        sr = SweepResults(results, param_name="W", param_values=np.array(sweep_vals))

        with pytest.raises(KeyError, match="sweep value not found"):
            sr.result_at(2.1e-6)

        r = sr.result_at(2.1e-6, nearest=True, tolerance=0.2e-6)
        assert r.waveforms["out"][0] == pytest.approx(2e-6)
        with pytest.raises(KeyError, match="outside tolerance"):
            sr.result_at(2.4e-6, nearest=True, tolerance=0.2e-6)
        with pytest.raises(ValueError, match="non-negative"):
            sr.result_at(2.1e-6, nearest=True, tolerance=-1e-9)

    def test_result_at_nearest_requires_numeric_axes(self):
        results = [
            SimResult(
                status="ok",
                waveforms={"out": np.array([1.0])},
                sweep_var=np.zeros(1),
                corner=None,
                metadata={},
            )
        ]
        sr = SweepResults(results, param_name="mode", param_values=np.array(["fast"]))

        with pytest.raises(TypeError, match="numeric sweep values"):
            sr.result_at("slow", nearest=True)

    def test_extract(self):
        sweep_vals = [1.0, 2.0, 3.0]
        results = []
        for v in sweep_vals:
            results.append(SimResult(
                status="ok",
                waveforms={"out": np.ones(10) * v},
                sweep_var=np.zeros(10),
                corner=None,
                metadata={},
            ))
        sr = SweepResults(results, param_name="X", param_values=np.array(sweep_vals))
        extracted = sr.extract(lambda r: r.waveforms["out"][0])
        np.testing.assert_array_equal(extracted, [1.0, 2.0, 3.0])

    def test_extract_grid_preserves_1d_axis(self):
        sweep_vals = [1.0, 2.0, 3.0]
        results = []
        for v in sweep_vals:
            results.append(SimResult(
                status="ok",
                waveforms={"out": np.array([v, v + 0.5])},
                sweep_var=np.zeros(2),
                corner=None,
                metadata={},
            ))
        sr = SweepResults(results, param_name="X", param_values=np.array(sweep_vals))

        scalar_grid = sr.extract_grid(lambda r: r.waveforms["out"][0])
        vector_grid = sr.extract_grid(lambda r: r.waveforms["out"])

        np.testing.assert_array_equal(scalar_grid, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(vector_grid, [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])

    def test_extract_grid_preserves_2d_axis_order(self):
        vgs_values = np.array([0.3, 0.4])
        vds_values = np.array([0.1, 0.6, 1.0])
        results = []
        for vgs in vgs_values:
            for vds in vds_values:
                results.append(SimResult(
                    status="ok",
                    waveforms={"out": np.array([vgs + vds])},
                    sweep_var=np.zeros(1),
                    corner=None,
                    metadata={},
                ))
        sr = SweepResults(
            results,
            param_name="Vgs",
            param_values=vgs_values,
            param2_name="Vds",
            param2_values=vds_values,
        )

        grid = sr.extract_grid(lambda r: r.waveforms["out"][0])

        np.testing.assert_allclose(
            grid,
            [
                [0.4, 0.9, 1.3],
                [0.5, 1.0, 1.4],
            ],
        )

    def test_to_arrays_includes_1d_axis_and_status(self):
        sweep_vals = np.array([1.0, 2.0, 3.0])
        results = [
            SimResult(
                status="ok",
                waveforms={"out": np.array([value])},
                sweep_var=np.zeros(1),
                corner=None,
                metadata={},
            )
            for value in sweep_vals
        ]
        sr = SweepResults(results, param_name="gain", param_values=sweep_vals)

        arrays = sr.to_arrays()

        assert list(arrays) == ["gain", "status"]
        np.testing.assert_array_equal(arrays["gain"], [1.0, 2.0, 3.0])
        assert arrays["status"].tolist() == ["ok", "ok", "ok"]

    def test_to_arrays_includes_2d_axes_and_metrics(self):
        vgs_values = np.array([0.3, 0.4])
        vds_values = np.array([0.1, 0.6])
        statuses = ["ok", "failed", "ok", "ok"]
        metric_values = [0.4, 0.9, 0.5, 1.0]
        results = [
            SimResult(
                status=status,
                waveforms={"out": np.array([value])},
                sweep_var=np.zeros(1),
                corner=None,
                metadata={},
            )
            for status, value in zip(statuses, metric_values, strict=True)
        ]
        sr = SweepResults(
            results,
            param_name="Vgs",
            param_values=vgs_values,
            param2_name="Vds",
            param2_values=vds_values,
        )

        arrays = sr.to_arrays({"vout": lambda result: result.waveforms["out"][0]})

        assert list(arrays) == ["Vgs", "Vds", "status", "vout"]
        np.testing.assert_allclose(arrays["Vgs"], [0.3, 0.3, 0.4, 0.4])
        np.testing.assert_allclose(arrays["Vds"], [0.1, 0.6, 0.1, 0.6])
        assert arrays["status"].tolist() == statuses
        np.testing.assert_allclose(arrays["vout"], [0.4, np.nan, 0.5, 1.0], equal_nan=True)

    def test_to_arrays_rejects_metric_name_conflicts(self):
        sr = SweepResults(
            [
                SimResult(
                    status="ok",
                    waveforms={"out": np.array([1.0])},
                    sweep_var=np.zeros(1),
                    corner=None,
                    metadata={},
                )
            ],
            param_name="gain",
            param_values=np.array([1.0]),
        )

        with pytest.raises(ValueError, match="sweep metadata"):
            sr.to_arrays({"gain": lambda result: result.waveforms["out"][0]})
        with pytest.raises(ValueError, match="sweep metadata"):
            sr.to_arrays({"status": lambda result: result.status})
