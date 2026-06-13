import numpy as np
import pytest
from monata.sim.montecarlo import MonteCarlo, MonteCarloResults
from monata.sim.results import SimResult
from monata.sim.analysis_spec import TranSpec


class TestMonteCarlo:
    def test_add_variation(self):
        spec = TranSpec(stop=1e-6)
        mc = MonteCarlo(circuit=None, analysis_spec=spec, n_samples=100)
        mc.add_variation("vth0", distribution="gaussian", sigma=0.05, nominal=0.4)
        assert len(mc._variations) == 1
        assert mc._variations[0].param == "vth0"
        assert mc._variations[0].sigma == 0.05

    def test_add_mismatch(self):
        spec = TranSpec(stop=1e-6)
        mc = MonteCarlo(circuit=None, analysis_spec=spec, n_samples=200)
        mc.add_mismatch(("M1", "M2"), "vth0", sigma=0.01)
        assert mc._variations[0].kind == "mismatch"
        assert mc._variations[0].pair == ("M1", "M2")

    def test_n_samples(self):
        spec = TranSpec(stop=1e-6)
        mc = MonteCarlo(circuit=None, analysis_spec=spec, n_samples=500)
        assert mc.n_samples == 500

    def test_seeded_sampling_is_reproducible(self):
        class CapturingExecutor:
            def __init__(self):
                self.tasks = []

            def map(self, tasks):  # pyright: ignore[reportIncompatibleMethodOverride]
                self.tasks.extend(tasks)
                class Future:
                    def __init__(self, task):
                        self.task = task

                    def result(self):
                        return SimResult("ok", {}, None, None, metadata=self.task.metadata)

                return [Future(task) for task in tasks]

        spec = TranSpec(stop=1e-6)
        first_executor = CapturingExecutor()
        second_executor = CapturingExecutor()
        first = MonteCarlo(
            circuit=None,
            analysis_spec=spec,
            n_samples=3,
            output_names=["out"],
            osdi_paths=["model.osdi"],
            metadata={"run": "mc"},
            simulator="custom-backend",
            timeout=None,
            backend_options={"rawfile_format": "binary"},
            artifacts="artifacts",
            seed=42,
        )
        second = MonteCarlo(circuit=None, analysis_spec=spec, n_samples=3, seed=42)
        first.add_variation("r", sigma=0.1, nominal=1.0)
        second.add_variation("r", sigma=0.1, nominal=1.0)

        first.run(first_executor)
        second.run(second_executor)

        assert [task.param_overrides for task in first_executor.tasks] == [
            task.param_overrides for task in second_executor.tasks
        ]
        assert first_executor.tasks[0].output_names == ("out",)
        assert [str(path) for path in first_executor.tasks[0].osdi_paths] == ["model.osdi"]
        assert first_executor.tasks[0].metadata["run"] == "mc"
        assert first_executor.tasks[0].metadata["sample_index"] == 0
        assert first_executor.tasks[0].metadata["seed"] == 42
        assert first_executor.tasks[0].metadata["sampled_overrides"] == first_executor.tasks[0].param_overrides
        assert first_executor.tasks[0].metadata["monte_carlo_mode"] == "task-expanded"
        assert first_executor.tasks[0].simulator == "custom-backend"
        assert first_executor.tasks[0].timeout is None
        assert first_executor.tasks[0].backend_options == {"rawfile_format": "binary"}
        assert str(first_executor.tasks[0].artifacts.directory) == "artifacts"

    def test_relative_variations_sample_against_nominal_value(self):
        class CapturingExecutor:
            def __init__(self):
                self.tasks = []

            def map(self, tasks):  # type: ignore[override]
                self.tasks.extend(tasks)

                class Future:
                    def __init__(self, task):
                        self.task = task

                    def result(self):
                        return SimResult("ok", {}, None, None, metadata=self.task.metadata)

                return [Future(task) for task in tasks]

        executor = CapturingExecutor()
        mc = MonteCarlo(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=1, seed=9)
        mc.add_process_variation("cap", distribution="relative_gaussian", sigma=0.1, nominal=2.0)
        mc.add_global_variation("gain", distribution="relative-uniform", sigma=0.2, nominal=10.0)
        expected_rng = np.random.default_rng(9)
        expected_cap = 2.0 * (1.0 + expected_rng.normal(0.0, 0.1))
        expected_gain = 10.0 * (1.0 + expected_rng.uniform(-0.2, 0.2))

        mc.run(executor)
        task = executor.tasks[0]

        assert task.param_overrides["cap"] == pytest.approx(expected_cap)
        assert task.param_overrides["gain"] == pytest.approx(expected_gain)
        assert task.metadata["sampled_variations"][0] == {
            "kind": "process",
            "param": "cap",
            "distribution": "relative_gaussian",
            "nominal": 2.0,
            "sigma": 0.1,
            "value": pytest.approx(expected_cap),
        }
        assert task.metadata["sampled_variations"][1]["distribution"] == "relative_uniform"
        assert task.metadata["sampled_variations"][1]["nominal"] == 10.0

    def test_relative_variation_requires_nominal_value(self):
        class CapturingExecutor:
            def map(self, tasks):  # type: ignore[override]
                return []

        mc = MonteCarlo(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=1, seed=9)
        mc.add_variation("cap", distribution="relative_gaussian", sigma=0.1)

        with pytest.raises(ValueError, match="relative_gaussian variation requires nominal"):
            mc.run(CapturingExecutor())

    def test_mismatch_sampling_creates_pair_overrides_and_metadata(self):
        class CapturingExecutor:
            def __init__(self):
                self.tasks = []

            def map(self, tasks):  # type: ignore[override]
                self.tasks.extend(tasks)

                class Future:
                    def __init__(self, task):
                        self.task = task

                    def result(self):
                        return SimResult("ok", {}, None, None, metadata=self.task.metadata)

                return [Future(task) for task in tasks]

        executor = CapturingExecutor()
        mc = MonteCarlo(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=1, seed=7)
        mc.add_mismatch(("M1", "M2"), "w", sigma=0.1, nominal=1.0)

        mc.run(executor)
        task = executor.tasks[0]

        assert set(task.param_overrides) == {"M1.w", "M2.w"}
        assert task.param_overrides["M1.w"] > 0
        assert task.param_overrides["M2.w"] > 0
        assert task.param_overrides["M1.w"] + task.param_overrides["M2.w"] == pytest.approx(2.0)
        assert task.metadata["sampled_variations"][0]["kind"] == "mismatch"
        assert task.metadata["sampled_variations"][0]["pair"] == ["M1", "M2"]
        assert task.metadata["sampled_variations"][0]["nominal"] == 1.0

    def test_mismatch_sampling_requires_nominal_value(self):
        class CapturingExecutor:
            def map(self, tasks):  # type: ignore[override]
                return []

        mc = MonteCarlo(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=1, seed=7)
        mc.add_mismatch(("M1", "M2"), "w", sigma=0.1)

        with pytest.raises(ValueError, match="requires nominal"):
            mc.run(CapturingExecutor())

    def test_run_submits_samples_as_batch(self):
        class CapturingExecutor:
            def __init__(self):
                self.map_call_count = 0

            def map(self, tasks):
                self.map_call_count += 1

                class Future:
                    def result(self):
                        return SimResult("ok", {}, None, None)

                return [Future() for _ in tasks]

        executor = CapturingExecutor()
        mc = MonteCarlo(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=4)

        results = mc.run(executor)

        assert executor.map_call_count == 1
        assert len(results.samples()) == 4

    def test_native_mode_uses_executor_capability(self):
        class NativeExecutor:
            def __init__(self):
                self.received = None

            def run_monte_carlo_native(self, monte_carlo):
                self.received = monte_carlo
                return MonteCarloResults([
                    SimResult("ok", {}, None, None, metadata={"monte_carlo_mode": "native"})
                ])

        executor = NativeExecutor()
        mc = MonteCarlo(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=4, mode="native")

        results = mc.run(executor)

        assert executor.received is mc
        assert len(results.samples()) == 1
        assert results.samples()[0].metadata["monte_carlo_mode"] == "native"

    def test_native_mode_requires_capable_executor(self):
        class TaskOnlyExecutor:
            def map(self, tasks):
                return []

        mc = MonteCarlo(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=1, mode="native")

        with pytest.raises(ValueError, match="native Monte Carlo mode is not available"):
            mc.run(TaskOnlyExecutor())

    def test_local_executor_provides_native_monte_carlo_capability(self):
        from monata.sim.executor import LocalExecutor

        executor = LocalExecutor()
        captured_tasks = []

        def capture_map(tasks):
            captured_tasks.extend(tasks)

            class Future:
                def __init__(self, task):
                    self.task = task

                def result(self):
                    return SimResult("ok", {}, None, None, metadata=self.task.metadata)

            return [Future(task) for task in tasks]

        executor.map = capture_map  # pyright: ignore[reportAttributeAccessIssue]
        mc = MonteCarlo(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=2, mode="native")
        mc.add_global_variation("gain", sigma=0.1, nominal=1.0)

        results = mc.run(executor)

        assert len(results.samples()) == 2
        assert all(task.metadata["monte_carlo_mode"] == "native" for task in captured_tasks)

    def test_local_executor_native_monte_carlo_uses_public_expansion_contract(self):
        from monata.sim.executor import LocalExecutor

        class PublicExpansionMonteCarlo(MonteCarlo):
            def __init__(self):
                super().__init__(circuit=None, analysis_spec=TranSpec(stop=1e-6), n_samples=1, mode="native")
                self.received = None

            def run_task_expanded(self, executor, *, mode: str):
                self.received = (executor, mode)
                return MonteCarloResults([
                    SimResult("ok", {}, None, None, metadata={"monte_carlo_mode": mode})
                ])

            def _run_task_expanded(self, executor, *, mode: str):
                raise AssertionError("executor must not call MonteCarlo private expansion hook")

        executor = LocalExecutor()
        mc = PublicExpansionMonteCarlo()

        results = executor.run_monte_carlo_native(mc)

        assert mc.received == (executor, "native")
        assert results.samples()[0].metadata["monte_carlo_mode"] == "native"


class TestMonteCarloResults:
    def _make_samples(self, n=100):
        rng = np.random.default_rng(42)
        results = []
        for _ in range(n):
            val = rng.normal(1.0, 0.1)
            results.append(SimResult(
                status="ok",
                waveforms={"out": np.array([val])},
                sweep_var=np.array([0.0]),
                corner=None,
                metadata={},
            ))
        return MonteCarloResults(results)

    def test_samples(self):
        mcr = self._make_samples(50)
        assert len(mcr.samples()) == 50

    def test_extract(self):
        mcr = self._make_samples(100)
        values = mcr.extract(lambda r: r.waveforms["out"][0])
        assert len(values) == 100
        assert np.mean(values) == pytest.approx(1.0, abs=0.05)

    def test_histogram(self):
        mcr = self._make_samples(200)
        bin_edges, counts = mcr.histogram(lambda r: r.waveforms["out"][0], bins=20)
        assert len(bin_edges) == 21
        assert counts.sum() == 200

    def test_histogram_requires_passing_samples(self):
        mcr = MonteCarloResults([])
        with pytest.raises(ValueError, match="No passing samples"):
            mcr.histogram(lambda r: r.waveforms["out"][0])

    def test_histogram_rejects_nonfinite_metric_values(self):
        mcr = MonteCarloResults([
            SimResult("ok", {"out": np.array([np.nan])}, np.array([0.0]), None),
        ])
        with pytest.raises(ValueError, match="finite"):
            mcr.histogram(lambda r: r.waveforms["out"][0])

    def test_sigma_yield(self):
        mcr = self._make_samples(1000)
        y = mcr.sigma_yield(lambda r: r.waveforms["out"][0], spec_min=0.7, spec_max=1.3)
        assert y > 0.95

    def test_sigma_yield_requires_passing_samples(self):
        mcr = MonteCarloResults([])
        with pytest.raises(ValueError, match="No passing samples"):
            mcr.sigma_yield(lambda r: r.waveforms["out"][0])

    def test_sigma_yield_rejects_invalid_spec_range(self):
        mcr = self._make_samples(10)
        with pytest.raises(ValueError, match="less than or equal"):
            mcr.sigma_yield(lambda r: r.waveforms["out"][0], spec_min=2.0, spec_max=1.0)

    def test_to_arrays(self):
        mcr = self._make_samples(10)
        columns = mcr.to_arrays({"val": lambda r: r.waveforms["out"][0]})
        assert set(columns) == {"val"}
        assert isinstance(columns["val"], np.ndarray)
        assert columns["val"].shape == (10,)

    def test_to_arrays_fills_failed_metrics_with_nan(self):
        mcr = MonteCarloResults([
            SimResult("ok", {"out": np.array([1.0])}, np.array([0.0]), None),
            SimResult("failed", {}, None, None),
        ])
        columns = mcr.to_arrays({"val": lambda r: r.waveforms["out"][0]})
        assert columns["val"][0] == pytest.approx(1.0)
        assert np.isnan(columns["val"][1])
