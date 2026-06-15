import json
from typing import Any, cast

import numpy as np
import pytest
from monata.optim.circuit import CircuitOptimizer
from monata.optim.base import OptimResult, Optimizer
from monata.sim.analysis_spec import ACSpec
from monata.sim.results import SimResult
from monata.workspace.experiment import Experiment


class MockExecutor:
    def submit(self, task):
        from concurrent.futures import Future
        f = Future()
        freq = np.logspace(0, 9, 100)
        # Simulate: wider W → higher gain
        w = task.param_overrides.get("M1.W", 5e-6)
        gain_db = 20 * np.log10(w * 1e6 * 10)  # gain scales with W
        mag = np.ones(100) * gain_db
        f.set_result(SimResult(
            status="ok",
            waveforms={"out": mag},
            sweep_var=freq,
            corner=None,
            metadata={"i_supply": w * 100},
        ))
        return f

    def map(self, tasks):
        return [self.submit(t) for t in tasks]


class CapturingExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, task):
        from concurrent.futures import Future
        self.tasks.append(task)
        f = Future()
        f.set_result(SimResult(
            status="ok",
            waveforms={"out": np.ones(4)},
            sweep_var=np.arange(4),
            corner=None,
            metadata=task.metadata,
        ))
        return f

    def map(self, tasks):
        return [self.submit(t) for t in tasks]


class FailingMutationExecutor:
    def submit(self, task):
        from concurrent.futures import Future
        f = Future()
        f.set_result(SimResult(
            status="failed",
            waveforms={},
            sweep_var=None,
            corner=None,
            metadata={"reason": "unsupported_param_overrides"},
            error_message="mutation target not found: M404.w",
        ))
        return f

    def map(self, tasks):
        return [self.submit(t) for t in tasks]


class OneShotOptimizer(Optimizer):
    def __init__(self, params):
        self.params = params

    def optimize(self, variables, objectives, constraints, eval_fn, n_iter):
        result = eval_fn(dict(self.params))
        result.update(self.params)
        return OptimResult(
            pareto_front=[result],
            all_evaluations=[result],
            best=result,
            n_evaluations=1,
        )


class TestCircuitOptimizer:
    def test_variable_registration(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        opt.variable("M1", "W", 1e-6, 10e-6)
        assert len(opt._variables) == 1
        assert opt._variables[0].name == "M1.W"

    def test_parameter_registration(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        opt.parameter("wn", 1e-6, 10e-6)
        assert len(opt._variables) == 1
        assert opt._variables[0].name == "wn"

    def test_parameter_rejects_instance_style_name(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        with pytest.raises(ValueError, match="simple identifier"):
            opt.parameter("M1.W", 1e-6, 10e-6)

    def test_default_native_executor_accepts_instance_style_variable(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        opt.variable("M1", "W", 1e-6, 10e-6)
        opt.maximize("gain", lambda r: 1.0)
        result = opt.run(optimizer=OneShotOptimizer({"M1.W": 2e-6}), n_iter=1)
        assert result.best["M1.W"] == 2e-6

    def test_optimizer_mutation_failures_raise_deterministically(self):
        opt = CircuitOptimizer(
            circuit=None,
            analysis=ACSpec(1, 1e9, 100),
            executor=FailingMutationExecutor(),
        )
        opt.variable("M404", "w", 1e-6, 10e-6)
        opt.maximize("gain", lambda r: 1.0)

        with pytest.raises(ValueError, match="mutation target not found"):
            opt.run(optimizer=OneShotOptimizer({"M404.w": 2e-6}), n_iter=1)

    def test_objective_registration(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        opt.maximize("GBW", lambda r: r.waveforms["out"][0])
        opt.minimize("power", lambda r: r.metadata["i_supply"])
        assert len(opt._objectives) == 2

    def test_constraint_registration(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        opt.constraint("PM", lambda r: 65.0, min=60)
        assert len(opt._constraints) == 1

    def test_metric_registration_rejects_non_callable(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        with pytest.raises(TypeError, match="metric function"):
            opt.maximize("gain", cast(Any, None))

    def test_run_rejects_missing_problem_contract_before_custom_optimizer(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        opt.parameter("wn", 1e-6, 10e-6)

        with pytest.raises(ValueError, match="at least one objective"):
            opt.run(optimizer=OneShotOptimizer({"wn": 2e-6}), n_iter=1)

    def test_run_rejects_invalid_iteration_budget_before_custom_optimizer(self):
        opt = CircuitOptimizer(circuit=None, analysis=ACSpec(1, 1e9, 100))
        opt.parameter("wn", 1e-6, 10e-6)
        opt.maximize("gain", lambda r: 1.0)

        with pytest.raises(ValueError, match="n_iter"):
            opt.run(optimizer=OneShotOptimizer({"wn": 2e-6}), n_iter=0)

    def test_run_returns_optim_result(self):
        opt = CircuitOptimizer(
            circuit=None,
            analysis=ACSpec(1, 1e9, 100),
            executor=MockExecutor(),
        )
        opt.variable("M1", "W", 1e-6, 10e-6)
        opt.maximize("gain", lambda r: r.waveforms["out"][0])
        result = opt.run(n_iter=10)
        assert isinstance(result, OptimResult)
        assert result.n_evaluations > 0

    def test_custom_optimizer(self):
        class DummyOptimizer(Optimizer):
            def optimize(self, variables, objectives, constraints, eval_fn, n_iter):
                # Just evaluate midpoint
                params = {v.name: (v.min + v.max) / 2 for v in variables}
                result = eval_fn(params)
                result.update(params)
                return OptimResult(
                    pareto_front=[result],
                    all_evaluations=[result],
                    best=result,
                    n_evaluations=1,
                )

        opt = CircuitOptimizer(
            circuit=None,
            analysis=ACSpec(1, 1e9, 100),
            executor=MockExecutor(),
        )
        opt.variable("M1", "W", 1e-6, 10e-6)
        opt.minimize("power", lambda r: r.metadata["i_supply"])
        result = opt.run(optimizer=DummyOptimizer(), n_iter=1)
        assert result.n_evaluations == 1

    def test_task_ingress_pass_through_and_metadata_json_safe(self, tmp_path):
        osdi = tmp_path / "model.osdi"
        osdi.write_text("placeholder")
        executor = CapturingExecutor()
        opt = CircuitOptimizer(
            circuit=None,
            analysis=ACSpec(1, 1e9, 100),
            executor=executor,
            output_names=["out"],
            osdi_paths=[osdi],
            metadata={"run": "nom", "numpy_value": np.float64(1.25)},
        )
        opt.parameter("wn", 1e-6, 10e-6)
        opt.maximize("gain", lambda r: r.waveforms["out"][0])

        result = opt.run(
            optimizer=OneShotOptimizer({"wn": np.float64(2e-6)}),
            n_iter=1,
        )

        assert result.n_evaluations == 1
        assert len(executor.tasks) == 1
        task = executor.tasks[0]
        assert task.output_names == ("out",)
        assert task.osdi_paths == (osdi,)
        assert task.param_overrides == {"wn": np.float64(2e-6)}
        assert task.metadata["run"] == "nom"
        assert task.metadata["numpy_value"] == 1.25
        assert task.metadata["monata_optimizer"]["evaluation_index"] == 0
        assert task.metadata["monata_optimizer"]["params"] == {"wn": 2e-6}
        json.dumps(task.metadata)

    def test_optimizer_metadata_survives_experiment_result_round_trip(self, tmp_path):
        executor = CapturingExecutor()
        opt = CircuitOptimizer(
            circuit=None,
            analysis=ACSpec(1, 1e9, 100),
            executor=executor,
            output_names=["out"],
            metadata={"run": "persist"},
        )
        opt.parameter("wn", 1e-6, 10e-6)
        opt.maximize("gain", lambda r: r.waveforms["out"][0])

        opt.run(optimizer=OneShotOptimizer({"wn": np.float64(3e-6)}), n_iter=1)
        task = executor.tasks[0]
        sim_result = SimResult(
            status="ok",
            waveforms={"out": np.ones(3)},
            sweep_var=np.arange(3),
            corner=None,
            metadata=task.metadata,
        )

        exp = Experiment(tmp_path / "exp")
        exp.save_results("optim", sim_result)
        loaded = exp.load_results("optim")
        assert isinstance(loaded, SimResult)
        assert loaded.metadata["run"] == "persist"
        assert loaded.metadata["monata_optimizer"]["params"] == {"wn": 3e-6}

    def test_user_optimizer_metadata_key_is_preserved(self):
        executor = CapturingExecutor()
        opt = CircuitOptimizer(
            circuit=None,
            analysis=ACSpec(1, 1e9, 100),
            executor=executor,
            metadata={"optimizer": "user-owned"},
        )
        opt.parameter("wn", 1e-6, 10e-6)
        opt.maximize("gain", lambda r: r.waveforms["out"][0])

        opt.run(optimizer=OneShotOptimizer({"wn": 4e-6}), n_iter=1)

        task = executor.tasks[0]
        assert task.metadata["optimizer"] == "user-owned"
        assert task.metadata["monata_optimizer"]["params"] == {"wn": 4e-6}

    def test_user_monata_optimizer_metadata_key_is_preserved(self):
        executor = CapturingExecutor()
        opt = CircuitOptimizer(
            circuit=None,
            analysis=ACSpec(1, 1e9, 100),
            executor=executor,
            metadata={"monata_optimizer": {"caller": "keep"}},
        )
        opt.parameter("wn", 1e-6, 10e-6)
        opt.maximize("gain", lambda r: r.waveforms["out"][0])

        opt.run(optimizer=OneShotOptimizer({"wn": 5e-6}), n_iter=1)

        task = executor.tasks[0]
        assert task.metadata["monata_optimizer"] == {"caller": "keep"}
        assert task.metadata["monata_optimizer_2"]["params"] == {"wn": 5e-6}
