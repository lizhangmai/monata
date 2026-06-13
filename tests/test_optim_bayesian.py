from typing import Any, cast

import pytest

from monata.optim.base import DesignVariable, Objective, Constraint, OptimResult
from monata.optim.bayesian import BayesianOptimizer


class TestBayesianOptimizer:
    def test_single_objective_quadratic(self):
        """Minimize (x-3)^2, x in [0, 6]. Optimum at x=3."""
        variables = [DesignVariable("x", min=0, max=6)]
        objectives = [Objective("f", direction="minimize")]

        def eval_fn(params):
            return {"f": (params["x"] - 3) ** 2}

        opt = BayesianOptimizer(n_initial=5, seed=1)
        result = opt.optimize(variables, objectives, [], eval_fn, n_iter=25)
        assert isinstance(result, OptimResult)
        assert abs(result.best["x"] - 3.0) < 1.0
        assert result.best["f"] < 1.0

    def test_2d_optimization(self):
        """Minimize (x-1)^2 + (y-2)^2."""
        variables = [
            DesignVariable("x", min=-5, max=5),
            DesignVariable("y", min=-5, max=5),
        ]
        objectives = [Objective("f", direction="minimize")]

        def eval_fn(params):
            return {"f": (params["x"] - 1) ** 2 + (params["y"] - 2) ** 2}

        opt = BayesianOptimizer(n_initial=10, seed=2)
        result = opt.optimize(variables, objectives, [], eval_fn, n_iter=30)
        assert result.best["f"] < 2.0

    def test_maximize(self):
        """Maximize -x^2 + 4x (peak at x=2, max=4)."""
        variables = [DesignVariable("x", min=0, max=4)]
        objectives = [Objective("f", direction="maximize")]

        def eval_fn(params):
            x = params["x"]
            return {"f": -x ** 2 + 4 * x}

        opt = BayesianOptimizer(n_initial=5, seed=3)
        result = opt.optimize(variables, objectives, [], eval_fn, n_iter=20)
        assert result.best["f"] > 3.0

    def test_with_constraint(self):
        """Minimize x^2 subject to x >= 2."""
        variables = [DesignVariable("x", min=0, max=5)]
        objectives = [Objective("f", direction="minimize")]
        constraints = [Constraint("x_val", min=2)]

        def eval_fn(params):
            return {"f": params["x"] ** 2, "x_val": params["x"]}

        opt = BayesianOptimizer(n_initial=5, seed=4)
        result = opt.optimize(variables, objectives, constraints, eval_fn, n_iter=20)
        assert result.best["x_val"] >= 1.8

    def test_log_scale(self):
        """Optimize in log space."""
        variables = [DesignVariable("C", min=1e-12, max=1e-9, scale="log")]
        objectives = [Objective("f", direction="minimize")]

        def eval_fn(params):
            return {"f": abs(params["C"] - 50e-12)}

        opt = BayesianOptimizer(n_initial=8, seed=5)
        result = opt.optimize(variables, objectives, [], eval_fn, n_iter=20)
        assert result.best["f"] < 30e-12

    def test_respects_iteration_budget_smaller_than_initial_samples(self):
        variables = [DesignVariable("x", min=0, max=1)]
        objectives = [Objective("f", direction="minimize")]
        calls = []

        def eval_fn(params):
            calls.append(params)
            return {"f": params["x"]}

        result = BayesianOptimizer(n_initial=5, seed=6).optimize(
            variables,
            objectives,
            [],
            eval_fn,
            n_iter=2,
        )

        assert len(calls) == 2
        assert result.n_evaluations == 2

    def test_rejects_invalid_setup(self):
        with pytest.raises(ValueError, match="n_initial"):
            BayesianOptimizer(n_initial=0)
        with pytest.raises(ValueError, match="unsupported acquisition"):
            BayesianOptimizer(acquisition="ucb")

    def test_rejects_invalid_problem_contracts(self):
        opt = BayesianOptimizer(n_initial=1, seed=7)
        variable = DesignVariable("x", min=0, max=1)
        objective = Objective("f", direction="minimize")

        def eval_fn(params):
            return {"f": params["x"]}

        with pytest.raises(ValueError, match="at least one design variable"):
            opt.optimize([], [objective], [], eval_fn, n_iter=1)
        with pytest.raises(ValueError, match="at least one objective"):
            opt.optimize([variable], [], [], eval_fn, n_iter=1)
        with pytest.raises(ValueError, match="n_iter"):
            opt.optimize([variable], [objective], [], eval_fn, n_iter=0)
        with pytest.raises(TypeError, match="eval_fn"):
            opt.optimize([variable], [objective], [], cast(Any, None), n_iter=1)
