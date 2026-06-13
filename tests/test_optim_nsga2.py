from typing import Any, cast

import pytest

from monata.optim.base import DesignVariable, Objective, Constraint
from monata.optim.nsga2 import NSGA2Optimizer


class TestNSGA2:
    def test_single_objective(self):
        """Minimize x^2, x in [-5, 5]. Optimum at x=0."""
        variables = [DesignVariable("x", min=-5, max=5)]
        objectives = [Objective("f", direction="minimize")]

        def eval_fn(params):
            return {"f": params["x"] ** 2}

        opt = NSGA2Optimizer(population_size=20, seed=1)
        result = opt.optimize(variables, objectives, [], eval_fn, n_iter=50)
        assert abs(result.best["x"]) < 0.5
        assert result.best["f"] < 0.25

    def test_two_objectives(self):
        """Bi-objective: minimize x^2 and (x-2)^2. Pareto front is [0, 2]."""
        variables = [DesignVariable("x", min=-1, max=4)]
        objectives = [
            Objective("f1", direction="minimize"),
            Objective("f2", direction="minimize"),
        ]

        def eval_fn(params):
            x = params["x"]
            return {"f1": x ** 2, "f2": (x - 2) ** 2}

        opt = NSGA2Optimizer(population_size=50, seed=2)
        result = opt.optimize(variables, objectives, [], eval_fn, n_iter=100)
        assert len(result.pareto_front) > 1
        xs = [p["x"] for p in result.pareto_front]
        assert min(xs) < 1.0
        assert max(xs) > 1.0

    def test_with_constraint(self):
        """Minimize x^2 subject to x >= 1."""
        variables = [DesignVariable("x", min=-5, max=5)]
        objectives = [Objective("f", direction="minimize")]
        constraints = [Constraint("x_val", min=1)]

        def eval_fn(params):
            return {"f": params["x"] ** 2, "x_val": params["x"]}

        opt = NSGA2Optimizer(population_size=20, seed=3)
        result = opt.optimize(variables, objectives, constraints, eval_fn, n_iter=50)
        assert result.best["x"] >= 0.9  # near constraint boundary

    def test_log_scale_variable(self):
        """Variable with log scale should explore orders of magnitude."""
        variables = [DesignVariable("C", min=1e-12, max=1e-9, scale="log")]
        objectives = [Objective("f", direction="minimize")]

        def eval_fn(params):
            target = 100e-12
            return {"f": abs(params["C"] - target)}

        opt = NSGA2Optimizer(population_size=20, seed=4)
        result = opt.optimize(variables, objectives, [], eval_fn, n_iter=50)
        assert abs(result.best["C"] - 100e-12) < 50e-12

    def test_rejects_invalid_setup(self):
        with pytest.raises(ValueError, match="population_size"):
            NSGA2Optimizer(population_size=0)
        with pytest.raises(ValueError, match="crossover_prob"):
            NSGA2Optimizer(crossover_prob=1.2)
        with pytest.raises(ValueError, match="mutation_prob"):
            NSGA2Optimizer(mutation_prob=float("nan"))

    def test_rejects_invalid_problem_contracts(self):
        opt = NSGA2Optimizer(population_size=4, seed=5)
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
