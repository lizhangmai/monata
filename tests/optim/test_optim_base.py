import pytest

from monata.optim.base import Constraint, DesignVariable, Objective, OptimResult, Optimizer


class TestDesignVariable:
    def test_linear(self):
        dv = DesignVariable("M1.W", min=1e-6, max=10e-6)
        assert dv.name == "M1.W"
        assert dv.min == 1e-6
        assert dv.max == 10e-6
        assert dv.scale == "linear"

    def test_log_scale(self):
        dv = DesignVariable("Cc", min=0.1e-12, max=10e-12, scale="log")
        assert dv.scale == "log"

    def test_rejects_invalid_bounds(self):
        with pytest.raises(ValueError, match="min < max"):
            DesignVariable("W", min=10, max=1)

    def test_rejects_invalid_scale(self):
        with pytest.raises(ValueError, match="invalid scale"):
            DesignVariable("W", min=1, max=10, scale="sqrt")

    def test_rejects_log_scale_non_positive_bounds(self):
        with pytest.raises(ValueError, match="positive bounds"):
            DesignVariable("W", min=0, max=10, scale="log")

    def test_rejects_non_finite_bounds(self):
        with pytest.raises(ValueError, match="finite numeric scalar"):
            DesignVariable("W", min=1, max=float("inf"))

    def test_rejects_non_string_name(self):
        with pytest.raises(TypeError, match="must be a string"):
            DesignVariable(123, min=1, max=2)  # type: ignore[arg-type]


class TestObjective:
    def test_minimize(self):
        obj = Objective("power", direction="minimize")
        assert obj.name == "power"
        assert obj.direction == "minimize"

    def test_maximize(self):
        obj = Objective("GBW", direction="maximize")
        assert obj.direction == "maximize"

    def test_rejects_invalid_direction(self):
        with pytest.raises(ValueError, match="invalid direction"):
            Objective("power", direction="lowest")


class TestConstraint:
    def test_min_only(self):
        c = Constraint("PM", min=60)
        assert c.name == "PM"
        assert c.min == 60
        assert c.max is None

    def test_both_bounds(self):
        c = Constraint("area", min=0, max=1000)
        assert c.min == 0
        assert c.max == 1000

    def test_rejects_reversed_bounds(self):
        with pytest.raises(ValueError, match="min <= max"):
            Constraint("area", min=10, max=1)

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            Constraint("")


class TestOptimResult:
    def test_creation(self):
        pareto = [{"W": 2e-6, "power": 100e-6, "GBW": 200e6}]
        result = OptimResult(
            pareto_front=pareto,
            all_evaluations=[{"W": 1e-6}, {"W": 2e-6}],
            best=pareto[0],
            n_evaluations=50,
        )
        assert result.n_evaluations == 50
        assert result.best["GBW"] == 200e6
        assert len(result.pareto_front) == 1


class TestOptimizerABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Optimizer()  # type: ignore[abstract]

    def test_subclass_must_implement(self):
        class BadOptimizer(Optimizer):
            pass
        with pytest.raises(TypeError):
            BadOptimizer()  # type: ignore[abstract]
