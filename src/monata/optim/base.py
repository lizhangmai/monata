"""Optimization base types and Optimizer ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Callable


_SCALES = {"linear", "log"}
_DIRECTIONS = {"minimize", "maximize"}


@dataclass(frozen=True, init=False)
class DesignVariable:
    name: str
    min: float
    max: float
    scale: str = "linear"

    def __init__(self, name: str, min: float, max: float, scale: str = "linear"):
        name = _validate_name(name, "design variable name")
        lower = _validate_finite_scalar(min, f"minimum for {name}")
        upper = _validate_finite_scalar(max, f"maximum for {name}")
        if lower >= upper:
            raise ValueError(f"design variable {name} requires min < max")
        scale = str(scale).lower()
        if scale not in _SCALES:
            expected = ", ".join(sorted(_SCALES))
            raise ValueError(f"design variable {name} has invalid scale {scale!r}; expected one of: {expected}")
        if scale == "log" and (lower <= 0 or upper <= 0):
            raise ValueError(f"log-scale design variable {name} requires positive bounds")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "min", lower)
        object.__setattr__(self, "max", upper)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True, init=False)
class Objective:
    name: str
    direction: str = "minimize"

    def __init__(self, name: str, direction: str = "minimize"):
        name = _validate_name(name, "objective name")
        direction = str(direction).lower()
        if direction not in _DIRECTIONS:
            expected = ", ".join(sorted(_DIRECTIONS))
            raise ValueError(f"objective {name} has invalid direction {direction!r}; expected one of: {expected}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, init=False)
class Constraint:
    name: str
    min: float | None = None
    max: float | None = None

    def __init__(self, name: str, min: float | None = None, max: float | None = None):
        name = _validate_name(name, "constraint name")
        lower = None if min is None else _validate_finite_scalar(min, f"minimum for constraint {name}")
        upper = None if max is None else _validate_finite_scalar(max, f"maximum for constraint {name}")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"constraint {name} requires min <= max")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "min", lower)
        object.__setattr__(self, "max", upper)


class OptimResult:
    def __init__(self, pareto_front: list[dict], all_evaluations: list[dict], best: dict, n_evaluations: int):
        self.pareto_front = pareto_front
        self.all_evaluations = all_evaluations
        self.best = best
        self.n_evaluations = n_evaluations

    def plot_pareto(self, x_obj: str, y_obj: str):
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        x = [p[x_obj] for p in self.pareto_front]
        y = [p[y_obj] for p in self.pareto_front]
        plt.scatter(x, y)
        plt.xlabel(x_obj)
        plt.ylabel(y_obj)
        plt.title("Pareto Front")
        return plt.gcf()

    def plot_convergence(self):
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        plt.plot(range(len(self.all_evaluations)))
        plt.xlabel("Evaluation")
        plt.title("Convergence")
        return plt.gcf()


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        variables: list[DesignVariable],
        objectives: list[Objective],
        constraints: list[Constraint],
        eval_fn: Callable,
        n_iter: int,
    ) -> OptimResult:
        ...


def validate_optimization_problem(
    variables: Sequence[DesignVariable],
    objectives: Sequence[Objective],
    constraints: Sequence[Constraint],
    n_iter: int,
) -> tuple[list[DesignVariable], list[Objective], list[Constraint]]:
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer")
    variable_list = list(variables)
    objective_list = list(objectives)
    constraint_list = list(constraints)
    if not variable_list:
        raise ValueError("optimizer requires at least one design variable")
    if not objective_list:
        raise ValueError("optimizer requires at least one objective")
    if not all(isinstance(variable, DesignVariable) for variable in variable_list):
        raise TypeError("variables must contain DesignVariable instances")
    if not all(isinstance(objective, Objective) for objective in objective_list):
        raise TypeError("objectives must contain Objective instances")
    if not all(isinstance(constraint, Constraint) for constraint in constraint_list):
        raise TypeError("constraints must contain Constraint instances")
    return variable_list, objective_list, constraint_list


def _validate_name(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    text = value
    if not text.strip():
        raise ValueError(f"{label} must be non-empty")
    if any(char in text for char in "\r\n;"):
        raise ValueError(f"{label} cannot contain control characters")
    return text


def _validate_finite_scalar(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite numeric scalar")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric scalar") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite numeric scalar")
    return number
