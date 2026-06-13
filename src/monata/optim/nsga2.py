"""NSGA-II multi-objective genetic algorithm."""

from __future__ import annotations

import math

import numpy as np

from monata.optim.base import (
    Constraint,
    DesignVariable,
    Objective,
    OptimResult,
    Optimizer,
    validate_optimization_problem,
)


class NSGA2Optimizer(Optimizer):
    def __init__(
        self,
        population_size: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        seed: int | None = None,
    ):
        if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size <= 0:
            raise ValueError("population_size must be a positive integer")
        crossover_prob = _validate_probability(crossover_prob, "crossover_prob")
        mutation_prob = _validate_probability(mutation_prob, "mutation_prob")
        self._pop_size = population_size
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._seed = seed

    def optimize(self, variables, objectives, constraints, eval_fn, n_iter) -> OptimResult:
        variables, objectives, constraints = validate_optimization_problem(
            variables,
            objectives,
            constraints,
            n_iter,
        )
        if not callable(eval_fn):
            raise TypeError("eval_fn must be callable")
        rng = np.random.default_rng(self._seed)
        n_vars = len(variables)

        # Initialize population in normalized [0,1] space
        pop = rng.random((self._pop_size, n_vars))
        all_evals = []

        for gen in range(n_iter):
            # Decode and evaluate
            decoded = [self._decode(ind, variables) for ind in pop]
            fitnesses = []
            for params in decoded:
                result = eval_fn(params)
                result.update(params)
                all_evals.append(result)
                fitnesses.append(result)

            # Non-dominated sorting
            obj_values = self._extract_objectives(fitnesses, objectives)
            fronts = self._fast_non_dominated_sort(obj_values, objectives)

            # Constraint handling
            feasible = self._check_constraints(fitnesses, constraints)

            # Selection + offspring
            offspring = self._generate_offspring(pop, fronts, feasible, obj_values, rng, variables)
            pop = offspring

        # Final evaluation
        decoded = [self._decode(ind, variables) for ind in pop]
        final_fits = []
        for params in decoded:
            result = eval_fn(params)
            result.update(params)
            final_fits.append(result)
            all_evals.append(result)

        # Filter for feasibility before building pareto front
        feasible_mask = self._check_constraints(final_fits, constraints)
        feasible_fits = [f for f, ok in zip(final_fits, feasible_mask) if ok]

        if feasible_fits:
            feas_obj = self._extract_objectives(feasible_fits, objectives)
            feas_fronts = self._fast_non_dominated_sort(feas_obj, objectives)
            pareto = [feasible_fits[i] for i in feas_fronts[0]] if feas_fronts else feasible_fits[:1]
        else:
            # No feasible solutions — fall back to full population pareto front
            obj_values = self._extract_objectives(final_fits, objectives)
            fronts = self._fast_non_dominated_sort(obj_values, objectives)
            pareto = [final_fits[i] for i in fronts[0]] if fronts else final_fits[:1]

        # Best = first pareto point (for single-obj) or lowest sum of normalized objectives
        best = self._select_best(pareto, objectives)

        return OptimResult(
            pareto_front=pareto,
            all_evaluations=all_evals,
            best=best,
            n_evaluations=len(all_evals),
        )

    def _decode(self, individual: np.ndarray, variables: list[DesignVariable]) -> dict:
        params = {}
        for i, var in enumerate(variables):
            x = individual[i]
            if var.scale == "log":
                log_min = np.log10(var.min)
                log_max = np.log10(var.max)
                params[var.name] = 10 ** (log_min + x * (log_max - log_min))
            else:
                params[var.name] = var.min + x * (var.max - var.min)
        return params

    def _extract_objectives(self, fitnesses: list[dict], objectives: list[Objective]) -> np.ndarray:
        values = np.zeros((len(fitnesses), len(objectives)))
        for i, fit in enumerate(fitnesses):
            for j, obj in enumerate(objectives):
                val = fit.get(obj.name, 0.0)
                if obj.direction == "maximize":
                    val = -val
                values[i, j] = val
        return values

    def _fast_non_dominated_sort(self, obj_values: np.ndarray, objectives: list[Objective]) -> list[list[int]]:
        n = len(obj_values)
        domination_count = np.zeros(n, dtype=int)
        dominated_set: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(obj_values[i], obj_values[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(obj_values[j], obj_values[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a <= b) and np.any(a < b))

    def _check_constraints(self, fitnesses: list[dict], constraints: list[Constraint]) -> np.ndarray:
        feasible = np.ones(len(fitnesses), dtype=bool)
        for i, fit in enumerate(fitnesses):
            for c in constraints:
                val = fit.get(c.name, 0.0)
                if c.min is not None and val < c.min:
                    feasible[i] = False
                if c.max is not None and val > c.max:
                    feasible[i] = False
        return feasible

    def _crowding_distance(self, obj_values: np.ndarray, front: list[int]) -> np.ndarray:
        """Compute crowding distance for individuals in a front."""
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        n_obj = obj_values.shape[1]
        distances = np.zeros(n)
        for m in range(n_obj):
            vals = obj_values[front, m]
            order = np.argsort(vals)
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            val_range = vals[order[-1]] - vals[order[0]]
            if val_range == 0:
                continue
            for k in range(1, n - 1):
                distances[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / val_range
        return distances

    def _generate_offspring(self, pop, fronts, feasible, obj_values, rng, variables):
        n = len(pop)
        n_vars = pop.shape[1]
        offspring = np.empty_like(pop)

        # Assign rank and crowding distance per individual
        rank = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        for front_idx, front in enumerate(fronts):
            for i in front:
                rank[i] = front_idx
            cd = self._crowding_distance(obj_values, front)
            for k, i in enumerate(front):
                crowding[i] = cd[k]

        def tournament(a, b):
            """Return winner index: prefer feasible, then lower rank, then higher crowding."""
            if feasible[a] and not feasible[b]:
                return a
            if feasible[b] and not feasible[a]:
                return b
            if rank[a] < rank[b]:
                return a
            if rank[b] < rank[a]:
                return b
            return a if crowding[a] >= crowding[b] else b

        for i in range(n):
            a, b = rng.integers(0, n, size=2)
            offspring[i] = pop[tournament(a, b)]

        # SBX crossover
        for i in range(0, n - 1, 2):
            if rng.random() < self._crossover_prob:
                beta = rng.random(n_vars)
                p1, p2 = offspring[i].copy(), offspring[i + 1].copy()
                offspring[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                offspring[i + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

        # Polynomial mutation
        for i in range(n):
            for j in range(n_vars):
                if rng.random() < self._mutation_prob:
                    offspring[i, j] += rng.normal(0, 0.1)

        # Clip to [0, 1]
        offspring = np.clip(offspring, 0, 1)
        return offspring

    def _select_best(self, pareto: list[dict], objectives: list[Objective]) -> dict:
        if len(pareto) == 1:
            return pareto[0]
        # For multi-objective: pick the point with lowest normalized sum
        obj_names = [o.name for o in objectives]
        values = np.array([[p[n] for n in obj_names] for p in pareto])
        # Normalize each column
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        normalized = (values - mins) / ranges
        # Flip maximize objectives
        for j, obj in enumerate(objectives):
            if obj.direction == "maximize":
                normalized[:, j] = 1 - normalized[:, j]
        sums = normalized.sum(axis=1)
        return pareto[int(np.argmin(sums))]


def _validate_probability(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite probability in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite probability in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite probability in [0, 1]")
    return number
