"""Bayesian Optimizer — Gaussian Process surrogate with Expected Improvement."""

from __future__ import annotations

import math

import numpy as np

from monata.optim.base import (
    Constraint,
    DesignVariable,
    OptimResult,
    Optimizer,
    validate_optimization_problem,
)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float) -> np.ndarray:
    """RBF (squared-exponential) kernel using numpy broadcasting."""
    # X1: (n, d), X2: (m, d) -> (n, m)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n, m, d)
    sq_dists = np.sum(diff ** 2, axis=-1)               # (n, m)
    return np.exp(-0.5 * sq_dists / length_scale ** 2)


class BayesianOptimizer(Optimizer):
    """Sample-efficient optimization using a GP surrogate model.

    Best for expensive simulations where each evaluation costs seconds/minutes.
    """

    def __init__(self, n_initial: int = 10, acquisition: str = "ei", seed: int | None = None):
        if isinstance(n_initial, bool) or not isinstance(n_initial, int) or n_initial <= 0:
            raise ValueError("n_initial must be a positive integer")
        if acquisition != "ei":
            raise ValueError(f"unsupported acquisition {acquisition!r}; expected 'ei'")
        self._n_initial = n_initial
        self._acquisition = acquisition
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

        # Single-objective only for GP (multi-obj falls back to scalarization)
        direction_signs = []
        for obj in objectives:
            direction_signs.append(-1.0 if obj.direction == "maximize" else 1.0)

        # Initial Latin Hypercube sampling
        n_initial = min(self._n_initial, n_iter)
        X = rng.random((n_initial, n_vars))
        y_values: list[float] = []
        all_evals = []

        for x in X:
            params = self._decode(x, variables)
            result = eval_fn(params)
            result.update(params)
            all_evals.append(result)
            # Scalarize objectives
            y = sum(direction_signs[j] * result[obj.name] for j, obj in enumerate(objectives))
            # Penalty for constraint violations
            y += self._constraint_penalty(result, constraints)
            y_values.append(y)

        X = np.array(X, dtype=float)
        Y = np.array(y_values, dtype=float)

        # Iterative acquisition
        for _ in range(n_iter - n_initial):
            # Find next point by maximizing acquisition
            x_next = self._maximize_acquisition(X, Y, variables, rng)

            # Evaluate
            params = self._decode(x_next, variables)
            result = eval_fn(params)
            result.update(params)
            all_evals.append(result)
            y_next = sum(direction_signs[j] * result[obj.name] for j, obj in enumerate(objectives))
            y_next += self._constraint_penalty(result, constraints)

            X = np.vstack([X, x_next.reshape(1, -1)])
            Y = np.append(Y, y_next)

        # Find best feasible
        best_idx = int(np.argmin(Y))
        best = all_evals[best_idx]

        # Pareto front (for single-obj, just the best point)
        pareto = [best]

        return OptimResult(
            pareto_front=pareto,
            all_evaluations=all_evals,
            best=best,
            n_evaluations=len(all_evals),
        )

    def _decode(self, x: np.ndarray, variables: list[DesignVariable]) -> dict:
        params = {}
        for i, var in enumerate(variables):
            val = x[i]
            if var.scale == "log":
                log_min = math.log10(var.min)
                log_max = math.log10(var.max)
                params[var.name] = 10 ** (log_min + val * (log_max - log_min))
            else:
                params[var.name] = var.min + val * (var.max - var.min)
        return params

    def _constraint_penalty(self, result: dict, constraints: list[Constraint]) -> float:
        penalty = 0.0
        for c in constraints:
            val = result.get(c.name, 0.0)
            if c.min is not None and val < c.min:
                penalty += (c.min - val) ** 2 * 1e6
            if c.max is not None and val > c.max:
                penalty += (val - c.max) ** 2 * 1e6
        return penalty

    def _gp_predict(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray):
        """Simple RBF kernel GP prediction."""
        length_scale = 0.3
        noise = 1e-6

        K = _rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(len(X_train))
        K_s = _rbf_kernel(X_train, X_test, length_scale)
        K_ss = _rbf_kernel(X_test, X_test, length_scale)

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = K_s.T @ alpha
            v = np.linalg.solve(L, K_s)
            var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
            sigma = np.sqrt(np.maximum(var, 1e-10))
        except np.linalg.LinAlgError:
            mu = np.mean(Y_train) * np.ones(len(X_test))
            sigma = np.std(Y_train) * np.ones(len(X_test))

        return mu, sigma

    def _maximize_acquisition(self, X: np.ndarray, Y: np.ndarray, variables, rng) -> np.ndarray:
        """Find next point by maximizing Expected Improvement."""
        n_vars = len(variables)
        best_y = float(np.min(Y))
        best_x = None
        best_ei = -math.inf

        # Random restarts
        for _ in range(50):
            x_candidate = rng.random(n_vars)
            mu, sigma = self._gp_predict(X, Y, x_candidate.reshape(1, -1))
            ei = _expected_improvement(float(mu[0]), float(sigma[0]), best_y)

            if ei > best_ei:
                best_ei = ei
                best_x = x_candidate

        return best_x if best_x is not None else rng.random(n_vars)


def _expected_improvement(mu: float, sigma: float, best_y: float, xi: float = 0.01) -> float:
    """Expected Improvement acquisition function (numpy/math only)."""
    if sigma <= 0.0:
        return 0.0
    z = (best_y - mu - xi) / sigma
    return (best_y - mu - xi) * _norm_cdf(z) + sigma * _norm_pdf(z)
