"""Monte Carlo statistical simulation — leverages simulator-native statistics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np

from monata.measure.statistics import histogram as _histogram
from monata.measure.statistics import sigma_yield as _sigma_yield
from monata.sim.results import SimResult
from monata.sim.task import DEFAULT_SIMULATOR, DEFAULT_SIM_TIMEOUT_SECONDS, SimArtifactOptions


class Variation:
    def __init__(
        self,
        param: str,
        distribution: str,
        sigma: float,
        nominal: float | None,
        *,
        kind: str = "process",
        pair: tuple[str, str] | None = None,
    ):
        self.param = param
        self.distribution = _variation_distribution(distribution)
        self.sigma = sigma
        self.nominal = nominal
        self.kind = kind
        self.pair = pair


class MonteCarlo:
    """Statistical simulation using simulator-native Monte Carlo.

    Generates backend-specific statistical statements:
      - ngspice: .param vth0 = agauss(nom, sigma, 3) + .control loop
      - Xyce: .SAMPLING / .EMBEDDEDSAMPLING
      - VACASK: Spectre statistics { ... } block
    """

    def __init__(
        self,
        circuit,
        analysis_spec,
        n_samples: int = 500,
        output_names=None,
        osdi_paths=None,
        metadata=None,
        simulator: str = DEFAULT_SIMULATOR,
        timeout: float | int | None = DEFAULT_SIM_TIMEOUT_SECONDS,
        backend_options: Mapping[str, Any] | None = None,
        artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None,
        seed: int | None = None,
        mode: str = "auto",
    ):
        self.circuit = circuit
        self.analysis_spec = analysis_spec
        self.n_samples = n_samples
        self.output_names = output_names
        self.osdi_paths = osdi_paths
        self.metadata = dict(metadata or {})
        self.simulator = str(simulator)
        self.timeout = timeout
        self.backend_options = dict(backend_options or {})
        self.artifacts = SimArtifactOptions.coerce(artifacts)
        self.seed = seed
        self.mode = _monte_carlo_mode(mode)
        self._variations: list[Variation] = []

    def add_variation(self, param: str, distribution: str = "gaussian", sigma: float = 0.1, nominal: float | None = None):
        self._variations.append(Variation(param, distribution, sigma, nominal, kind="process"))

    def add_process_variation(
        self,
        param: str,
        distribution: str = "gaussian",
        sigma: float = 0.1,
        nominal: float | None = None,
    ):
        self._variations.append(Variation(param, distribution, sigma, nominal, kind="process"))

    def add_global_variation(
        self,
        param: str,
        distribution: str = "gaussian",
        sigma: float = 0.1,
        nominal: float | None = None,
    ):
        self._variations.append(Variation(param, distribution, sigma, nominal, kind="global"))

    def add_mismatch(
        self,
        element_pair: tuple[str, str],
        param: str,
        sigma: float,
        nominal: float | None = None,
    ):
        self._variations.append(
            Variation(param, "gaussian", sigma, nominal, kind="mismatch", pair=element_pair)
        )

    def run(self, executor=None) -> "MonteCarloResults":
        from monata.sim.executor import LocalExecutor

        if executor is None:
            executor = LocalExecutor()

        native_runner = getattr(executor, "run_monte_carlo_native", None)
        if self.mode in {"auto", "native"} and native_runner is not None:
            native_results = native_runner(self)
            if isinstance(native_results, MonteCarloResults):
                return native_results
            return MonteCarloResults(list(native_results))
        if self.mode == "native":
            raise ValueError("native Monte Carlo mode is not available for this executor")

        return self.run_task_expanded(executor, mode="task-expanded")

    def run_task_expanded(self, executor, *, mode: str) -> "MonteCarloResults":
        """Run Monte Carlo as explicit per-sample tasks through an executor."""

        return self._run_task_expanded(executor, mode=mode)

    def _run_task_expanded(self, executor, *, mode: str) -> "MonteCarloResults":
        from monata.sim.task import SimTask

        rng = np.random.default_rng(self.seed)
        tasks = []
        for sample_index in range(self.n_samples):
            overrides = {}
            sampled: list[dict] = []
            for var in self._variations:
                nom = var.nominal if var.nominal is not None else 0.0
                if var.kind == "mismatch" and var.pair is not None:
                    if var.nominal is None:
                        raise ValueError(f"mismatch variation requires nominal value: {var.param}")
                    delta = rng.normal(0.0, var.sigma)
                    first, second = var.pair
                    overrides[f"{first}.{var.param}"] = nom + delta
                    overrides[f"{second}.{var.param}"] = nom - delta
                    sampled.append({
                        "kind": var.kind,
                        "pair": list(var.pair),
                        "param": var.param,
                        "nominal": nom,
                        "delta": delta,
                    })
                    continue
                sample = _sample_variation(rng, var, nom)
                overrides[var.param] = sample
                sampled.append({
                    "kind": var.kind,
                    "param": var.param,
                    "distribution": var.distribution,
                    "nominal": nom,
                    "sigma": var.sigma,
                    "value": sample,
                })
            metadata = dict(self.metadata)
            metadata.update({
                "sample_index": sample_index,
                "seed": self.seed,
                "sampled_overrides": dict(overrides),
                "sampled_variations": sampled,
                "monte_carlo_mode": mode,
            })
            task = SimTask(
                circuit=self.circuit,
                analysis_spec=self.analysis_spec,
                param_overrides=overrides,
                output_names=self.output_names,
                osdi_paths=self.osdi_paths,
                metadata=metadata,
                simulator=self.simulator,
                timeout=self.timeout,
                backend_options=self.backend_options,
                artifacts=self.artifacts,
            )
            tasks.append(task)

        futures = executor.map(tasks)
        results = [future.result() for future in futures]
        return MonteCarloResults(results)


def _monte_carlo_mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in {"auto", "native", "task-expanded"}:
        raise ValueError("Monte Carlo mode must be one of: auto, native, task-expanded")
    return normalized


def _variation_distribution(distribution: str) -> str:
    return str(distribution).strip().lower().replace("-", "_")


def _sample_variation(rng, var: Variation, nominal: float) -> float:
    if var.distribution == "gaussian":
        return float(rng.normal(nominal, var.sigma))
    if var.distribution == "uniform":
        return float(rng.uniform(nominal - var.sigma, nominal + var.sigma))
    if var.distribution == "relative_gaussian":
        if var.nominal is None:
            raise ValueError(f"relative_gaussian variation requires nominal value: {var.param}")
        return float(nominal * (1.0 + rng.normal(0.0, var.sigma)))
    if var.distribution == "relative_uniform":
        if var.nominal is None:
            raise ValueError(f"relative_uniform variation requires nominal value: {var.param}")
        return float(nominal * (1.0 + rng.uniform(-var.sigma, var.sigma)))
    raise ValueError(f"unsupported Monte Carlo distribution: {var.distribution}")


class MonteCarloResults:
    def __init__(self, results: list[SimResult]):
        self._results = results

    def samples(self) -> list[SimResult]:
        return self._results

    def extract(self, metric_fn: Callable) -> np.ndarray:
        return np.array([metric_fn(r) for r in self._results if r.status == "ok"])

    def histogram(self, metric_fn: Callable, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        bin_edges, counts = _histogram(self._passing_metric_values(metric_fn), bins=bins)
        return np.asarray(bin_edges), counts

    def sigma_yield(self, metric_fn: Callable, spec_min: float | None = None, spec_max: float | None = None) -> float:
        return _sigma_yield(self._passing_metric_values(metric_fn), spec_min=spec_min, spec_max=spec_max)

    def to_arrays(self, metrics: dict[str, Callable] | None = None) -> dict[str, np.ndarray]:
        if metrics is None:
            return {"status": np.asarray([r.status for r in self._results], dtype=object)}
        columns = {name: [] for name in metrics}
        for result in self._results:
            if result.status == "ok":
                for name, fn in metrics.items():
                    columns[name].append(fn(result))
            else:
                for name in metrics:
                    columns[name].append(np.nan)
        return {name: np.asarray(values) for name, values in columns.items()}

    def _passing_metric_values(self, metric_fn: Callable) -> np.ndarray:
        values = self.extract(metric_fn)
        if values.size == 0:
            raise ValueError("No passing samples to evaluate")
        return values
