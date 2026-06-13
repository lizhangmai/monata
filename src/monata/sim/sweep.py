"""Parameter sweep — 1D and 2D sweeps over design variables."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

from monata.sim.results import SimResult
from monata.sim.task import DEFAULT_SIMULATOR, DEFAULT_SIM_TIMEOUT_SECONDS, SimArtifactOptions, SimTask


class ParameterSweep:
    def __init__(
        self,
        circuit,
        analysis_spec,
        corner=None,
        output_names=None,
        osdi_paths=None,
        metadata=None,
        simulator: str = DEFAULT_SIMULATOR,
        timeout: float | int | None = DEFAULT_SIM_TIMEOUT_SECONDS,
        backend_options: Mapping[str, Any] | None = None,
        artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None,
    ):
        self.circuit = circuit
        self.analysis_spec = analysis_spec
        self.corner = corner
        self.output_names = output_names
        self.osdi_paths = osdi_paths
        self.metadata = dict(metadata or {})
        self.simulator = str(simulator)
        self.timeout = timeout
        self.backend_options = dict(backend_options or {})
        self.artifacts = SimArtifactOptions.coerce(artifacts)
        self._param: str | None = None
        self._values: list = []
        self._param2: str | None = None
        self._values2: list = []

    def sweep(self, param: str, values, param2: str | None = None, values2=None):
        if (param2 is None) != (values2 is None):
            raise ValueError("param2 and values2 must be provided together")
        self._param = param
        self._values = list(values)
        self._param2 = None
        self._values2 = []
        if param2 is not None:
            if values2 is None:
                raise ValueError("param2 and values2 must be provided together")
            self._param2 = param2
            self._values2 = list(values2)

    def tasks(self) -> list[SimTask]:
        if self._param is None:
            raise ValueError("ParameterSweep.sweep must be configured before tasks are created")
        task_list = []
        if self._param2 is not None:
            for v1, v2 in product(self._values, self._values2):
                overrides = {self._param: v1, self._param2: v2}
                metadata = self._task_metadata(overrides)
                task_list.append(SimTask(
                    circuit=self.circuit,
                    analysis_spec=self.analysis_spec,
                    corner=self.corner,
                    param_overrides=overrides,
                    output_names=self.output_names,
                    osdi_paths=self.osdi_paths,
                    metadata=metadata,
                    simulator=self.simulator,
                    timeout=self.timeout,
                    backend_options=self.backend_options,
                    artifacts=self.artifacts,
                ))
        else:
            for v in self._values:
                overrides = {self._param: v}
                metadata = self._task_metadata(overrides)
                task_list.append(SimTask(
                    circuit=self.circuit,
                    analysis_spec=self.analysis_spec,
                    corner=self.corner,
                    param_overrides=overrides,
                    output_names=self.output_names,
                    osdi_paths=self.osdi_paths,
                    metadata=metadata,
                    simulator=self.simulator,
                    timeout=self.timeout,
                    backend_options=self.backend_options,
                    artifacts=self.artifacts,
                ))
        return task_list

    def _task_metadata(self, overrides: dict) -> dict:
        metadata = dict(self.metadata)
        metadata["sweep_overrides"] = dict(overrides)
        return metadata

    def run(self, executor=None) -> "SweepResults":
        tasks = self.tasks()
        if self._param is None:
            raise ValueError("ParameterSweep.sweep must be configured before tasks are run")
        if executor is None:
            from monata.sim.executor import LocalExecutor
            executor = LocalExecutor()
        futures = executor.map(tasks)
        results = [f.result() for f in futures]
        return SweepResults(
            results,
            param_name=self._param,
            param_values=np.array(self._values),
            param2_name=self._param2,
            param2_values=np.array(self._values2),
        )


class SweepResults:
    def __init__(
        self,
        results: list[SimResult],
        param_name: str,
        param_values: np.ndarray,
        param2_name: str | None = None,
        param2_values: np.ndarray | None = None,
    ):
        self._results = results
        self._param_name = param_name
        self._param_values = param_values
        self._param2_name = param2_name
        self._param2_values = param2_values if param2_values is not None else np.array([])

    def values(self) -> np.ndarray:
        return self._param_values

    def values2(self) -> np.ndarray:
        return self._param2_values

    def result_at(
        self,
        value,
        value2=None,
        *,
        nearest: bool = False,
        tolerance: float | int | None = None,
    ) -> SimResult:
        idx = _axis_index(
            self._param_values,
            value,
            self._param_name,
            nearest=nearest,
            tolerance=tolerance,
        )
        if self._param2_name is None:
            return self._results[idx]
        if value2 is None:
            raise ValueError("2D sweep result lookup requires value2")
        idx2 = _axis_index(
            self._param2_values,
            value2,
            self._param2_name,
            nearest=nearest,
            tolerance=tolerance,
        )
        idx = idx * len(self._param2_values) + idx2
        return self._results[idx]

    def extract(self, metric_fn: Callable) -> np.ndarray:
        return np.array([metric_fn(r) for r in self._results])

    def extract_grid(self, metric_fn: Callable) -> np.ndarray:
        self._validate_result_count()
        values = self.extract(metric_fn)
        axis_shape = self._axis_shape()
        if values.shape[0] != len(self._results):
            raise ValueError(
                f"sweep metric extraction produced {values.shape[0]} values for {len(self._results)} result(s)"
            )
        return values.reshape((*axis_shape, *values.shape[1:]))

    def to_arrays(self, metrics: dict[str, Callable] | None = None) -> dict[str, np.ndarray]:
        self._validate_result_count()
        metric_fns = {str(name): fn for name, fn in dict(metrics or {}).items()}
        metadata_names = {self._param_name, "status"}
        if self._param2_name is not None:
            metadata_names.add(self._param2_name)
        conflicts = sorted(set(metric_fns).intersection(metadata_names))
        if conflicts:
            names = ", ".join(conflicts)
            raise ValueError(f"metric names conflict with sweep metadata columns: {names}")

        columns: dict[str, list[Any]] = {
            self._param_name: [],
            "status": [],
            **{name: [] for name in metric_fns},
        }
        if self._param2_name is not None:
            columns = {
                self._param_name: [],
                self._param2_name: [],
                "status": [],
                **{name: [] for name in metric_fns},
            }

        for index, result in enumerate(self._results):
            value1, value2 = self._axis_value_at(index)
            columns[self._param_name].append(value1)
            if self._param2_name is not None:
                columns[self._param2_name].append(value2)
            columns["status"].append(result.status)
            for name, fn in metric_fns.items():
                columns[name].append(fn(result) if result.status == "ok" else np.nan)
        return {name: np.asarray(values) for name, values in columns.items()}

    def _axis_shape(self) -> tuple[int, ...]:
        if self._param2_name is None:
            return (len(self._param_values),)
        return (len(self._param_values), len(self._param2_values))

    def _axis_result_count(self) -> int:
        count = len(self._param_values)
        if self._param2_name is not None:
            count *= len(self._param2_values)
        return count

    def _validate_result_count(self) -> None:
        expected = self._axis_result_count()
        if len(self._results) != expected:
            raise ValueError(f"sweep has {len(self._results)} result(s) for {expected} sweep point(s)")

    def _axis_value_at(self, index: int) -> tuple[Any, Any | None]:
        if self._param2_name is None:
            return self._param_values[index], None
        axis2_count = len(self._param2_values)
        index1, index2 = divmod(index, axis2_count)
        return self._param_values[index1], self._param2_values[index2]


def _axis_index(
    values: np.ndarray,
    value: Any,
    name: str,
    *,
    nearest: bool,
    tolerance: float | int | None,
) -> int:
    if tolerance is not None and tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    matches = np.flatnonzero(values == value)
    if matches.size:
        return int(matches[0])
    if not nearest:
        raise KeyError(f"sweep value not found for {name}: {value!r}")

    try:
        numeric_values = values.astype(float)
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"nearest lookup for {name} requires numeric sweep values") from exc

    distances = np.abs(numeric_values - numeric_value)
    index = int(np.argmin(distances))
    if tolerance is not None and distances[index] > tolerance:
        raise KeyError(
            f"nearest sweep value for {name} is outside tolerance: "
            f"value={value!r}, nearest={values[index]!r}, tolerance={tolerance!r}"
        )
    return index
