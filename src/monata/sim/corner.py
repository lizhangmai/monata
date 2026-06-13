"""CornerMatrix helpers for PVT-style operating-corner sweeps."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from monata.corner import OperatingCorner, corner_to_payload
from monata.measure.statistics import histogram as _histogram
from monata.measure.statistics import sigma_yield as _sigma_yield
from monata.measure.statistics import worst_case as _worst_case
from monata.sim.results import SimResult
from monata.sim.task import DEFAULT_SIMULATOR, DEFAULT_SIM_TIMEOUT_SECONDS, SimArtifactOptions

_MODEL_FILE_SUFFIXES = frozenset({".cir", ".ckt", ".lib", ".mod", ".osdi", ".scs", ".sp", ".spice"})


class CornerMatrix:
    def __init__(
        self,
        circuit,
        analysis_spec,
        output_names=None,
        osdi_paths=None,
        metadata=None,
        simulator: str = DEFAULT_SIMULATOR,
        timeout: float | int | None = DEFAULT_SIM_TIMEOUT_SECONDS,
        backend_options: Mapping[str, Any] | None = None,
        artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None,
        model_manifest=None,
    ):
        self.circuit = circuit
        self.analysis_spec = analysis_spec
        self.output_names = output_names
        self.osdi_paths = osdi_paths
        self.metadata = dict(metadata or {})
        self.simulator = str(simulator)
        self.timeout = timeout
        self.backend_options = dict(backend_options or {})
        self.artifacts = SimArtifactOptions.coerce(artifacts)
        self.model_manifest = model_manifest
        self._temperatures: list[float] = []
        self._voltages: dict[str, list[float]] = {}
        self._model_corners: dict[str, str] = {}

    def add_temperatures(self, *temps: float):
        self._temperatures.extend(temps)

    def add_voltages(self, name: str, *values: float):
        self._voltages[name] = list(values)

    def add_model_corners(self, **corners: str):
        if self.model_manifest is None:
            raise ValueError("CornerMatrix.add_model_corners requires model_manifest")
        for process, selection in corners.items():
            if _looks_like_model_file_path(selection):
                raise ValueError(
                    "CornerMatrix.add_model_corners accepts model selection names, "
                    "not model file paths; register files in model_manifest"
                )
            self._model_corners[str(process)] = str(selection)

    def corners(self) -> list[OperatingCorner]:
        temps = self._temperatures or [27]
        voltage_keys = list(self._voltages.keys())
        voltage_combos = list(product(*(self._voltages[k] for k in voltage_keys))) if voltage_keys else [()]
        processes = list(self._model_corners.items()) if self._model_corners else [(None, None)]

        result = []
        for temp in temps:
            for v_combo in voltage_combos:
                voltage_dict = dict(zip(voltage_keys, v_combo)) if voltage_keys else {}
                for proc_name, model_ref in processes:
                    v_str = "_".join(f"{v:.1f}" for v in v_combo) if v_combo else ""
                    p_str = f"{proc_name}_" if proc_name else ""
                    name = f"{p_str}{temp}C{'_' + v_str if v_str else ''}"
                    corner = OperatingCorner(
                        name=name,
                        temperature=temp,
                        voltages=voltage_dict,
                        process=proc_name,
                        metadata={"model_selection": model_ref} if model_ref else {},
                    )
                    result.append(corner)
        return result

    def run(self, executor=None) -> "CornerResults":
        from monata.sim.task import SimTask
        tasks = []
        for corner in self.corners():
            selection = self._resolve_corner_selection(corner)
            if selection is not None:
                model_files = [entry.model_file for entry in selection.entries if entry.model_file]
                if model_files:
                    corner = corner.with_updates(model_file=str(model_files[0]))
            metadata = dict(self.metadata)
            metadata["corner"] = corner_to_payload(corner)
            if selection is not None:
                metadata.update(selection.task_metadata())
            circuit = self.circuit
            if selection is not None and circuit is not None:
                circuit = selection.apply_to_circuit(deepcopy(circuit))
            task = SimTask(
                circuit=circuit,
                analysis_spec=self.analysis_spec,
                corner=corner,
                output_names=self.output_names,
                osdi_paths=_merged_paths(self.osdi_paths, selection.osdi_paths if selection else ()),
                metadata=metadata,
                simulator=self.simulator,
                timeout=self.timeout,
                backend_options=self.backend_options,
                artifacts=self.artifacts,
            )
            tasks.append(task)
        if executor is None:
            from monata.sim.executor import LocalExecutor
            executor = LocalExecutor()
        futures = executor.map(tasks)
        results = [f.result() for f in futures]
        return CornerResults(results)

    def _resolve_corner_selection(self, corner: OperatingCorner):
        if self.model_manifest is None or corner.process is None:
            return None
        selection_name = corner.metadata.get("model_selection") or corner.process
        return self.model_manifest.resolve(name=selection_name)


def _merged_paths(first, second) -> list:
    result = []
    seen = set()
    for path in [*(first or ()), *(second or ())]:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _looks_like_model_file_path(value: object) -> bool:
    text = str(value)
    if "/" in text or "\\" in text:
        return True
    return any(text.lower().endswith(suffix) for suffix in _MODEL_FILE_SUFFIXES)


class CornerResults:
    def __init__(self, results: list[SimResult]):
        self._results = results

    def __iter__(self) -> Iterator[SimResult]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, corner_name: str) -> SimResult:
        for r in self._results:
            if r.corner and r.corner.name == corner_name:
                return r
        raise KeyError(f"No result for corner: {corner_name}")

    def filter(self, temperature=None, voltages=None, process=None) -> list[SimResult]:
        results = self._results
        if temperature is not None:
            results = [r for r in results if r.corner and r.corner.temperature == temperature]
        if voltages is not None:
            results = [r for r in results if r.corner and r.corner.voltages == voltages]
        if process is not None:
            results = [r for r in results if r.corner and r.corner.process == process]
        return results

    def passed(self) -> list[SimResult]:
        return [r for r in self._results if r.status == "ok"]

    def failed(self) -> list[SimResult]:
        return [r for r in self._results if r.status == "failed"]

    def extract(self, metric_fn: Callable) -> np.ndarray:
        return np.array([metric_fn(result) for result in self.passed()])

    def histogram(self, metric_fn: Callable, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        bin_edges, counts = _histogram(self._passing_metric_values(metric_fn), bins=bins)
        return np.asarray(bin_edges), counts

    def sigma_yield(self, metric_fn: Callable, spec_min: float | None = None, spec_max: float | None = None) -> float:
        return _sigma_yield(self._passing_metric_values(metric_fn), spec_min=spec_min, spec_max=spec_max)

    def worst_case(self, metric_fn: Callable) -> SimResult:
        values = self._passing_metric_values(metric_fn)
        _worst_case(values)
        return self.passed()[int(np.argmin(values))]

    def to_arrays(self, metrics: dict[str, Callable] | None = None) -> dict[str, np.ndarray]:
        voltage_names = sorted({
            name
            for result in self._results
            if result.corner is not None
            for name in result.corner.voltages
        })
        metric_fns = {str(name): fn for name, fn in dict(metrics or {}).items()}
        metadata_names = {"corner", "temperature", "process", "model_file", "status", *voltage_names}
        conflicts = sorted(set(metric_fns).intersection(metadata_names))
        if conflicts:
            names = ", ".join(conflicts)
            raise ValueError(f"metric names conflict with corner metadata columns: {names}")
        metric_names = tuple(metric_fns)
        columns: dict[str, list] = {
            "corner": [],
            "temperature": [],
            "process": [],
            "model_file": [],
            "status": [],
            **{name: [] for name in voltage_names},
            **{name: [] for name in metric_names},
        }
        for result in self._results:
            corner = result.corner
            columns["corner"].append(corner.name if corner else None)
            columns["temperature"].append(corner.temperature if corner else np.nan)
            columns["process"].append(corner.process if corner else None)
            columns["model_file"].append(corner.model_file if corner else None)
            columns["status"].append(result.status)
            for name in voltage_names:
                columns[name].append(corner.voltages.get(name, np.nan) if corner else np.nan)
            if metric_fns:
                for name, fn in metric_fns.items():
                    columns[name].append(fn(result) if result.status == "ok" else np.nan)
        return {
            name: _corner_array_column(name, values, voltage_names, metric_names)
            for name, values in columns.items()
        }

    def _passing_metric_values(self, metric_fn: Callable) -> np.ndarray:
        values = self.extract(metric_fn)
        if values.size == 0:
            raise ValueError("No passing results to evaluate")
        return values


def _corner_array_column(
    name: str,
    values: list,
    voltage_names: list[str],
    metric_names: tuple[str, ...],
) -> np.ndarray:
    if name in {"temperature", *voltage_names}:
        return np.asarray(values, dtype=float)
    if name in metric_names:
        return np.asarray(values)
    return np.asarray(values, dtype=object)


__all__ = ["CornerMatrix", "CornerResults"]
