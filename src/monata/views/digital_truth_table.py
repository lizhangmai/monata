from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

from monata.sim.digital_claims import DigitalTransientObservation
from monata.sim.digital_plan import digital_task_metadata
from monata.sim.digital_recipe import DigitalResolvedSimulationRecipe, DigitalSimulationRecipe
from monata.sim.digital_results import DigitalTruthTableResult
from monata.sim.digital_table import (
    DigitalTruthTable,
    DigitalTruthTableMode,
    ExpectedTable,
    DigitalTruthTableSpec,
    build_digital_truth_table_from_spec,
    resolve_digital_measurements,
    resolve_digital_truth_table_mode,
)
from monata.sim.results import SimResult
from monata.views.base import View
from monata.views.path_safety import read_cell_json_mapping, resolve_cell_relative_path
from monata.views.simulation import SimulationProgressCallback, SimulationView

TruthTableMode = DigitalTruthTableMode


class DigitalTruthTableView(View):
    """First-class Monata view for digital truth-table verification."""

    def __init__(
        self,
        cell,
        entry: str,
        *,
        mode: str = "transient",
        config: Mapping[str, Any] | None = None,
        generated: bool = False,
        view_format: str | None = "monata-digital-truth-table-json",
        schema_version: int | None = 1,
    ):
        super().__init__(
            view_type="digital_truth_table",
            cell=cell,
            entry=entry,
            generated=generated,
            format=view_format,
            schema_version=schema_version,
        )
        self._mode = mode
        self._config = dict(config or {})

    @property
    def default_mode(self) -> str:
        return self.spec().simulation_mode

    @property
    def simulation_view_type(self) -> str:
        return "simulation"

    @property
    def metadata(self) -> dict[str, Any]:
        control_keys = {
            "entry",
            "format",
            "function",
            "mode",
            "schema_version",
            "simulation_view",
            "generated",
        }
        return {
            str(key): value
            for key, value in self._config.items()
            if str(key) not in control_keys
        }

    def spec(self) -> DigitalTruthTableSpec:
        payload = read_cell_json_mapping(
            self.path(),
            self.entry,
            label="digital_truth_table.entry",
        )
        _validate_schema_version(
            payload,
            self.schema_version,
            label="digital_truth_table",
        )
        expected = _expected_table_from_spec_payload(self.path(), payload)
        return DigitalTruthTableSpec.from_mapping(payload, expected=expected)

    def load(self, **kwargs: Any) -> DigitalTruthTable:
        table, _resolved = self._load_with_recipe(**kwargs)
        return table

    def run(
        self,
        mode: TruthTableMode | None = None,
        *,
        executor=None,
        max_workers: int | None = None,
        observation: DigitalTransientObservation | Mapping[str, Any] | None = None,
        measurements: Iterable[str] | None = None,
        artifact_dir: str | Path | None = None,
        progress: SimulationProgressCallback | None = None,
        **kwargs: Any,
    ) -> DigitalTruthTableResult:
        resolved_mode = _truth_table_mode(mode or self._mode)
        selected_measurements = resolve_digital_measurements(
            measurements,
            default=("truth_table", "max_propagation_delay"),
        )
        table, recipe = self._load_with_recipe(
            mode=resolved_mode,
            observation=observation,
            **kwargs,
        )
        simulation = self._simulation_boundary(max_workers=max_workers)

        if resolved_mode == "transient":
            tasks = table.transient_observation_tasks(
                recipe.observation,
                measurements=selected_measurements,
            )
            sim_results = simulation.run_tasks(
                tasks,
                executor=executor,
                artifact_dir=artifact_dir,
                progress=progress,
            )
            result = table.extract_transient_results(
                sim_results,
                measurements=selected_measurements,
            )
            _write_digital_run_artifacts(
                artifact_dir,
                table=table,
                mode=resolved_mode,
                result=result,
            )
            return result
        raise ValueError(f"unsupported digital truth-table mode: {resolved_mode}")

    def _load_with_recipe(
        self,
        **kwargs: Any,
    ) -> tuple[DigitalTruthTable, DigitalResolvedSimulationRecipe]:
        spec = self.spec()
        mode = _truth_table_mode(str(kwargs.pop("mode", spec.simulation_mode or self._mode)))
        if mode != _truth_table_mode(spec.simulation_mode):
            raise ValueError(
                f"digital_truth_table mode {mode!r} does not match spec simulation_mode "
                f"{spec.simulation_mode!r}"
            )
        run_config = kwargs.pop("run_config", None)
        if run_config is None:
            raise ValueError("digital_truth_table load requires run_config for recipe profile selection")
        observation = kwargs.pop("observation", None)
        cycles_per_vector = _optional_int(kwargs.pop("cycles_per_vector", None), "cycles_per_vector")
        slots_per_task = _optional_int(kwargs.pop("slots_per_task", None), "slots_per_task")
        artifacts = kwargs.pop("artifacts", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported digital_truth_table load options: {unknown}")

        simulation = self._simulation_boundary()
        loaded = simulation.load()
        if not isinstance(loaded, DigitalSimulationRecipe):
            raise TypeError(
                f"{self.simulation_view_type!r} simulation view must load a DigitalSimulationRecipe"
            )
        resolved = loaded.resolve(
            library=self.cell.library,
            run_config=run_config,
            observation=observation,
            cycles_per_vector=cycles_per_vector,
            slots_per_task=slots_per_task,
            artifacts=artifacts,
        )
        builder_kwargs = dict(resolved.builder_kwargs)
        builder_kwargs["mode"] = mode
        return (
            build_digital_truth_table_from_spec(
                self.cell.library,
                spec,
                run_config=resolved.run_config,
                **builder_kwargs,
            ),
            resolved,
        )

    def _simulation_boundary(self, *, max_workers: int | None = None) -> SimulationView:
        del max_workers
        simulation_view = self.simulation_view_type
        if simulation_view in self.cell:
            view = self.cell[simulation_view]
            if not isinstance(view, SimulationView):
                raise TypeError(f"{simulation_view!r} view is not a SimulationView")
            return view
        raise RuntimeError(f"{self.cell.qualified_name} is missing required {simulation_view!r} view")


def _truth_table_mode(value: str) -> TruthTableMode:
    return resolve_digital_truth_table_mode(value)


def _expected_table_from_spec_payload(root: Path, payload: Mapping[str, Any]) -> ExpectedTable:
    expected_ref = payload.get("expected")
    if not isinstance(expected_ref, Mapping):
        raise TypeError("digital truth-table expected must be an object")
    entry = expected_ref.get("entry")
    if not isinstance(entry, str):
        raise ValueError("digital truth-table expected.entry must be a string")
    path = resolve_cell_relative_path(root, entry, label="expected.entry")
    return ExpectedTable.from_json(path)


def _validate_schema_version(
    payload: Mapping[str, Any],
    expected: int | None,
    *,
    label: str,
) -> None:
    if expected is None:
        return
    actual = payload.get("schema_version")
    if actual != expected:
        raise ValueError(f"{label} schema_version {actual!r} does not match view config {expected}")


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, str)):
        return int(value)
    raise TypeError(f"{label} must be int-compatible")


def _write_digital_run_artifacts(
    artifact_dir: str | Path | None,
    *,
    table: DigitalTruthTable,
    mode: TruthTableMode,
    result: DigitalTruthTableResult,
    extra_sim_results: Iterable[SimResult] = (),
) -> None:
    if artifact_dir is None:
        return
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    measurements = result.measurements_as_dict()
    (root / "measurements.json").write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_payload = {
        "schema": "monata-digital-run-v1",
        "view": "digital_truth_table",
        "mode": mode,
        "dut": table.dut_name,
        "measurements": sorted(measurements),
        "tasks": _artifact_task_payloads(result, extra_sim_results=extra_sim_results),
    }
    (root / "run.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_task_payloads(
    result: DigitalTruthTableResult,
    *,
    extra_sim_results: Iterable[SimResult] = (),
) -> list[dict[str, object]]:
    payloads = []
    seen: set[object] = set()
    for sim_result in (
        *result.sim_results,
        *result.propagation_delay_sim_results,
        *tuple(extra_sim_results),
    ):
        metadata = dict(sim_result.metadata)
        artifacts = metadata.get("artifacts")
        if isinstance(artifacts, Mapping):
            directory = artifacts.get("directory")
        else:
            directory = None
        control_metadata = digital_task_metadata(metadata)
        measurements = control_metadata.get("measurements")
        stimulus = control_metadata.get("stimulus")
        index = metadata.get("simulation_artifact_index")
        task_payload: dict[str, object] = {}
        if index is not None:
            task_payload["index"] = index
        if directory is not None:
            task_payload["directory"] = directory
        if isinstance(measurements, (list, tuple)):
            task_payload["measurements"] = list(measurements)
        if isinstance(stimulus, Mapping):
            task_payload["stimulus"] = dict(stimulus)
        if task_payload:
            key = _artifact_task_identity(task_payload, sim_result)
            if key in seen:
                continue
            seen.add(key)
            payloads.append(task_payload)
    return sorted(payloads, key=_artifact_task_sort_key)


def _artifact_task_identity(task_payload: Mapping[str, object], sim_result: SimResult) -> object:
    directory = task_payload.get("directory")
    if isinstance(directory, str):
        return ("directory", directory)
    index = task_payload.get("index")
    if isinstance(index, int):
        return ("index", index)
    return ("result", id(sim_result))


def _artifact_task_sort_key(task_payload: Mapping[str, object]) -> tuple[int, int | str]:
    index = task_payload.get("index")
    if isinstance(index, int):
        return (0, index)
    directory = task_payload.get("directory")
    if isinstance(directory, str):
        return (1, directory)
    return (2, "")
