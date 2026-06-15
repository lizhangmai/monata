from __future__ import annotations

from collections.abc import Iterable, Mapping
import inspect
import json
from pathlib import Path
from typing import Any, cast

from monata.sim.digital_claims import DigitalTransientObservation, ExpectedFn
from monata.sim.digital_plan import digital_task_metadata
from monata.sim.digital_results import DigitalTruthTableResult
from monata.sim.digital_table import (
    DigitalTruthTable,
    DigitalTruthTableMode,
    ExpectedTable,
    DigitalTruthTableSpec,
    resolve_digital_measurements,
    resolve_digital_truth_table_mode,
)
from monata.sim.results import SimResult
from monata.views.base import View
from monata.views.declarative import schematic_view_to_subcircuit
from monata.views.simulation import SimulationProgressCallback, SimulationView

TruthTableMode = DigitalTruthTableMode


class DigitalTruthTableView(View):
    """First-class Monata view for digital truth-table verification."""

    def __init__(
        self,
        cell,
        entry: str,
        function_name: str = "build_truth_table",
        *,
        mode: str = "transient",
        simulation_view: str = "simulation",
        config: Mapping[str, Any] | None = None,
        generated: bool = False,
    ):
        super().__init__(view_type="digital_truth_table", cell=cell, entry=entry, generated=generated)
        self._function_name = function_name
        self._mode = mode
        self._simulation_view = simulation_view
        self._config = dict(config or {})

    @property
    def default_mode(self) -> str:
        return self._mode

    @property
    def simulation_view_type(self) -> str:
        return self._simulation_view

    @property
    def metadata(self) -> dict[str, Any]:
        control_keys = {"entry", "function", "mode", "simulation_view", "generated"}
        return {
            str(key): value
            for key, value in self._config.items()
            if str(key) not in control_keys
        }

    def spec(self) -> Any:
        return self.load_python_attribute("digital_truth_table", "SPEC")

    def load(self, **kwargs: Any) -> DigitalTruthTable:
        spec = self.spec()
        if not isinstance(spec, DigitalTruthTableSpec):
            raise TypeError("digital_truth_table view must define SPEC as a DigitalTruthTableSpec")
        simulation = self._simulation_boundary()
        factory = simulation.load()
        table = _call_view_factory(
            factory,
            cell=self.cell,
            view=self,
            simulation=simulation,
            spec=spec,
            **kwargs,
        )
        if isinstance(table, DigitalTruthTable):
            return table
        if isinstance(table, Mapping):
            return self._table_from_mapping(table)
        raise TypeError(
            "digital_truth_table view factory must return DigitalTruthTable or a mapping"
        )

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
        table = self.load(mode=resolved_mode, **kwargs)
        simulation = self._simulation_boundary(max_workers=max_workers)

        if resolved_mode == "transient":
            tasks = table.transient_observation_tasks(
                observation,
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

    def _simulation_boundary(self, *, max_workers: int | None = None) -> SimulationView:
        if self._simulation_view in self.cell:
            view = self.cell[self._simulation_view]
            if not isinstance(view, SimulationView):
                raise TypeError(f"{self._simulation_view!r} view is not a SimulationView")
            return view
        return SimulationView(
            cell=self.cell,
            entry="simulation.py",
            function_name="main",
            max_workers=max_workers,
        )

    def _table_from_mapping(self, config: Mapping[str, Any]) -> DigitalTruthTable:
        dut = config.get("dut")
        if not isinstance(dut, str):
            raise ValueError("digital_truth_table mapping requires string 'dut'")
        library = self.cell.library
        schematic_cls = schematic_view_to_subcircuit(
            library[dut]["schematic"],
            allow_trusted_python=False,
            reason="digital_truth_table DUT",
        )
        dependencies = [
            schematic_view_to_subcircuit(
                library[str(name)]["schematic"],
                allow_trusted_python=False,
                reason="digital_truth_table dependency",
            )
            for name in config.get("dependencies", ())
        ]
        expected = config.get("expected")
        if expected is not None and not callable(expected) and not isinstance(expected, ExpectedTable):
            expected = ExpectedTable.from_rows(cast(Any, expected))
        expected_input = cast(ExpectedFn | ExpectedTable | None, expected)
        return DigitalTruthTable(
            schematic_cls,
            inputs=tuple(config.get("inputs", ())),
            outputs=tuple(config.get("outputs", ())),
            expected=expected_input,
            oracle=str(config.get("oracle", "exact")),
            dependencies=dependencies,
            rails=tuple(config.get("rails", ("vdd", "0"))),  # type: ignore[arg-type]
            complement_inputs=tuple(config.get("complement_inputs", ())),
            metadata=dict(config.get("metadata", {})),
        )


def _truth_table_mode(value: str) -> TruthTableMode:
    return resolve_digital_truth_table_mode(value)


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


def _call_view_factory(factory, **values: Any):
    signature = inspect.signature(factory)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return factory(**values)
    accepted = {
        name: values[name]
        for name in signature.parameters
        if name in values
    }
    return factory(**accepted)
