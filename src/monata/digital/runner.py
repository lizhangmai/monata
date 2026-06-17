"""Generic digital verification runner for Monata library testbenches."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import json
import re
import shutil
import time
from typing import Any, Literal

from monata.corner import OperatingCorner
from monata.models import SimulationModelConfig
from monata.netlist import SubCircuit
from monata.sim.core import LocalExecutor
from monata.digital.claims import DigitalTransientObservation
from monata.digital.circuits import SubCircuitInput
from monata.digital.library import DigitalTestbenchEntry
from monata.digital.model_context import resolve_digital_model_context
from monata.digital.recipe import DigitalSimulationRecipe
from monata.digital.results import DigitalTruthTableResult
from monata.digital.stim import DigitalStimulusConfig
from monata.digital.verification import write_digital_verification_artifacts
from monata.digital.verify import DigitalWaveformAnalyzer

TruthTableMode = Literal["transient"]
ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class DigitalRunConfig:
    model: str
    techlib: str | None
    corner: OperatingCorner | None
    vdd: float
    vdd_source: str
    model_config: SimulationModelConfig | None = None
    model_flow_identity: str | None = None

    @property
    def threshold(self) -> float:
        return self.vdd / 2.0

    @property
    def label(self) -> str:
        return self.corner_name or self.model

    @property
    def corner_name(self) -> str | None:
        return self.corner.name if self.corner is not None else None


@dataclass(frozen=True)
class DigitalRunnerOptions:
    max_workers: int
    cell_workers: int | None = None
    digital_cycles_per_vector: int | None = None
    digital_slots_per_task: int | None = None


@dataclass(frozen=True)
class DigitalMatrixJob:
    index: int
    total: int
    entry: DigitalTestbenchEntry
    run_config: DigitalRunConfig
    mode: TruthTableMode
    artifact_dir: Path
    task_workers: int


@dataclass(frozen=True)
class DigitalMatrixJobResult:
    index: int
    payload: dict[str, object]
    summary: str


def run_digital_matrix(
    entries: tuple[DigitalTestbenchEntry, ...],
    run_configs: tuple[DigitalRunConfig, ...],
    *,
    options: DigitalRunnerOptions,
    output_dir: Path,
    selection: Mapping[str, object],
    event_writer: Any = None,
    render_progress: bool = True,
    progress_factory: Callable[[DigitalMatrixJob], ProgressCallback | None] | None = None,
    manifest_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    started_at = time.monotonic()
    total = len(entries) * len(run_configs)
    jobs = tuple(digital_matrix_jobs(entries, run_configs, output_dir))
    matrix_entries: list[dict[str, object] | None] = [None] * total
    for job in jobs:
        result = run_digital_matrix_job(
            job,
            options=options,
            output_dir=output_dir,
            event_writer=event_writer,
            render_progress=render_progress,
            progress_factory=progress_factory,
        )
        matrix_entries[result.index - 1] = result.payload
    ordered_entries = [entry for entry in matrix_entries if entry is not None]
    failed_entries = sum(1 for entry in ordered_entries if entry["status"] == "FAIL")
    error_entries = sum(1 for entry in ordered_entries if entry["status"] == "ERROR")
    payload: dict[str, object] = {
        "schema_version": 8,
        "generated_at": timestamp(),
        "selection": dict(selection),
        "execution": digital_execution_payload(options, total),
        "elapsed_seconds": elapsed_seconds(started_at),
        "total_entries": len(ordered_entries),
        "passed_entries": sum(1 for entry in ordered_entries if entry["status"] == "PASS"),
        "failed_entries": failed_entries,
        "error_entries": error_entries,
        "entries": ordered_entries,
    }
    if manifest_summary is not None:
        payload["simulation_manifest"] = dict(manifest_summary)
    return payload


def run_digital_matrix_job(
    job: DigitalMatrixJob,
    *,
    options: DigitalRunnerOptions,
    output_dir: Path,
    event_writer: Any = None,
    render_progress: bool = True,
    progress_factory: Callable[[DigitalMatrixJob], ProgressCallback | None] | None = None,
) -> DigitalMatrixJobResult:
    emit_digital_event(
        event_writer, "start", job, task_workers=job.task_workers,
    )
    started_at = time.monotonic()
    progress = progress_factory(job) if progress_factory is not None else None
    try:
        reset_entry_artifact_dir(job.artifact_dir)
        result = run_digital_entry(
            job.entry,
            job.run_config,
            job.mode,
            options=options,
            max_workers=job.task_workers,
            artifact_dir=job.artifact_dir,
            progress=progress,
        )
        elapsed = elapsed_seconds(started_at)
        payload = digital_result_payload(
            job.entry, job.run_config, job.mode, result, elapsed, artifact_dir=job.artifact_dir,
        )
        detail_paths = write_digital_result_artifacts(
            output_dir, job.entry, job.run_config, payload, result,
        )
        matrix_payload = digital_matrix_entry_payload(payload, detail_paths)
        status = str(payload["status"])
        tpd = payload.get("max_propagation_delay")
        tpd_str = f"max_tpd={tpd:.6g}s" if isinstance(tpd, (int, float)) else "max_tpd=N/A"
        summary = (
            f"[{job.index}/{job.total}] {job.entry.spec.dut} "
            f"{job.run_config.corner_name or job.run_config.model} {status}: "
            f"rows={payload['rows']} failed={payload['failed_rows']} "
            f"{tpd_str} elapsed={elapsed:.2f}s"
        )
        emit_digital_event(event_writer, "done", job, status=status)
    except Exception as exc:
        elapsed = elapsed_seconds(started_at)
        matrix_payload = digital_error_payload(
            job.entry, job.run_config, job.mode, exc, elapsed, artifact_dir=job.artifact_dir,
        )
        summary = (
            f"[{job.index}/{job.total}] {job.entry.spec.dut} "
            f"{job.run_config.corner_name or job.run_config.model} ERROR: {type(exc).__name__}: {exc}"
        )
        emit_digital_event(
            event_writer, "done", job, status="ERROR", error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        close = getattr(progress, "close", None)
        if callable(close):
            close()
    return DigitalMatrixJobResult(index=job.index, payload=matrix_payload, summary=summary)


def run_digital_entry(
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
    mode: TruthTableMode,
    *,
    options: DigitalRunnerOptions,
    max_workers: int,
    artifact_dir: Path,
    progress: ProgressCallback | None = None,
) -> DigitalTruthTableResult:
    resolved_workers = max_workers if max_workers > 1 else options.max_workers
    executor = LocalExecutor(max_workers=resolved_workers)
    try:
        simulation_recipe = digital_simulation_recipe_for_entry(entry)
        observation = digital_observation_for_recipe(simulation_recipe, options)
        resolved_recipe = simulation_recipe.resolve(
            library=entry.cell.library,
            run_config=run_config,
            observation=observation,
        )
        measurements = entry.spec.measurements
        resolved_slots = resolved_recipe.observation.slots_per_task
        if resolved_slots is None and options.digital_slots_per_task is None:
            resolved_slots = auto_slots_per_task(
                2 ** len(entry.spec.inputs),
                max_workers=resolved_workers,
            )
        resolved_settle = resolved_recipe.observation.stop
        if resolved_settle is None:
            resolved_settle = estimate_settle_time(entry.spec.dependencies)
        stimulus = digital_stimulus_for_entry(entry, resolved_recipe)
        sim_results = stimulus.run(
            executor=executor,
            measurements=measurements,
            initial_settle=float(resolved_settle),
            uic=resolved_recipe.observation.uic,
            clock_period=resolved_recipe.observation.clock_period,
            slots_per_task=resolved_slots,
            progress=progress,
        )
        analyzer = DigitalWaveformAnalyzer(entry.spec)
        try:
            result = analyzer.verify(
                sim_results,
                measurements=measurements,
                vdd=stimulus.vdd,
                sample_fraction=stimulus.sample_fraction,
            )
        except RuntimeError:
            result = analyzer.verify(
                sim_results,
                measurements=("truth_table",),
                vdd=stimulus.vdd,
                sample_fraction=stimulus.sample_fraction,
            )
        write_digital_verification_artifacts(
            artifact_dir,
            table=stimulus,
            analysis=mode,
            result=result,
        )
        return result
    finally:
        executor.shutdown(wait=True)


def digital_result_payload(
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
    mode: TruthTableMode,
    result: DigitalTruthTableResult,
    elapsed: float,
    artifact_dir: Path,
) -> dict[str, object]:
    failed_rows = len(result.failed)
    status = "PASS" if failed_rows == 0 else "FAIL"
    payload: dict[str, object] = {
        **digital_plan_entry_payload_from_mode(entry, run_config, mode),
        "status": status,
        "rows": len(result),
        "failed_rows": failed_rows,
        "elapsed_seconds": elapsed,
        "artifact_dir": str(artifact_dir),
        "claim": result.claim.summary() if result.claim is not None else entry.spec.claim_summary,
        "results": result.as_dicts(),
    }
    if status == "PASS" and requires_max_propagation_delay(entry):
        max_delay = result.max_propagation_delay
        if max_delay is not None:
            worst_arc = result.max_propagation_delay_arc
            payload["max_propagation_delay"] = max_delay
            payload["max_propagation_delay_arc"] = None if worst_arc is None else worst_arc.as_dict()
    return payload


def digital_error_payload(
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
    mode: TruthTableMode,
    exc: Exception,
    elapsed: float,
    artifact_dir: Path,
) -> dict[str, object]:
    return {
        **digital_plan_entry_payload_from_mode(entry, run_config, mode),
        "status": "ERROR",
        "rows": 0,
        "failed_rows": 0,
        "elapsed_seconds": elapsed,
        "artifact_dir": str(artifact_dir),
        "error": f"{type(exc).__name__}: {exc}",
    }


def digital_plan_entry_payload(
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
    options: DigitalRunnerOptions,
) -> dict[str, object]:
    mode: TruthTableMode = "transient"
    return {
        "testbench_cell": entry.testbench_cell,
        "dut": entry.spec.dut,
        "row_count": entry.spec.row_count,
        "model": run_config.model,
        "techlib": run_config.techlib,
        "corner": run_config.corner_name,
        "vdd": run_config.vdd,
        "vdd_source": run_config.vdd_source,
        "mode": mode,
        "verification_view": "verification",
        "simulation_view": "simulation",
        "model_profile": selected_model_profile_for_entry(entry, run_config),
        "effective_observation": effective_observation_payload(entry, run_config, options),
        "measures": list(entry.spec.measurements),
        "requires_max_propagation_delay": requires_max_propagation_delay(entry),
    }


def digital_plan_entry_payload_from_mode(
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
    mode: TruthTableMode,
) -> dict[str, object]:
    return {
        "testbench_cell": entry.testbench_cell,
        "dut": entry.spec.dut,
        "oracle": entry.spec.oracle,
        "model": run_config.model,
        "techlib": run_config.techlib,
        "corner": run_config.corner_name,
        "vdd": run_config.vdd,
        "vdd_source": run_config.vdd_source,
        "mode": mode,
        "model_flow_identity": run_config.model_flow_identity,
        "measures": list(entry.spec.measurements),
        "requires_max_propagation_delay": requires_max_propagation_delay(entry),
    }


def write_digital_result_artifacts(
    output_dir: Path,
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
    payload: dict[str, object],
    result: DigitalTruthTableResult,
) -> dict[str, object]:
    stem = result_stem(entry, run_config)
    json_path = output_dir / f"{stem}.json"
    text_path = output_dir / f"{stem}.txt"
    write_json(json_path, payload)
    text_path.write_text(
        format_result_rows(result, require_delay=requires_max_propagation_delay(entry)),
        encoding="utf-8",
    )
    return {"detail_json": str(json_path), "detail_text": str(text_path)}


def dry_run_payload(
    entries: tuple[DigitalTestbenchEntry, ...],
    run_configs: tuple[DigitalRunConfig, ...],
    *,
    options: DigitalRunnerOptions,
    selection: Mapping[str, object],
    manifest_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    pairs = tuple(digital_matrix_pairs(entries, run_configs))
    payload: dict[str, object] = {
        "schema_version": 8,
        "generated_at": timestamp(),
        "dry_run": True,
        "selection": dict(selection),
        "execution": digital_execution_payload(options, len(pairs)),
        "entry_count": len(pairs),
        "views": [
            digital_plan_entry_payload(entry, run_config, options)
            for entry, run_config in pairs
        ],
    }
    if manifest_summary is not None:
        payload["simulation_manifest"] = dict(manifest_summary)
    return payload


def digital_matrix_jobs(
    entries: tuple[DigitalTestbenchEntry, ...],
    run_configs: tuple[DigitalRunConfig, ...],
    output_dir: Path,
) -> Iterable[DigitalMatrixJob]:
    pairs = tuple(digital_matrix_pairs(entries, run_configs))
    total = len(pairs)
    for index, (entry, run_config) in enumerate(pairs, start=1):
        yield DigitalMatrixJob(
            index=index,
            total=total,
            entry=entry,
            run_config=run_config,
            mode="transient",
            artifact_dir=result_artifact_dir(output_dir, entry, run_config),
            task_workers=1,
        )


def digital_matrix_pairs(
    entries: tuple[DigitalTestbenchEntry, ...],
    run_configs: tuple[DigitalRunConfig, ...],
) -> Iterable[tuple[DigitalTestbenchEntry, DigitalRunConfig]]:
    for entry in entries:
        for run_config in run_configs:
            yield entry, run_config


def digital_observation_for_recipe(
    recipe: DigitalSimulationRecipe,
    options: DigitalRunnerOptions,
) -> DigitalTransientObservation:
    return DigitalTransientObservation.resolve(
        recipe.observation,
        cycles_per_vector=options.digital_cycles_per_vector,
        slots_per_task=options.digital_slots_per_task,
    )


def digital_simulation_recipe_for_entry(entry: DigitalTestbenchEntry) -> DigitalSimulationRecipe:
    if "simulation" not in entry.cell:
        raise RuntimeError(f"{entry.testbench_cell} is missing required 'simulation' view")
    recipe = entry.cell["simulation"].load()
    if not isinstance(recipe, DigitalSimulationRecipe):
        raise TypeError(f"{entry.testbench_cell} simulation view must load a DigitalSimulationRecipe")
    return recipe


def digital_stimulus_for_entry(
    entry: DigitalTestbenchEntry,
    resolved_recipe,
) -> DigitalStimulusConfig:
    spec = entry.spec
    run_config = resolved_recipe.run_config
    builder_kwargs = resolved_recipe.builder_kwargs
    projection_library = builder_kwargs.get("projection_library")
    return DigitalStimulusConfig(
        dut=schematic_subcircuit_for_entry(entry, spec.dut),
        inputs=spec.inputs,
        outputs=spec.outputs,
        complement_inputs=spec.complement_inputs,
        dependencies=tuple(
            schematic_subcircuit_for_entry(entry, dependency)
            for dependency in spec.dependencies
        ),
        rails=spec.rails,
        vdd=float(getattr(run_config, "vdd", 1.0)),
        threshold=getattr(run_config, "threshold", None) or float(getattr(run_config, "vdd", 1.0)) / 2.0,
        period=builder_kwargs.get("period", 1e-9),
        step=builder_kwargs.get("step"),
        transition=builder_kwargs.get("transition", 0.0),
        skew_step=builder_kwargs.get("skew_step", 0.0),
        load_cap=builder_kwargs.get("load_cap"),
        setup=builder_kwargs.get("setup"),
        model_context=resolve_digital_model_context(
            projection_library=projection_library,
            corner=getattr(run_config, "corner", None),
            model_config=getattr(run_config, "model_config", None),
        ),
        backend_options=builder_kwargs.get("backend_options"),
        artifacts=builder_kwargs.get("artifacts"),
        metadata={**builder_kwargs.get("metadata", {}), "simulation_analysis": "transient"},
    )


def schematic_subcircuit_for_entry(
    entry: DigitalTestbenchEntry,
    cell_name: str,
) -> SubCircuitInput:
    view = entry.cell.library[cell_name]["schematic"]
    to_circuit = getattr(view, "to_circuit", None)
    if not callable(to_circuit):
        raise TypeError(f"{entry.testbench_cell}: schematic view for {cell_name!r} is not convertible")
    circuit = to_circuit()
    if isinstance(circuit, type) and issubclass(circuit, SubCircuit):
        return circuit
    if isinstance(circuit, SubCircuit):
        return circuit
    raise TypeError(f"{entry.testbench_cell}: schematic view for {cell_name!r} must resolve to a SubCircuit")


def selected_model_profile_for_entry(
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
) -> str:
    profile_name, _profile = digital_simulation_recipe_for_entry(entry).select_profile(run_config)
    return profile_name


def effective_observation_payload(
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
    options: DigitalRunnerOptions,
) -> dict[str, object]:
    del run_config
    return digital_observation_for_recipe(
        digital_simulation_recipe_for_entry(entry), options,
    ).as_dict()


def auto_slots_per_task(
    vectors: int,
    *,
    max_workers: int,
    min_slots: int = 16,
    max_chunks: int = 512,
) -> int:
    target = max(1, min(max_workers, max_chunks))
    raw = max(1, (vectors - 1) // target)
    return max(min_slots, raw)


def estimate_settle_time(dependencies: tuple[str, ...], *, base: float = 5e-8) -> float:
    depth = 1
    for dep in dependencies:
        lower = dep.lower()
        if any(g in lower for g in ("nand", "nor", "xor", "xnor", "aoi", "oai")):
            depth = max(depth, 2)
        elif any(g in lower for g in ("tg", "mux", "latch", "dff")):
            depth = max(depth, 3)
        elif any(g in lower for g in ("adder", "mul", "multiplier")):
            depth = max(depth, 4)
    return max(base, depth * base / 2.0)


def digital_execution_payload(options: DigitalRunnerOptions, total: int) -> dict[str, object]:
    return {"max_workers": options.max_workers, "entries": total}


def digital_matrix_entry_payload(
    payload: dict[str, object],
    detail_paths: dict[str, object],
) -> dict[str, object]:
    matrix_payload = {key: value for key, value in payload.items() if key != "results"}
    matrix_payload.update(detail_paths)
    return matrix_payload


def result_artifact_dir(
    output_dir: Path,
    entry: DigitalTestbenchEntry,
    run_config: DigitalRunConfig,
) -> Path:
    return output_dir / "artifacts" / result_stem(entry, run_config)


def reset_entry_artifact_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def result_stem(entry: DigitalTestbenchEntry, run_config: DigitalRunConfig) -> str:
    return safe_stem("_".join(filter(None, (entry.spec.dut, run_config.label))))


def requires_max_propagation_delay(entry: DigitalTestbenchEntry) -> bool:
    return "max_propagation_delay" in entry.spec.measurements


def format_result_rows(result: DigitalTruthTableResult, *, require_delay: bool) -> str:
    lines = []
    if require_delay:
        if not result.failed and result.max_propagation_delay is not None:
            max_delay = result.max_propagation_delay
            lines.append(f"max_propagation_delay {max_delay:.12g}")
        else:
            lines.append("max_propagation_delay unavailable — truth table did not pass")
    lines.append("inputs actual expected status")
    for row in result:
        expected = "-" if row.expected is None else "".join(str(bit) for bit in row.expected)
        lines.append(
            " ".join(
                (
                    "".join(str(bit) for bit in row.inputs),
                    "".join(str(bit) for bit in row.actual),
                    expected,
                    "PASS" if row.passed else "FAIL",
                )
            )
        )
    return "\n".join(lines) + "\n"


def emit_digital_event(
    writer: Any,
    event: Literal["start", "done"],
    job: DigitalMatrixJob,
    **fields: object,
) -> None:
    if writer is None:
        return
    payload: dict[str, object] = {
        "event": event,
        "timestamp": timestamp(),
        "index": job.index,
        "total": job.total,
        "testbench_cell": job.entry.testbench_cell,
        "dut": job.entry.spec.dut,
        "model": job.run_config.model,
        "corner": job.run_config.corner_name,
        "mode": job.mode,
    }
    payload.update(fields)
    writer.write(payload)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds(started_at: float) -> float:
    return round(max(time.monotonic() - started_at, 0.0), 6)


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
