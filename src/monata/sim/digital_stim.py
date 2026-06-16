"""Digital clock-driven stimulus description and task construction.

This module owns the *simulation* side of the digital pipeline: it
describes a clocked Gray-code stimulus, builds the corresponding
SPICE circuits, and produces ``SimTask`` objects ready for execution.
It has no knowledge of truth tables, expected outputs, or verification
— those concerns live in ``digital_verify.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monata.sim._digital_bits import bits_to_text, gray_code_chunks, gray_code_sequence
from monata.sim.analysis_spec import TranSpec
from monata.sim.digital_circuits import (
    DigitalTruthTableCircuitBuilder,
    SubCircuitInput,
    _subckt_name,
)
from monata.sim.results import SimResult
from monata.sim.task import SimArtifactOptions, SimTask


@dataclass(frozen=True)
class DigitalStimulusConfig:
    """Pure simulation description of a clock-driven digital stimulus.

    A stimulus knows what signals to drive, at what voltage and timing,
    and how to split the state sequence into parallel-simulatable chunks.
    It does **not** know what the correct outputs should be — that is the
    verifier's responsibility.
    """

    dut: SubCircuitInput
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    complement_inputs: tuple[str, ...] = ()
    dependencies: tuple[SubCircuitInput, ...] = ()
    rails: tuple[str, str] = ("vdd", "0")
    vdd: float = 1.0
    threshold: float = 0.5
    period: float = 1e-9
    step: float | None = None
    transition: float = 0.0
    skew_step: float = 0.0
    sample_fraction: float = 0.9
    load_cap: str | float | None = None
    setup: Any = None
    library: Any = None
    corner: Any = None
    model_config: Any = None
    backend_options: Mapping[str, Any] | None = None
    artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_spec_and_recipe(
        cls,
        spec: Any,
        recipe: Any,  # kept for API symmetry; resolved_recipe provides builder_kwargs
        *,
        run_config: Any,
        library: Any,
        resolved_recipe: Any,
    ) -> "DigitalStimulusConfig":
        """Build a stimulus config from verification spec + simulation recipe.

        The *spec* provides DUT identity, pin lists, and dependencies.
        The *recipe* provides timing, loading, and model configuration.
        Neither object is modified; the stimulus config is a pure
        simulation view of their intersection.
        """
        from monata.views.declarative import schematic_view_to_subcircuit

        def _load_schematic(cell_name: str) -> SubCircuitInput:
            return schematic_view_to_subcircuit(
                library[cell_name]["schematic"],
                reason="digital stimulus",
            )

        bk = resolved_recipe.builder_kwargs
        return cls(
            dut=_load_schematic(spec.dut),
            inputs=spec.inputs,
            outputs=spec.outputs,
            complement_inputs=spec.complement_inputs,
            dependencies=tuple(_load_schematic(name) for name in spec.dependencies),
            rails=spec.rails,
            vdd=float(getattr(run_config, "vdd", 1.0)),
            threshold=getattr(run_config, "threshold", None) or float(getattr(run_config, "vdd", 1.0)) / 2.0,
            period=bk.get("period", 1e-9),
            step=bk.get("step"),
            transition=bk.get("transition", 0.0),
            skew_step=bk.get("skew_step", 0.0),
            load_cap=bk.get("load_cap"),
            setup=bk.get("setup"),
            library=bk.get("projection_library"),
            corner=getattr(run_config, "corner", None),
            model_config=getattr(run_config, "model_config", None),
            backend_options=bk.get("backend_options"),
            artifacts=bk.get("artifacts"),
            metadata={**bk.get("metadata", {}), "simulation_analysis": "transient"},
        )

    @property
    def dut_name(self) -> str:
        return _subckt_name(self.dut)

    @property
    def model_flow(self):
        """Lazy-resolved model flow for OSDI path discovery."""
        if self.library is None or self.corner is None or self.model_config is None:
            return None
        try:
            from monata.sim.digital_table_config import _resolve_model_flow
            return _resolve_model_flow(self.library, self.corner, self.model_config)
        except Exception:
            return None

    def clocked_sequence_circuit(
        self, states, *, initial_settle, clock_period,
    ):
        """Build a clocked circuit for this stimulus (test / debug helper)."""
        builder = DigitalTruthTableCircuitBuilder(self)  # type: ignore[arg-type]
        return builder.clocked_sequence_circuit(
            states, initial_settle=initial_settle, clock_period=clock_period,
        )

    def combinations(self) -> tuple[tuple[int, ...], ...]:
        return gray_code_sequence(len(self.inputs))

    def build_tasks(
        self,
        *,
        initial_settle: float = 0.0,
        uic: bool = False,
        measurements: tuple[str, ...] = ("truth_table",),
        clock_period: float | None = None,
        slots_per_task: int | None = None,
    ) -> list[SimTask]:
        """Build ``SimTask`` objects for clocked Gray-code transient simulation.

        Each task simulates one chunk of the Gray-code sequence.
        """
        resolved_clock = self.period if clock_period is None else float(clock_period)
        if resolved_clock <= 0:
            raise ValueError("clock_period must be positive")
        chunks = gray_code_chunks(
            len(self.inputs),
            slots_per_chunk=slots_per_task,
        )
        builder = DigitalTruthTableCircuitBuilder(self)  # type: ignore[arg-type]
        tasks: list[SimTask] = []
        for chunk_index, initial_state, states in chunks:
            full_states = (initial_state,) + states
            stop_time = initial_settle + len(states) * resolved_clock
            stimulus = {
                "kind": "digital_sequence",
                "clock_period": resolved_clock,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "state_sequence": [bits_to_text(state) for state in full_states],
                "initial_settle": initial_settle,
                "transition": self.transition,
                "initial_state": bits_to_text(initial_state),
            }
            coverage = {
                "truth_vector_indices": list(range(
                    chunk_index * (len(states) or 1),
                    chunk_index * (len(states) or 1) + len(states),
                )),
            }
            transient = {"stop": stop_time, "uic": uic}
            tasks.append(
                SimTask(
                    circuit=builder.clocked_sequence_circuit(
                        full_states,
                        initial_settle=initial_settle,
                        clock_period=resolved_clock,
                    ),
                    analysis_spec=TranSpec(
                        step=self.step if self.step is not None else resolved_clock / 50.0,
                        stop=stop_time,
                        uic=uic,
                    ),
                    corner=self.corner,
                    output_names=tuple((*self.inputs, "clk", *self.outputs)),
                    osdi_paths=_model_flow_osdi_paths(self),
                    backend_options=self.backend_options,
                    artifacts=self.artifacts,
                    metadata=_task_metadata(
                        self,
                        "digital-sequence",
                        measurements=measurements,
                        stimulus=stimulus,
                        coverage=coverage,
                        transient=transient,
                    ),
                )
            )
        return tasks

    def run(
        self,
        executor=None,
        *,
        initial_settle: float = 0.0,
        uic: bool = False,
        measurements: tuple[str, ...] = ("truth_table",),
        clock_period: float | None = None,
        slots_per_task: int | None = None,
        progress=None,
    ) -> list[SimResult]:
        """Convenience: build tasks, submit to executor, return results."""
        from concurrent.futures import as_completed

        tasks = self.build_tasks(
            initial_settle=initial_settle,
            uic=uic,
            measurements=measurements,
            clock_period=clock_period,
            slots_per_task=slots_per_task,
        )
        if executor is None:
            from monata.sim.executor import LocalExecutor
            executor = LocalExecutor()
        futures = executor.map(tasks)
        total = len(tasks)
        completed = 0
        if progress is not None:
            progress({"event": "tasks_start", "completed": 0, "total": total})
        results: list[SimResult] = []
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if progress is not None:
                progress({
                    "event": "task_done", "completed": completed, "total": total,
                    "status": result.status,
                })
            if result.status != "ok":
                raise RuntimeError(
                    f"{self.dut_name} transient chunk failed: {result.error_message}"
                )
            results.append(result)
        if progress is not None:
            progress({"event": "tasks_done", "completed": total, "total": total})
        return results


def _model_flow_osdi_paths(stim: DigitalStimulusConfig) -> tuple[str, ...]:
    mf = stim.model_flow
    if mf is None:
        return ()
    return tuple(getattr(mf.model_selection, "osdi_paths", ()))


def _task_metadata(
    stim: DigitalStimulusConfig,
    task_kind: str,
    *,
    measurements: Iterable[str],
    stimulus: Mapping[str, object] | None = None,
    coverage: Mapping[str, object] | None = None,
    transient: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    from monata.sim.digital_plan import MONATA_METADATA_KEY, DIGITAL_TASK_METADATA_KEY

    metadata = dict(stim.metadata or {})
    payload: dict[str, Any] = {
        "schema": "monata.sim.digital-task.v1",
        "measurements": list(measurements),
        "digital_verification": {
            "task_kind": task_kind,
            "dut": stim.dut_name,
            "inputs": list(stim.inputs),
            "outputs": list(stim.outputs),
            "vectors": len(stim.combinations()),
            "vdd": stim.vdd,
            "threshold": stim.threshold,
        },
    }
    if stimulus is not None:
        payload["stimulus"] = dict(stimulus)
    if coverage is not None:
        payload["coverage"] = dict(coverage)
    if transient is not None:
        payload["transient"] = dict(transient)
    namespace = dict(metadata.get(MONATA_METADATA_KEY, {}))
    namespace[DIGITAL_TASK_METADATA_KEY] = payload
    metadata[MONATA_METADATA_KEY] = namespace
    return metadata
