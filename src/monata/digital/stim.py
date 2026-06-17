"""Digital clock-driven stimulus description and task construction.

This module owns the *simulation* side of the digital pipeline: it
describes a clocked Gray-code stimulus, builds the corresponding
SPICE circuits, and produces ``SimTask`` objects ready for execution.
It has no knowledge of truth tables, expected outputs, or verification
— those concerns live in ``monata.digital.verify``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monata.digital.bits import bits_to_text, gray_code_chunks, gray_code_sequence
from monata.digital.circuits import (
    DigitalTruthTableCircuitBuilder,
    SubCircuitInput,
    _subckt_name,
)
from monata.digital.model_context import DigitalModelContext
from monata.sim.analysis_spec import TranSpec
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
    model_context: DigitalModelContext = DigitalModelContext()
    backend_options: Mapping[str, Any] | None = None
    artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None
    metadata: dict[str, Any] | None = None

    @property
    def dut_name(self) -> str:
        return _subckt_name(self.dut)

    @property
    def corner(self):
        return self.model_context.corner

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
                    osdi_paths=self.model_context.osdi_paths,
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


def _task_metadata(
    stim: DigitalStimulusConfig,
    task_kind: str,
    *,
    measurements: Iterable[str],
    stimulus: Mapping[str, object] | None = None,
    coverage: Mapping[str, object] | None = None,
    transient: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    from monata.digital.plan import MONATA_METADATA_KEY, DIGITAL_TASK_METADATA_KEY

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
