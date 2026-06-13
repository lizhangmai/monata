"""Internal task construction helpers for digital truth-table simulations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from monata.sim._digital_bits import bits_to_text
from monata.sim.analysis_spec import TranSpec
from monata.sim.digital_circuits import DigitalTruthTableCircuitBuilder
from monata.sim.digital_plan import DigitalTruthTablePlan
from monata.sim.digital_timing import (
    DigitalPropagationDelayArc,
    _digital_sequence_stop_time,
    _states_for_arc_sequence,
    _truth_vector_indices_for_arcs,
)
from monata.sim.task import SimTask

if TYPE_CHECKING:
    from monata.sim.digital_table import DigitalTruthTable


@dataclass(frozen=True)
class DigitalTruthTableTaskFactory:
    table: DigitalTruthTable
    plan: DigitalTruthTablePlan

    def transient_tasks(
        self,
        *,
        stop: float | None = None,
        uic: bool = False,
        measurements: tuple[str, ...],
        cycles_per_vector: int | None = None,
        slots_per_task: int | None = None,
    ) -> list[SimTask]:
        table = self.table
        plan = self.plan
        builder = DigitalTruthTableCircuitBuilder(table)
        initial_settle = 0.0 if stop is None else float(stop)
        resolved_cycles = table.cycles_per_vector if cycles_per_vector is None else int(cycles_per_vector)
        resolved_slots = table.slots_per_task if slots_per_task is None else int(slots_per_task)
        if resolved_cycles < 1:
            raise ValueError("cycles_per_vector must be positive")
        if resolved_slots is not None and resolved_slots < 1:
            raise ValueError("slots_per_task must be positive")
        requested_delay = "max_propagation_delay" in measurements
        chunks = plan.single_bit_arc_sequence_chunks(
            chunk_size=resolved_slots,
            require_expected=requested_delay,
            require_transition=requested_delay,
        )
        total_measurable_arcs = plan.measurable_single_bit_arc_count(
            require_expected=requested_delay,
            require_transition=requested_delay,
        )
        tasks = []
        for chunk_index, chunk in enumerate(chunks):
            scheduled_arcs = plan.scheduled_single_bit_sequence_arcs(
                chunk,
                initial_settle=initial_settle,
                cycles_per_vector=resolved_cycles,
            )
            states = _states_for_arc_sequence(chunk)
            slot_duration = plan.digital_sequence_slot_duration(resolved_cycles)
            stop_time = _digital_sequence_stop_time(
                scheduled_arcs,
                initial_settle=initial_settle,
                slot_duration=slot_duration,
            )
            stimulus_metadata = {
                "kind": "digital_single_bit_arc_sequence",
                "arc_coverage": "directed_single_bit_exhaustive",
                "arcs": len(chunk),
                "total_arcs": plan.single_bit_arc_count,
                "measurable_arcs": sum(1 for arc in chunk if arc.output_indices),
                "total_measurable_arcs": total_measurable_arcs,
                "states": len(chunk) + 1,
                "state_sequence": [bits_to_text(state) for state in states],
                "vectors": len(plan.combinations()),
                "cycles_per_vector": resolved_cycles,
                "slot_duration": slot_duration,
                "initial_settle": initial_settle,
                "transition": table.transition,
                "skew_step": table.skew_step,
                "arc_start": chunk[0].index,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
            }
            coverage_metadata = {
                "truth_vector_indices": _truth_vector_indices_for_arcs(chunk),
                "single_bit_arc_indices": [arc.index for arc in chunk],
                "measurable_delay_arc_indices": [
                    arc.index for arc in chunk if arc.output_indices
                ],
            }
            transient_metadata = {
                "stop": stop_time,
                "uic": uic,
            }
            tasks.append(
                SimTask(
                    circuit=builder.digital_sequence_circuit(
                        scheduled_arcs,
                        initial_settle=initial_settle,
                        slot_duration=slot_duration,
                    ),
                    analysis_spec=TranSpec(
                        step=_digital_sequence_step(table, slot_duration=slot_duration),
                        stop=stop_time,
                        uic=uic,
                    ),
                    corner=table.corner,
                    output_names=tuple((*table.inputs, *table.outputs)),
                    osdi_paths=_model_flow_osdi_paths(table),
                    backend_options=table.backend_options,
                    artifacts=table.artifacts,
                    metadata=plan.task_metadata(
                        "digital-single-bit-arc-sequence",
                        measurements=measurements,
                        stimulus=stimulus_metadata,
                        coverage=coverage_metadata,
                        transient=transient_metadata,
                    ),
                )
            )
        return tasks

    def propagation_delay_task(
        self,
        arcs: Iterable[DigitalPropagationDelayArc] | None = None,
        *,
        chunk_index: int | None = None,
        chunk_count: int | None = None,
        step: float | None = None,
        uic: bool = True,
    ) -> SimTask:
        table = self.table
        plan = self.plan
        source_arcs = plan.single_bit_arcs() if arcs is None else tuple(arcs)
        if not source_arcs:
            raise RuntimeError(f"{table.dut_name} propagation delay requires at least one arc")
        cycles = table.cycles_per_vector
        slot_duration = plan.digital_sequence_slot_duration(cycles)
        scheduled_arcs = plan.scheduled_single_bit_sequence_arcs(
            source_arcs,
            initial_settle=0.0,
            cycles_per_vector=cycles,
        )
        states = _states_for_arc_sequence(source_arcs)
        stop_time = _digital_sequence_stop_time(
            scheduled_arcs,
            initial_settle=0.0,
            slot_duration=slot_duration,
        )
        stimulus_metadata: dict[str, object] = {
            "kind": "digital_single_bit_arc_sequence",
            "arc_coverage": "directed_single_bit_exhaustive",
            "arcs": len(source_arcs),
            "total_arcs": plan.single_bit_arc_count,
            "measurable_arcs": sum(1 for arc in source_arcs if arc.output_indices),
            "total_measurable_arcs": plan.measurable_single_bit_arc_count(),
            "states": len(source_arcs) + 1,
            "state_sequence": [bits_to_text(state) for state in states],
            "vectors": len(plan.combinations()),
            "cycles_per_vector": cycles,
            "slot_duration": slot_duration,
            "initial_settle": 0.0,
            "transition": table.transition,
            "skew_step": table.skew_step,
            "arc_start": source_arcs[0].index,
        }
        if chunk_index is not None:
            stimulus_metadata["chunk_index"] = chunk_index
        if chunk_count is not None:
            stimulus_metadata["chunk_count"] = chunk_count
        return SimTask(
            circuit=DigitalTruthTableCircuitBuilder(table).digital_sequence_circuit(
                scheduled_arcs,
                initial_settle=0.0,
                slot_duration=slot_duration,
            ),
            analysis_spec=TranSpec(
                step=table.step if step is None else float(step),
                stop=stop_time,
                uic=uic,
            ),
            corner=table.corner,
            output_names=tuple((*table.inputs, *table.outputs)),
            osdi_paths=_model_flow_osdi_paths(table),
            backend_options=table.backend_options,
            artifacts=table.artifacts,
            metadata=plan.task_metadata(
                "digital-single-bit-arc-sequence",
                measurements=("max_propagation_delay",),
                stimulus=stimulus_metadata,
                coverage={
                    "truth_vector_indices": _truth_vector_indices_for_arcs(source_arcs),
                    "single_bit_arc_indices": [arc.index for arc in source_arcs],
                    "measurable_delay_arc_indices": [
                        arc.index for arc in source_arcs if arc.output_indices
                    ],
                },
                transient={
                    "stop": stop_time,
                    "uic": uic,
                },
            ),
        )


def _digital_sequence_step(table: DigitalTruthTable, *, slot_duration: float) -> float:
    if table.step is not None:
        return table.step
    if table.truth_table_step is not None:
        return table.truth_table_step
    return slot_duration / 20.0


def _model_flow_osdi_paths(table: DigitalTruthTable) -> tuple[str, ...]:
    if table.model_flow is None:
        return ()
    return tuple(table.model_flow.model_selection.osdi_paths)
