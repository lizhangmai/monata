"""Internal digital truth-table planning primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from monata.sim._digital_bits import bit_combinations
from monata.sim.digital_timing import (
    _DIGITAL_SINGLE_BIT_ARC_CHUNK_SIZE,
    _PROPAGATION_DELAY_CHUNK_SIZE,
    _PROPAGATION_DELAY_TRIGGER_FRACTION,
    DigitalPropagationDelayArc,
    _order_propagation_delay_arcs,
)
from monata.sim.results import SimResult

if TYPE_CHECKING:
    from monata.sim.digital_table import DigitalTruthTable


DIGITAL_TASK_METADATA_SCHEMA = "monata.sim.digital-task.v1"
MONATA_METADATA_KEY = "monata"
DIGITAL_TASK_METADATA_KEY = "digital_task_v1"


@dataclass(frozen=True)
class DigitalSequenceSchedule:
    """Scheduled digital sequence recovered from task/result metadata."""

    arcs: tuple[DigitalPropagationDelayArc, ...]
    initial_settle: float
    slot_duration: float


@dataclass(frozen=True)
class DigitalStimulusMetadata:
    """Typed view of digital stimulus metadata stored on SimTask/SimResult."""

    kind: str
    arc_start: int
    arcs: int
    initial_settle: float
    cycles_per_vector: int
    slot_duration: float
    chunk_index: int

    def require_kind(self, expected: str, *, dut_name: str) -> None:
        if self.kind != expected:
            raise RuntimeError(
                f"{dut_name} expected digital stimulus metadata kind {expected!r}, got {self.kind!r}"
            )


@dataclass
class DigitalTruthTablePlan:
    """Owns digital arc planning, timing schedules, and task metadata."""

    table: DigitalTruthTable
    _single_bit_arc_cache: dict[tuple[bool, bool], tuple[DigitalPropagationDelayArc, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def combinations(self) -> tuple[tuple[int, ...], ...]:
        return bit_combinations(len(self.table.inputs))

    @property
    def single_bit_arc_count(self) -> int:
        return len(self.combinations()) * len(self.table.inputs)

    def single_bit_arcs(
        self,
        *,
        require_expected: bool = True,
        require_transition: bool = True,
    ) -> tuple[DigitalPropagationDelayArc, ...]:
        cache_key = (require_expected, require_transition)
        cached = self._single_bit_arc_cache.get(cache_key)
        if cached is not None:
            return cached

        table = self.table
        if require_expected and table.expected is None:
            raise RuntimeError(f"{table.dut_name} propagation delay requires an expected output table")
        if require_transition and table.transition <= 0:
            raise RuntimeError(f"{table.dut_name} propagation delay requires a positive input transition")

        arcs: list[DigitalPropagationDelayArc] = []
        for from_inputs in self.combinations():
            from_outputs = table.expected_for(from_inputs) if table.expected is not None else ()
            if from_outputs is None:
                raise RuntimeError(f"{table.dut_name} propagation delay requires expected outputs")
            for input_index in range(len(table.inputs)):
                to_inputs = tuple(
                    1 - bit if index == input_index else bit
                    for index, bit in enumerate(from_inputs)
                )
                to_outputs = table.expected_for(to_inputs) if table.expected is not None else ()
                if to_outputs is None:
                    raise RuntimeError(f"{table.dut_name} propagation delay requires expected outputs")
                output_indices = (
                    tuple(
                        index
                        for index, (before, after) in enumerate(zip(from_outputs, to_outputs))
                        if before != after
                    )
                    if table.expected is not None
                    else ()
                )
                arcs.append(
                    DigitalPropagationDelayArc(
                        index=len(arcs),
                        from_inputs=from_inputs,
                        to_inputs=to_inputs,
                        from_outputs=from_outputs,
                        to_outputs=to_outputs,
                        input_index=input_index,
                        output_indices=output_indices,
                        start=0.0,
                        reset_end=0.0,
                        trigger_start=0.0,
                        trigger_end=0.0,
                        stop=0.0,
                    )
                )

        ordered_arcs = _order_propagation_delay_arcs(arcs)
        result = tuple(
            DigitalPropagationDelayArc(
                index=index,
                from_inputs=arc.from_inputs,
                to_inputs=arc.to_inputs,
                from_outputs=arc.from_outputs,
                to_outputs=arc.to_outputs,
                input_index=arc.input_index,
                output_indices=arc.output_indices,
                start=0.0,
                reset_end=0.0,
                trigger_start=0.0,
                trigger_end=0.0,
                stop=0.0,
            )
            for index, arc in enumerate(ordered_arcs)
        )
        self._single_bit_arc_cache[cache_key] = result
        return result

    def measurable_single_bit_arc_count(
        self,
        *,
        require_expected: bool = True,
        require_transition: bool = True,
    ) -> int:
        if self.table.expected is None and not require_expected:
            return 0
        return sum(
            1
            for arc in self.single_bit_arcs(
                require_expected=require_expected,
                require_transition=require_transition,
            )
            if arc.output_indices
        )

    def propagation_delay_arcs(self) -> tuple[DigitalPropagationDelayArc, ...]:
        table = self.table
        arcs = tuple(arc for arc in self.single_bit_arcs() if arc.output_indices)
        if not arcs:
            raise RuntimeError(
                f"{table.dut_name} propagation delay has no output-changing single-input transitions"
            )
        return arcs

    def single_bit_arc_sequence_chunks(
        self,
        *,
        chunk_size: int | None = None,
        require_expected: bool = True,
        require_transition: bool = True,
    ) -> tuple[tuple[DigitalPropagationDelayArc, ...], ...]:
        resolved_chunk_size = _resolve_chunk_size(
            chunk_size,
            default=_DIGITAL_SINGLE_BIT_ARC_CHUNK_SIZE,
            label="single-bit arc",
        )
        arcs = self.single_bit_arcs(
            require_expected=require_expected,
            require_transition=require_transition,
        )
        return tuple(
            arcs[index : index + resolved_chunk_size]
            for index in range(0, len(arcs), resolved_chunk_size)
        )

    def propagation_delay_arc_chunks(
        self,
        *,
        chunk_size: int | None = None,
    ) -> tuple[tuple[DigitalPropagationDelayArc, ...], ...]:
        resolved_chunk_size = _resolve_chunk_size(
            chunk_size,
            default=_PROPAGATION_DELAY_CHUNK_SIZE,
            label="propagation delay",
        )
        arcs = self.propagation_delay_arcs()
        chunks = [
            self.scheduled_propagation_delay_arcs(arcs[index : index + resolved_chunk_size])
            for index in range(0, len(arcs), resolved_chunk_size)
        ]
        return tuple(chunks)

    def logic_value(self, value: Any) -> int:
        return int(float(value) > self.table.threshold)

    def task_metadata(
        self,
        task_kind: str,
        *,
        measurements: Iterable[str],
        stimulus: Mapping[str, object] | None = None,
        coverage: Mapping[str, object] | None = None,
        transient: Mapping[str, object] | None = None,
    ) -> dict:
        metadata = dict(self.table.metadata)
        payload = self.digital_task_payload(task_kind, measurements=measurements)
        if stimulus is not None:
            payload["stimulus"] = dict(stimulus)
        if coverage is not None:
            payload["coverage"] = dict(coverage)
        if transient is not None:
            payload["transient"] = dict(transient)
        metadata[MONATA_METADATA_KEY] = _monata_metadata_with_digital_task(
            metadata.get(MONATA_METADATA_KEY),
            payload,
        )
        return metadata

    def digital_task_payload(self, task_kind: str, *, measurements: Iterable[str]) -> dict:
        """Return the versioned digital control payload for tests or custom task assembly."""

        table = self.table
        return {
            "schema": DIGITAL_TASK_METADATA_SCHEMA,
            "measurements": list(measurements),
            "digital_truth_table": {
                "task_kind": task_kind,
                "dut": table.dut_name,
                "inputs": list(table.inputs),
                "outputs": list(table.outputs),
                "vectors": len(self.combinations()),
                "vdd": table.vdd,
                "threshold": table.threshold,
            },
        }

    def digital_sequence_slot_duration(self, cycles_per_vector: int) -> float:
        return self.table.period * int(cycles_per_vector)

    def scheduled_single_bit_sequence_arcs(
        self,
        arcs: Iterable[DigitalPropagationDelayArc],
        *,
        initial_settle: float,
        cycles_per_vector: int,
    ) -> tuple[DigitalPropagationDelayArc, ...]:
        table = self.table
        source_arcs = tuple(arcs)
        if not source_arcs:
            raise RuntimeError(f"{table.dut_name} digital sequence requires at least one arc")
        slot_duration = self.digital_sequence_slot_duration(cycles_per_vector)
        if table.transition + self._max_input_skew() >= slot_duration:
            raise RuntimeError(
                f"{table.dut_name} digital sequence period is too short for transition timing: "
                f"slot_duration={slot_duration}, transition={table.transition}, skew_step={table.skew_step}"
            )
        scheduled = []
        for local_index, arc in enumerate(source_arcs):
            start = initial_settle + local_index * slot_duration
            trigger_start = (
                initial_settle
                + (local_index + 1) * slot_duration
                + self._input_skew(arc.input_index)
            )
            trigger_end = trigger_start + max(table.transition, 0.0)
            stop = initial_settle + (local_index + 2) * slot_duration
            scheduled.append(
                DigitalPropagationDelayArc(
                    index=arc.index,
                    from_inputs=arc.from_inputs,
                    to_inputs=arc.to_inputs,
                    from_outputs=arc.from_outputs,
                    to_outputs=arc.to_outputs,
                    input_index=arc.input_index,
                    output_indices=arc.output_indices,
                    start=start,
                    reset_end=trigger_start,
                    trigger_start=trigger_start,
                    trigger_end=trigger_end,
                    stop=stop,
                )
            )
        return tuple(scheduled)

    def sequence_for_result(self, sim_result: SimResult) -> DigitalSequenceSchedule:
        table = self.table
        task_metadata = sim_result_digital_task_metadata(sim_result)
        stimulus = digital_stimulus_metadata(
            sim_result,
            default_cycles_per_vector=table.cycles_per_vector,
            default_slot_duration=table.period,
        )
        stimulus.require_kind("digital_single_bit_arc_sequence", dut_name=table.dut_name)
        measurements = task_metadata.get("measurements", ())
        requested_delay = (
            isinstance(measurements, (list, tuple))
            and "max_propagation_delay" in measurements
        )
        if stimulus.arcs < 1:
            raise RuntimeError(f"{table.dut_name} digital sequence metadata contains no arcs")
        arcs = self.single_bit_arcs(
            require_expected=requested_delay,
            require_transition=requested_delay,
        )[stimulus.arc_start : stimulus.arc_start + stimulus.arcs]
        if len(arcs) != stimulus.arcs:
            raise RuntimeError(
                f"{table.dut_name} digital sequence metadata exceeds available arcs: "
                f"arc_start={stimulus.arc_start}, arcs={stimulus.arcs}, total={self.single_bit_arc_count}"
            )
        slot_duration = self.digital_sequence_slot_duration(stimulus.cycles_per_vector)
        if stimulus.slot_duration and not _metadata_float_close(stimulus.slot_duration, slot_duration):
            raise RuntimeError(
                f"{table.dut_name} digital sequence metadata has inconsistent slot_duration: "
                f"metadata={stimulus.slot_duration}, cycles_per_vector={stimulus.cycles_per_vector}, "
                f"expected={slot_duration}"
            )
        return DigitalSequenceSchedule(
            arcs=self.scheduled_single_bit_sequence_arcs(
                arcs,
                initial_settle=stimulus.initial_settle,
                cycles_per_vector=stimulus.cycles_per_vector,
            ),
            initial_settle=stimulus.initial_settle,
            slot_duration=slot_duration,
        )

    def propagation_delay_coverage(self, measured_rows: int) -> dict[str, object]:
        return {
            "kind": "directed_single_bit_exhaustive",
            "input_count": len(self.table.inputs),
            "vector_count": len(self.combinations()),
            "candidate_arc_count": self.single_bit_arc_count,
            "measurable_arc_count": self.measurable_single_bit_arc_count(),
            "measured_delay_rows": measured_rows,
        }

    def scheduled_propagation_delay_arcs(
        self,
        arcs: Iterable[DigitalPropagationDelayArc] | None = None,
    ) -> tuple[DigitalPropagationDelayArc, ...]:
        table = self.table
        source_arcs = self.propagation_delay_arcs() if arcs is None else tuple(arcs)
        if not source_arcs:
            raise RuntimeError(f"{table.dut_name} propagation delay requires at least one arc")
        return tuple(
            self._schedule_propagation_delay_arc(arc, index)
            for index, arc in enumerate(source_arcs)
        )

    def _schedule_propagation_delay_arc(
        self,
        arc: DigitalPropagationDelayArc,
        index: int,
    ) -> DigitalPropagationDelayArc:
        table = self.table
        start = index * table.period
        reset_end = start + table.transition + self._max_input_skew()
        trigger_start = (
            start
            + table.period * _PROPAGATION_DELAY_TRIGGER_FRACTION
            + self._input_skew(arc.input_index)
        )
        if trigger_start <= reset_end:
            trigger_start = reset_end + table.transition
        trigger_end = trigger_start + table.transition
        stop = start + table.period
        if trigger_end >= stop:
            raise RuntimeError(
                f"{table.dut_name} propagation delay period is too short for transition timing: "
                f"period={table.period}, transition={table.transition}"
            )
        return DigitalPropagationDelayArc(
            index=index,
            from_inputs=arc.from_inputs,
            to_inputs=arc.to_inputs,
            from_outputs=arc.from_outputs,
            to_outputs=arc.to_outputs,
            input_index=arc.input_index,
            output_indices=arc.output_indices,
            start=start,
            reset_end=reset_end,
            trigger_start=trigger_start,
            trigger_end=trigger_end,
            stop=stop,
        )

    def _input_skew(self, input_index: int) -> float:
        return max(self.table.skew_step, 0.0) * int(input_index)

    def _max_input_skew(self) -> float:
        return self._input_skew(max(len(self.table.inputs) - 1, 0))


def sim_result_digital_task_metadata(sim_result: SimResult) -> Mapping[str, Any]:
    return digital_task_metadata(sim_result.metadata)


def sim_result_has_digital_task_metadata(sim_result: SimResult) -> bool:
    namespace = sim_result.metadata.get(MONATA_METADATA_KEY)
    return isinstance(namespace, Mapping) and DIGITAL_TASK_METADATA_KEY in namespace


def digital_task_metadata(task_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    namespace = task_metadata.get(MONATA_METADATA_KEY)
    if not isinstance(namespace, Mapping):
        raise RuntimeError("digital task metadata requires metadata['monata']")
    if DIGITAL_TASK_METADATA_KEY not in namespace:
        raise RuntimeError("digital task metadata missing metadata['monata']['digital_task_v1']")
    payload = namespace[DIGITAL_TASK_METADATA_KEY]
    if not isinstance(payload, Mapping):
        raise RuntimeError("digital task metadata payload must be a mapping")
    schema = payload.get("schema")
    if schema != DIGITAL_TASK_METADATA_SCHEMA:
        raise RuntimeError(f"unsupported digital task metadata schema: {schema}")
    return payload


def digital_stimulus_metadata(
    sim_result: SimResult,
    *,
    default_cycles_per_vector: int = 1,
    default_slot_duration: float | None = None,
) -> DigitalStimulusMetadata:
    task_metadata = sim_result_digital_task_metadata(sim_result)
    raw_stimulus = task_metadata.get("stimulus")
    if not isinstance(raw_stimulus, Mapping):
        raise RuntimeError("digital task metadata missing stimulus")
    stimulus = cast(Mapping[str, object], raw_stimulus)
    return DigitalStimulusMetadata(
        kind=str(stimulus.get("kind", "")),
        arc_start=_metadata_int(stimulus, "arc_start", 0),
        arcs=_metadata_int(stimulus, "arcs", 0),
        initial_settle=_metadata_float(stimulus, "initial_settle", 0.0),
        cycles_per_vector=_metadata_int(stimulus, "cycles_per_vector", int(default_cycles_per_vector)),
        slot_duration=_metadata_float(stimulus, "slot_duration", float(default_slot_duration or 0.0)),
        chunk_index=_metadata_int(stimulus, "chunk_index", 0),
    )


def _resolve_chunk_size(chunk_size: int | None, *, default: int, label: str) -> int:
    resolved = default if chunk_size is None else int(chunk_size)
    if resolved < 1:
        raise ValueError(f"{label} chunk_size must be positive")
    return resolved


def _metadata_int(metadata: Mapping[str, object], name: str, default: int) -> int:
    try:
        return int(cast(float | int | str, metadata.get(name, default)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"digital stimulus metadata field {name!r} must be an integer") from exc


def _metadata_float(metadata: Mapping[str, object], name: str, default: float) -> float:
    try:
        return float(cast(float | int | str, metadata.get(name, default)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"digital stimulus metadata field {name!r} must be numeric") from exc


def _metadata_float_close(value: float, expected: float) -> bool:
    return abs(value - expected) <= max(abs(expected), 1.0) * 1e-12


def _monata_metadata_with_digital_task(value: object, payload: Mapping[str, object]) -> dict[str, object]:
    if value is None:
        namespace: dict[str, object] = {}
    elif isinstance(value, Mapping):
        namespace = dict(value)
    else:
        raise ValueError("metadata['monata'] is reserved for Monata namespace mappings")
    namespace[DIGITAL_TASK_METADATA_KEY] = dict(payload)
    return namespace
