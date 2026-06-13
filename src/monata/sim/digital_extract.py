"""Internal result extraction helpers for digital truth-table simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from monata.measure.time_domain import cross
from monata.sim._digital_bits import bits_to_text
from monata.sim.digital_claims import (
    DigitalComparisonContext,
    DigitalComparisonResult,
    DigitalOutputTolerance,
)
from monata.sim.digital_plan import (
    DigitalTruthTablePlan,
    digital_stimulus_metadata,
    sim_result_has_digital_task_metadata,
)
from monata.sim.digital_results import (
    DigitalPropagationDelayRow,
    DigitalTruthTableResult,
    DigitalTruthTableRow,
)
from monata.sim.digital_timing import (
    DigitalPropagationDelayArc,
    _states_for_arc_sequence,
)
from monata.sim.results import SimResult

if TYPE_CHECKING:
    from monata.sim.digital_table import (
        DigitalMeasurementName,
        DigitalTruthTable,
        DigitalTruthTableMode,
    )


@dataclass(frozen=True)
class DigitalTruthTableResultExtractor:
    table: DigitalTruthTable
    plan: DigitalTruthTablePlan

    def transient(
        self,
        sim_results: Iterable[SimResult],
        *,
        measurements: tuple[DigitalMeasurementName, ...],
    ) -> DigitalTruthTableResult:
        return self._transient_results(
            sim_results,
            "transient",
            measurements=measurements,
        )

    def propagation_delay(
        self,
        sim_result: SimResult,
        arcs: Iterable[DigitalPropagationDelayArc] | None = None,
    ) -> tuple[DigitalPropagationDelayRow, ...]:
        table = self.table
        if sim_result.status != "ok":
            raise RuntimeError(f"{table.dut_name} propagation delay failed: {sim_result.error_message}")
        if sim_result.sweep_var is None:
            raise RuntimeError(f"{table.dut_name} propagation delay did not return a sweep vector")

        if _is_digital_sequence_result(sim_result):
            time = _digital_sequence_time(table, sim_result)
            schedule = self.plan.sequence_for_result(sim_result)
            sequence_rows = self._propagation_delay_rows_for_sequence(sim_result, schedule.arcs, time)
            if not sequence_rows:
                raise RuntimeError(f"{table.dut_name} propagation delay produced no measurable arcs")
            return sequence_rows

        time = np.asarray(sim_result.sweep_var, dtype=float).reshape(-1)
        if len(time) < 2:
            raise RuntimeError(f"{table.dut_name} propagation delay returned too few time samples")

        scheduled_arcs = self.plan.scheduled_propagation_delay_arcs(arcs)
        if not scheduled_arcs:
            raise RuntimeError(f"{table.dut_name} propagation delay produced no scheduled arcs")
        if float(time[-1]) < scheduled_arcs[-1].stop * 0.99:
            raise RuntimeError(
                f"{table.dut_name} propagation delay transient ended early: "
                f"final_time={float(time[-1])}, expected={scheduled_arcs[-1].stop}"
            )

        rows: list[DigitalPropagationDelayRow] = []
        for arc in scheduled_arcs:
            input_name = table.inputs[arc.input_index]
            input_crossing = _first_threshold_crossing(
                time,
                sim_result.waveforms[input_name],
                threshold=table.threshold,
                edge=arc.input_edge,
                start=arc.reset_end,
                stop=arc.trigger_end,
                signal_name=input_name,
            )
            for output_index in arc.output_indices:
                output_name = table.outputs[output_index]
                output_crossing = _first_threshold_crossing(
                    time,
                    sim_result.waveforms[output_name],
                    threshold=table.threshold,
                    edge=arc.output_edge(output_index),
                    start=arc.trigger_start,
                    stop=arc.stop,
                    signal_name=output_name,
                )
                rows.append(
                    DigitalPropagationDelayRow(
                        from_inputs=arc.from_inputs,
                        to_inputs=arc.to_inputs,
                        input_name=input_name,
                        output_name=output_name,
                        input_edge=arc.input_edge,
                        output_edge=arc.output_edge(output_index),
                        input_crossing=input_crossing,
                        output_crossing=output_crossing,
                        delay=output_crossing - input_crossing,
                    )
                )
        if not rows:
            raise RuntimeError(f"{table.dut_name} propagation delay produced no measurable arcs")
        return tuple(rows)

    def _transient_results(
        self,
        sim_results: Iterable[SimResult],
        mode: DigitalTruthTableMode,
        *,
        measurements: tuple[DigitalMeasurementName, ...],
    ) -> DigitalTruthTableResult:
        table = self.table
        combinations = table.combinations()
        results = list(sim_results)
        if results and all(_is_digital_sequence_result(result) for result in results):
            return self._digital_sequence_results(results, combinations, mode, measurements=measurements)
        if len(results) != len(combinations):
            raise RuntimeError(
                f"{table.dut_name} transient result count mismatch: "
                f"expected={len(combinations)}, got={len(results)}"
            )
        rows = []
        first_result = results[0]
        for bits, sim_result in zip(combinations, results):
            if sim_result.status != "ok":
                raise RuntimeError(
                    f"{table.dut_name} transient static-vector task failed for {bits_to_text(bits)}: "
                    f"{sim_result.error_message}"
                )
            samples = {}
            actual_values = []
            for output in table.outputs:
                value = float(np.asarray(sim_result.waveforms[output]).reshape(-1)[-1])
                samples[output] = value
                actual_values.append(self.plan.logic_value(value))
            actual = tuple(actual_values)
            sample_time = None
            if sim_result.sweep_var is not None and len(sim_result.sweep_var):
                sample_time = float(np.asarray(sim_result.sweep_var).reshape(-1)[-1])
            rows.append(self._row(bits, actual, samples=samples, sample_time=sample_time))

        return DigitalTruthTableResult(rows, first_result, mode, claim=table.claim, sim_results=results)

    def _digital_sequence_results(
        self,
        sim_results: Iterable[SimResult],
        combinations: tuple[tuple[int, ...], ...],
        mode: DigitalTruthTableMode,
        *,
        measurements: tuple[DigitalMeasurementName, ...],
    ) -> DigitalTruthTableResult:
        table = self.table
        results = tuple(sorted(sim_results, key=_digital_sequence_result_sort_key))
        if not results:
            raise RuntimeError(f"{table.dut_name} digital sequence returned no results")

        rows_by_bits: dict[tuple[int, ...], DigitalTruthTableRow] = {}
        propagation_delays: list[DigitalPropagationDelayRow] = []
        requested_delay = "max_propagation_delay" in measurements
        for sim_result in results:
            if sim_result.status != "ok":
                raise RuntimeError(
                    f"{table.dut_name} digital sequence failed: {sim_result.error_message}"
                )
            time = _digital_sequence_time(table, sim_result)
            schedule = self.plan.sequence_for_result(sim_result)
            states = _states_for_arc_sequence(schedule.arcs)
            expected_stop = schedule.initial_settle + len(states) * schedule.slot_duration
            if float(time[-1]) < expected_stop * 0.99:
                raise RuntimeError(
                    f"{table.dut_name} digital sequence ended early: "
                    f"final_time={float(time[-1])}, expected={expected_stop}"
                )

            for state_index, bits in enumerate(states):
                sample_time = (
                    schedule.initial_settle
                    + state_index * schedule.slot_duration
                    + schedule.slot_duration * table.sample_fraction
                )
                samples = {}
                actual_values = []
                for output in table.outputs:
                    values = np.asarray(sim_result.waveforms[output], dtype=float).reshape(-1)
                    value = float(np.interp(sample_time, time, values))
                    samples[output] = value
                    actual_values.append(self.plan.logic_value(value))
                row = self._row(
                    bits,
                    tuple(actual_values),
                    samples=samples,
                    sample_time=sample_time,
                )
                existing = rows_by_bits.get(bits)
                if existing is not None and existing.actual != row.actual:
                    raise RuntimeError(
                        f"{table.dut_name} digital sequence produced inconsistent samples for "
                        f"input vector {bits_to_text(bits)}"
                    )
                rows_by_bits.setdefault(bits, row)

            if requested_delay:
                propagation_delays.extend(
                    self._propagation_delay_rows_for_sequence(sim_result, schedule.arcs, time)
                )

        missing = [bits_to_text(bits) for bits in combinations if bits not in rows_by_bits]
        if missing:
            raise RuntimeError(
                f"{table.dut_name} digital sequence missing vector results: "
                + ", ".join(missing[:16])
            )
        if requested_delay and not propagation_delays:
            raise RuntimeError(f"{table.dut_name} propagation delay produced no measurable arcs")
        rows = [rows_by_bits[bits] for bits in combinations]
        coverage = (
            self.plan.propagation_delay_coverage(len(propagation_delays))
            if requested_delay
            else None
        )
        return DigitalTruthTableResult(
            rows,
            results[0],
            mode,
            claim=table.claim,
            sim_results=results,
            propagation_delays=propagation_delays,
            propagation_delay_coverage=coverage,
        )

    def _propagation_delay_rows_for_sequence(
        self,
        sim_result: SimResult,
        scheduled_arcs: tuple[DigitalPropagationDelayArc, ...],
        time: np.ndarray,
    ) -> tuple[DigitalPropagationDelayRow, ...]:
        table = self.table
        rows: list[DigitalPropagationDelayRow] = []
        for arc in scheduled_arcs:
            if not arc.output_indices:
                continue
            input_name = table.inputs[arc.input_index]
            input_crossing = _first_threshold_crossing(
                time,
                sim_result.waveforms[input_name],
                threshold=table.threshold,
                edge=arc.input_edge,
                start=arc.trigger_start,
                stop=arc.trigger_end,
                signal_name=input_name,
            )
            for output_index in arc.output_indices:
                output_name = table.outputs[output_index]
                output_crossing = _first_threshold_crossing(
                    time,
                    sim_result.waveforms[output_name],
                    threshold=table.threshold,
                    edge=arc.output_edge(output_index),
                    start=arc.trigger_start,
                    stop=arc.stop,
                    signal_name=output_name,
                )
                rows.append(
                    DigitalPropagationDelayRow(
                        from_inputs=arc.from_inputs,
                        to_inputs=arc.to_inputs,
                        input_name=input_name,
                        output_name=output_name,
                        input_edge=arc.input_edge,
                        output_edge=arc.output_edge(output_index),
                        input_crossing=input_crossing,
                        output_crossing=output_crossing,
                        delay=output_crossing - input_crossing,
                    )
                )
        return tuple(rows)

    def _row(
        self,
        inputs: tuple[int, ...],
        actual: tuple[int, ...],
        *,
        samples: dict[str, float] | None = None,
        sample_time: float | None = None,
    ) -> DigitalTruthTableRow:
        table = self.table
        expected = table.expected_for(inputs)
        comparison = self._compare(inputs, actual, expected, samples or {})
        return DigitalTruthTableRow(
            inputs=inputs,
            actual=actual,
            expected=expected,
            samples=samples,
            sample_time=sample_time,
            claim=table.claim,
            comparison=comparison,
        )

    def _compare(
        self,
        inputs: tuple[int, ...],
        actual: tuple[int, ...],
        expected: tuple[int, ...] | None,
        samples: Mapping[str, float],
    ) -> DigitalComparisonResult | None:
        table = self.table
        if table.comparator is None:
            return None
        context = DigitalComparisonContext(
            inputs=inputs,
            outputs=table.outputs,
            actual=actual,
            expected=expected,
            samples=samples,
            vdd=table.vdd,
            threshold=table.threshold,
            claim=table.claim,
        )
        if isinstance(table.comparator, DigitalOutputTolerance):
            return table.comparator.compare(context)
        return self._coerce_comparison_result(table.comparator(context))

    @staticmethod
    def _coerce_comparison_result(value: bool | DigitalComparisonResult) -> DigitalComparisonResult:
        if isinstance(value, DigitalComparisonResult):
            return value
        return DigitalComparisonResult(bool(value))


def _is_digital_sequence_result(sim_result: SimResult) -> bool:
    if not sim_result_has_digital_task_metadata(sim_result):
        return False
    return digital_stimulus_metadata(sim_result).kind == "digital_single_bit_arc_sequence"


def _digital_sequence_result_sort_key(sim_result: SimResult) -> tuple[int, int]:
    stimulus = digital_stimulus_metadata(sim_result)
    return (stimulus.chunk_index, stimulus.arc_start)


def _digital_sequence_time(table: DigitalTruthTable, sim_result: SimResult) -> np.ndarray:
    if sim_result.sweep_var is None:
        raise RuntimeError(f"{table.dut_name} digital sequence did not return a sweep vector")
    time = np.asarray(sim_result.sweep_var, dtype=float).reshape(-1)
    if len(time) < 2:
        raise RuntimeError(f"{table.dut_name} digital sequence returned too few time samples")
    return time


def _first_threshold_crossing(
    time: np.ndarray,
    values,
    *,
    threshold: float,
    edge: Literal["rise", "fall"],
    start: float,
    stop: float,
    signal_name: str,
) -> float:
    edge_name = "rising" if edge == "rise" else "falling"
    try:
        return float(
            cross(
                time,
                values,
                threshold=threshold,
                edge=edge_name,
                start=start,
                stop=stop,
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            f"{signal_name} did not cross threshold={threshold} on {edge} edge "
            f"between {start} and {stop}"
        ) from exc
