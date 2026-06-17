"""Digital waveform analysis — truth-table verification and delay measurement.

This module owns the *verification* side of the digital pipeline: it
takes simulation waveforms and extracts truth-table rows, propagation
delays, and pass/fail results.  It has no knowledge of how the
waveforms were produced — that concern lives in ``monata.digital.stim``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from monata.measure.time_domain import cross
from monata.digital.bits import bits_to_text, gray_code_bit_flip
from monata.digital.plan import (
    digital_stimulus_metadata,
    sim_result_has_digital_task_metadata,
)
from monata.digital.results import (
    DigitalPropagationDelayRow,
    DigitalTruthTableResult,
    DigitalTruthTableRow,
)
from monata.digital.spec import (
    DigitalMeasurementName,
    DigitalVerificationSpec,
)
from monata.sim.results import SimResult


@dataclass(frozen=True)
class DigitalWaveformAnalyzer:
    """Analyse simulation waveforms against a verification spec.

    This is a pure function of *waveforms* + *spec*: it has no
    dependency on how the simulation was built or executed.
    """

    spec: DigitalVerificationSpec

    @property
    def _threshold(self) -> float:
        return 0.5

    @property
    def _outputs(self) -> tuple[str, ...]:
        return self.spec.outputs

    def verify(
        self,
        sim_results: Iterable[SimResult],
        *,
        measurements: tuple[DigitalMeasurementName, ...] = ("truth_table",),
        vdd: float = 1.0,
        sample_fraction: float = 0.9,
        vector_sequence: tuple[tuple[int, ...], ...] | None = None,
    ) -> DigitalTruthTableResult:
        return self._transient_results(
            sim_results,
            "transient",
            measurements=measurements,
            vdd=vdd,
            sample_fraction=sample_fraction,
            vector_sequence=vector_sequence,
        )

    def _transient_results(
        self,
        sim_results: Iterable[SimResult],
        mode: str,
        *,
        measurements: tuple[DigitalMeasurementName, ...],
        vdd: float,
        sample_fraction: float,
        vector_sequence: tuple[tuple[int, ...], ...] | None,
    ) -> DigitalTruthTableResult:
        spec = self.spec
        combinations = (
            tuple(
                tuple((v >> s) & 1 for s in range(len(spec.inputs) - 1, -1, -1))
                for v in range(2 ** len(spec.inputs))
            )
            if vector_sequence is None
            else tuple(vector_sequence)
        )
        results = list(sim_results)
        if results and all(_is_digital_sequence_result(r) for r in results):
            return _digital_sequence_results(
                spec,
                results,
                combinations,
                mode,
                measurements=measurements,
                vdd=vdd,
                sample_fraction=sample_fraction,
            )
        if len(results) != len(combinations):
            raise RuntimeError(
                f"{spec.dut} transient result count mismatch: "
                f"expected={len(combinations)}, got={len(results)}"
            )
        rows = []
        first_result = results[0]
        for bits, sim_result in zip(combinations, results):
            if sim_result.status != "ok":
                raise RuntimeError(
                    f"{spec.dut} transient static-vector task failed for {bits_to_text(bits)}: "
                    f"{sim_result.error_message}"
                )
            samples = {}
            actual_values = []
            for output in self._outputs:
                value = float(np.asarray(sim_result.waveforms[output]).reshape(-1)[-1])
                samples[output] = value
                actual_values.append(int(value > self._threshold))
            actual = tuple(actual_values)
            sample_time = None
            if sim_result.sweep_var is not None and len(sim_result.sweep_var):
                sample_time = float(np.asarray(sim_result.sweep_var).reshape(-1)[-1])
            rows.append(_row(spec, bits, actual, spec.expected, samples=samples, sample_time=sample_time))

        return DigitalTruthTableResult(rows, first_result, mode, claim=None, sim_results=results)


def _is_digital_sequence_result(sim_result: SimResult) -> bool:
    if not sim_result_has_digital_task_metadata(sim_result):
        return False
    return digital_stimulus_metadata(sim_result).kind == "digital_sequence"


def _digital_sequence_result_sort_key(sim_result: SimResult) -> tuple[int, int]:
    stimulus = digital_stimulus_metadata(sim_result)
    return (stimulus.chunk_index, 0)


def _bits_from_text(text: str) -> tuple[int, ...]:
    bits = tuple(int(bit) for bit in text.strip())
    if not bits or any(bit not in {0, 1} for bit in bits):
        raise RuntimeError(f"digital stimulus state_sequence contains invalid bits: {text!r}")
    return bits


def _digital_sequence_results(
    spec: DigitalVerificationSpec,
    sim_results: Iterable[SimResult],
    combinations: tuple[tuple[int, ...], ...],
    mode: str,
    *,
    measurements: tuple[DigitalMeasurementName, ...],
    vdd: float,
    sample_fraction: float,
) -> DigitalTruthTableResult:
    results = tuple(sorted(sim_results, key=_digital_sequence_result_sort_key))
    if not results:
        raise RuntimeError(f"{spec.dut} digital sequence returned no results")

    rows_by_bits: dict[tuple[int, ...], DigitalTruthTableRow] = {}
    propagation_delays: list[DigitalPropagationDelayRow] = []
    requested_delay = "max_propagation_delay" in measurements
    all_sim_results: list[SimResult] = []
    for sim_result in results:
        if sim_result.status != "ok":
            raise RuntimeError(f"{spec.dut} digital sequence failed: {sim_result.error_message}")
        all_sim_results.append(sim_result)
        time = _digital_sequence_time(spec, sim_result)
        task_metadata = sim_result.metadata.get("monata", {}).get("digital_task_v1", {})
        stimulus = task_metadata.get("stimulus", {})
        initial_settle = float(stimulus.get("initial_settle", 0.0))
        clock_period = float(stimulus.get("clock_period", 1e-9))
        raw_states = list(stimulus.get("state_sequence", []))
        if not raw_states:
            raise RuntimeError(f"{spec.dut} digital sequence metadata missing state_sequence")
        states = tuple(_bits_from_text(str(text)) for text in raw_states)
        expected_stop = initial_settle + (len(states) - 1) * clock_period
        if float(time[-1]) < expected_stop * 0.99:
            raise RuntimeError(
                f"{spec.dut} digital sequence ended early: "
                f"final_time={float(time[-1])}, expected={expected_stop}"
            )

        for cycle, bits in enumerate(states):
            if cycle == 0:
                window_start = 0.0
                window_stop = initial_settle
            else:
                window_start = initial_settle + float(cycle - 1) * clock_period
                window_stop = initial_settle + float(cycle) * clock_period
            row = _functional_window_row(
                spec, bits, time, sim_result.waveforms,
                threshold=vdd / 2.0,
                start=window_start,
                stop=window_stop,
            )
            existing = rows_by_bits.get(bits)
            if existing is not None and existing.actual != row.actual:
                raise RuntimeError(
                    f"{spec.dut} digital sequence produced inconsistent samples for "
                    f"input vector {bits_to_text(bits)}"
                )
            rows_by_bits.setdefault(bits, row)

        if requested_delay:
            propagation_delays.extend(
                _propagation_delay_rows_for_sequence(
                    spec, sim_result, states, time,
                    initial_settle=initial_settle, clock_period=clock_period,
                    vdd=vdd,
                )
            )

    missing = [bits_to_text(bits) for bits in combinations if bits not in rows_by_bits]
    if missing:
        raise RuntimeError(
            f"{spec.dut} digital sequence missing vector results: " + ", ".join(missing[:16])
        )
    if requested_delay and not propagation_delays:
        raise RuntimeError(f"{spec.dut} propagation delay produced no measurable arcs")
    rows = [rows_by_bits[bits] for bits in combinations]
    return DigitalTruthTableResult(
        rows, results[0], mode, claim=None,
        sim_results=all_sim_results,
        propagation_delays=propagation_delays,
        propagation_delay_coverage={
            "kind": "directed_single_bit_exhaustive",
            "input_count": len(spec.inputs),
            "vector_count": len(combinations),
            "measured_delay_rows": len(propagation_delays),
        } if requested_delay else None,
    )


def _propagation_delay_rows_for_sequence(
    spec: DigitalVerificationSpec,
    sim_result: SimResult,
    states: tuple[tuple[int, ...], ...],
    time: np.ndarray,
    *,
    initial_settle: float,
    clock_period: float,
    vdd: float,
) -> tuple[DigitalPropagationDelayRow, ...]:
    threshold = vdd / 2.0
    if len(states) < 2:
        return ()
    try:
        flips = gray_code_bit_flip(states)
    except ValueError:
        return ()
    rows: list[DigitalPropagationDelayRow] = []
    for cycle, (from_state, to_state, input_index) in enumerate(
        zip(states, states[1:], flips)
    ):
        from_outputs = spec.expected(from_state) if spec.expected is not None else None
        to_outputs = spec.expected(to_state) if spec.expected is not None else None
        if from_outputs is None or to_outputs is None:
            continue
        clock_rising_edge = initial_settle + float(cycle) * clock_period
        input_name = spec.inputs[input_index]
        to_bit = to_state[input_index]
        input_edge: Literal["rise", "fall"] = "rise" if to_bit else "fall"
        input_crossing = _first_threshold_crossing(
            time, sim_result.waveforms[input_name],
            threshold=threshold, edge=input_edge,
            start=clock_rising_edge, stop=clock_rising_edge + clock_period,
            signal_name=input_name,
        )
        # The output must settle within one clock period — that is the
        # definition of combinational propagation delay.  If the next
        # input transition arrives before the output has crossed, the
        # measurement is ambiguous and the circuit fails timing at this
        # clock frequency.  Increase clock_period in the recipe for
        # slower circuits instead of widening this window.
        next_input_edge = initial_settle + float(cycle + 1) * clock_period
        output_stop = min(next_input_edge, float(time[-1]))
        output_settles = tuple(
            (
                _functional_settle_time(
                    time,
                    sim_result.waveforms[output_name],
                    expected_bit=to_outputs[output_index],
                    threshold=threshold,
                    start=input_crossing,
                    stop=output_stop,
                    signal_name=output_name,
                ),
                output_index,
                output_name,
            )
            for output_index, output_name in enumerate(spec.outputs)
        )
        output_settle, output_index, output_name = max(output_settles, key=lambda item: item[0])
        output_edge: Literal["rise", "fall"] = "rise" if to_outputs[output_index] else "fall"
        rows.append(
            DigitalPropagationDelayRow(
                from_inputs=from_state, to_inputs=to_state,
                input_name=input_name, output_name=output_name,
                input_edge=input_edge, output_edge=output_edge,
                input_crossing=input_crossing, output_crossing=output_settle,
                delay=output_settle - clock_rising_edge,
            )
        )
    return tuple(rows)


def _row(
    spec: DigitalVerificationSpec,
    inputs: tuple[int, ...],
    actual: tuple[int, ...],
    expected_fn: Any,
    *,
    samples: dict[str, float] | None = None,
    sample_time: float | None = None,
) -> DigitalTruthTableRow:
    expected = expected_fn(inputs) if expected_fn is not None else None
    return DigitalTruthTableRow(
        inputs=inputs, actual=actual, expected=expected,
        samples=samples, sample_time=sample_time,
        claim=None, comparison=None,
    )


def _functional_window_row(
    spec: DigitalVerificationSpec,
    bits: tuple[int, ...],
    time: np.ndarray,
    waveforms: Mapping[str, np.ndarray],
    *,
    threshold: float,
    start: float,
    stop: float,
) -> DigitalTruthTableRow:
    expected = spec.expected(bits) if spec.expected is not None else None
    sample_time = stop
    samples = {
        output: float(np.interp(sample_time, time, np.asarray(waveforms[output], dtype=float).reshape(-1)))
        for output in spec.outputs
    }
    sampled_actual = tuple(int(samples[output] > threshold) for output in spec.outputs)
    if expected is None:
        return _row(spec, bits, sampled_actual, spec.expected, samples=samples, sample_time=sample_time)

    settled = True
    for output_index, output in enumerate(spec.outputs):
        try:
            _functional_settle_time(
                time,
                waveforms[output],
                expected_bit=expected[output_index],
                threshold=threshold,
                start=start,
                stop=stop,
                signal_name=output,
            )
        except RuntimeError:
            settled = False
            break
    actual = expected if settled else sampled_actual
    return _row(spec, bits, actual, spec.expected, samples=samples, sample_time=sample_time)


def _build_sample_times(
    num_states: int, initial_settle: float, clock_period: float, sample_fraction: float,
) -> np.ndarray:
    times = np.empty(num_states, dtype=float)
    times[0] = initial_settle * sample_fraction
    if num_states > 1:
        offsets = np.arange(1, num_states, dtype=float) - 1
        times[1:] = initial_settle + offsets * clock_period + clock_period * sample_fraction
    return times


def _batch_interp_samples(
    time: np.ndarray, waveforms: dict[str, np.ndarray],
    sample_times: np.ndarray, outputs: tuple[str, ...],
) -> tuple[dict[str, float], ...]:
    interpolated: dict[str, np.ndarray] = {}
    for output in outputs:
        values = np.asarray(waveforms[output], dtype=float).reshape(-1)
        interpolated[output] = np.interp(sample_times, time, values)
    result: list[dict[str, float]] = []
    for j in range(len(sample_times)):
        samples: dict[str, float] = {}
        for out in outputs:
            samples[out] = float(interpolated[out][j])
        result.append(samples)
    return tuple(result)


def _digital_sequence_time(spec: DigitalVerificationSpec, sim_result: SimResult) -> np.ndarray:
    if sim_result.sweep_var is None:
        raise RuntimeError(f"{spec.dut} digital sequence did not return a sweep vector")
    time = np.asarray(sim_result.sweep_var, dtype=float).reshape(-1)
    if len(time) < 2:
        raise RuntimeError(f"{spec.dut} digital sequence returned too few time samples")
    return time


def _first_threshold_crossing(
    time: np.ndarray, values, *, threshold: float,
    edge: Literal["rise", "fall"], start: float, stop: float, signal_name: str,
) -> float:
    edge_name = "rising" if edge == "rise" else "falling"
    try:
        return float(cross(time, values, threshold=threshold, edge=edge_name, start=start, stop=stop))
    except ValueError as exc:
        raise RuntimeError(
            f"{signal_name} did not cross threshold={threshold} on {edge} edge "
            f"between {start} and {stop}"
        ) from exc


def _functional_settle_time(
    time: np.ndarray,
    values,
    *,
    expected_bit: int,
    threshold: float,
    start: float,
    stop: float,
    signal_name: str,
) -> float:
    if stop < start:
        raise RuntimeError(f"{signal_name} settle window is empty: start={start}, stop={stop}")
    value_array = np.asarray(values, dtype=float).reshape(-1)
    if len(time) != len(value_array):
        raise RuntimeError(f"{signal_name} waveform length does not match sweep vector")
    interior = time[(time > start) & (time < stop)]
    window_time = np.concatenate(([start], interior, [stop]))
    window_values = np.interp(window_time, time, value_array)
    correct = window_values > threshold if expected_bit else window_values <= threshold
    suffix_correct = np.logical_and.accumulate(correct[::-1])[::-1]
    indexes = np.flatnonzero(suffix_correct)
    if not len(indexes):
        expected = "high" if expected_bit else "low"
        raise RuntimeError(
            f"{signal_name} did not settle {expected} between {start} and {stop}"
        )
    index = int(indexes[0])
    if index == 0:
        return float(start)
    before_time = float(window_time[index - 1])
    after_time = float(window_time[index])
    before_value = float(window_values[index - 1])
    after_value = float(window_values[index])
    edge: Literal["rise", "fall"] = "rise" if expected_bit else "fall"
    if before_time == after_time or before_value == after_value:
        return after_time
    try:
        return float(
            cross(
                np.asarray([before_time, after_time]),
                np.asarray([before_value, after_value]),
                threshold=threshold,
                edge="rising" if edge == "rise" else "falling",
            )
        )
    except ValueError:
        return after_time
