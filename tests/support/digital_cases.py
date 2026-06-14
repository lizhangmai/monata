from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from monata.circuits import (
    TransistorParams,
    add_inverter,
    add_nand2,
    add_nmos,
    add_nor2,
    add_pmos,
    add_transmission_gate,
)
from monata.corner import OperatingCorner
from monata.netlist import SubCircuit
from monata.sim.core import SimResult, SimTask, TranSpec
from monata.sim.digital_claims import (
    DigitalVerificationClaim,
)
from monata.sim.digital_plan import DigitalTruthTablePlan
from monata.sim.digital_table import (
    DigitalTruthTable,
    ExpectedTable,
)

class And2(SubCircuit):
    NAME = "and2"
    NODES = ("a", "b", "out", "vdd", "gnd")

    def build(self):
        pass


class HelperCell(SubCircuit):
    NAME = "helper"
    NODES = ("in", "out", "vdd", "gnd")

    def build(self):
        pass


class Inverter(SubCircuit):
    NAME = "inv"
    NODES = ("vin", "out", "vdd", "gnd")

    def build(self):
        pass


class PdkInverter(SubCircuit):
    NAME = "pdk_inv"
    NODES = ("vin", "out", "vdd", "gnd")

    def build(self):
        self.pdk_instance(
            "mn",
            lib="PTM_BULK",
            cell="nmos",
            view="ngspice",
            pins={"d": "out", "g": "vin", "s": "gnd", "b": "gnd"},
            params={"w": "1.2u", "l": "65n"},
        )


class BarePdkInverter(SubCircuit):
    NAME = "bare_pdk_inv"
    NODES = ("vin", "out", "vdd", "gnd")

    def build(self):
        self.pdk_instance(
            "mn",
            lib="PTM_MG",
            cell="nfet",
            view="ngspice",
            pins={"d": "out", "g": "vin", "s": "gnd", "b": "gnd"},
        )
        self.pdk_instance(
            "mn_explicit",
            lib="PTM_MG",
            cell="nfet",
            view="ngspice",
            pins={"d": "out2", "g": "vin", "s": "gnd", "b": "gnd"},
            params={"l": "50n"},
        )


LOGIC_GATE_PARAMS = TransistorParams(
    techlib="UNIT_CMOS",
    nmos_cell="nmos",
    pmos_cell="pmos",
    w_n="1u",
    l_n="14n",
    w_p="2u",
    l_p="14n",
    power_node="vdd",
    ground_node="gnd",
)


class LogicInverter(SubCircuit):
    NAME = "inverter"
    NODES = ("vin", "out", "vdd", "gnd")

    def build(self):
        add_inverter(self, "inv", "vin", "out", LOGIC_GATE_PARAMS)


class LogicNand2(SubCircuit):
    NAME = "nand2"
    NODES = ("a", "b", "out", "vdd", "gnd")

    def build(self):
        add_nand2(self, "nand", "a", "b", "out", LOGIC_GATE_PARAMS)


class LogicNor2(SubCircuit):
    NAME = "nor2"
    NODES = ("a", "b", "out", "vdd", "gnd")

    def build(self):
        add_nor2(self, "nor", "a", "b", "out", LOGIC_GATE_PARAMS)


class LogicAnd2(SubCircuit):
    NAME = "and2"
    NODES = ("a", "b", "out", "vdd", "gnd")

    def build(self):
        add_nand2(self, "nand", "a", "b", "nand_out", LOGIC_GATE_PARAMS)
        add_inverter(self, "inv", "nand_out", "out", LOGIC_GATE_PARAMS)


class LogicOr2(SubCircuit):
    NAME = "or2"
    NODES = ("a", "b", "out", "vdd", "gnd")

    def build(self):
        add_nor2(self, "nor", "a", "b", "nor_out", LOGIC_GATE_PARAMS)
        add_inverter(self, "inv", "nor_out", "out", LOGIC_GATE_PARAMS)


class LogicXor2(SubCircuit):
    NAME = "xor2"
    NODES = ("a", "b", "out", "vdd", "gnd")

    def build(self):
        add_inverter(self, "ia", "a", "a_bar", LOGIC_GATE_PARAMS)
        add_inverter(self, "ib", "b", "b_bar", LOGIC_GATE_PARAMS)
        add_transmission_gate(self, "tg1", "a", "out", "b_bar", "b", LOGIC_GATE_PARAMS)
        add_transmission_gate(self, "tg2", "a_bar", "out", "b", "b_bar", LOGIC_GATE_PARAMS)


class LogicXor2_6T(SubCircuit):
    NAME = "xor2_6t"
    NODES = ("a", "b", "out", "vdd", "gnd")

    def build(self):
        add_inverter(self, "inv", "a", "a_bar", LOGIC_GATE_PARAMS)
        add_transmission_gate(self, "tg", "b", "out", "a_bar", "a", LOGIC_GATE_PARAMS)
        add_pmos(self, "sp", "out", "b", "a", "vdd", LOGIC_GATE_PARAMS)
        add_nmos(self, "sn", "out", "b", "a_bar", "gnd", LOGIC_GATE_PARAMS)


@dataclass(frozen=True)
class LogicGateCase:
    name: str
    circuit: type[SubCircuit]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    expected: "ExpectedTable"
    pdk_instance_count: int


EXACT_CLAIM = DigitalVerificationClaim.from_oracle("exact").as_dict()


OBSERVED_CLAIM = DigitalVerificationClaim.from_oracle("observed").as_dict()


TOLERANCED_CLAIM = DigitalVerificationClaim.from_oracle("toleranced").as_dict()


CUSTOM_CLAIM = DigitalVerificationClaim.from_oracle("custom").as_dict()


class RecordingProjectionLibrary:
    def __init__(self):
        self.calls = []

    def project_pdk_instances(
        self,
        netlist,
        registry=None,
        corner=None,
        reference_mode="concrete",
        include_models=True,
    ):
        del registry, reference_mode, include_models
        corner_name = corner.name if isinstance(corner, OperatingCorner) else corner
        subcircuits = [subcircuit.ensure_built() for subcircuit in netlist.subcircuits]
        self.calls.append(
            {
                "corner": corner,
                "corner_name": corner_name,
                "subcircuit_names": [subcircuit.name for subcircuit in subcircuits],
                "pdk_instance_counts": [
                    len(subcircuit.pdk_instances) for subcircuit in subcircuits
                ],
            }
        )
        netlist.options(projected=corner_name or "default")
        return netlist


def _expected_and(bits):
    return (bits[0] & bits[1],)


AND2_EXPECTED_TABLE = ExpectedTable.from_rows(
    [
        ("00", "0"),
        ("01", "0"),
        ("10", "0"),
        ("11", "1"),
    ]
)


INV_EXPECTED_TABLE = ExpectedTable.from_rows(
    [
        ("0", "1"),
        ("1", "0"),
    ]
)


XOR2_EXPECTED_TABLE = ExpectedTable.from_rows(
    [
        ("00", "0"),
        ("01", "1"),
        ("10", "1"),
        ("11", "0"),
    ]
)


LOGIC_GATE_CASES = (
    LogicGateCase("inverter", LogicInverter, ("vin",), ("out",), INV_EXPECTED_TABLE, 2),
    LogicGateCase(
        "nand2",
        LogicNand2,
        ("a", "b"),
        ("out",),
        ExpectedTable.from_rows(
            [
                ("00", "1"),
                ("01", "1"),
                ("10", "1"),
                ("11", "0"),
            ]
        ),
        4,
    ),
    LogicGateCase(
        "nor2",
        LogicNor2,
        ("a", "b"),
        ("out",),
        ExpectedTable.from_rows(
            [
                ("00", "1"),
                ("01", "0"),
                ("10", "0"),
                ("11", "0"),
            ]
        ),
        4,
    ),
    LogicGateCase("and2", LogicAnd2, ("a", "b"), ("out",), AND2_EXPECTED_TABLE, 6),
    LogicGateCase(
        "or2",
        LogicOr2,
        ("a", "b"),
        ("out",),
        ExpectedTable.from_rows(
            [
                ("00", "0"),
                ("01", "1"),
                ("10", "1"),
                ("11", "1"),
            ]
        ),
        6,
    ),
    LogicGateCase("xor2", LogicXor2, ("a", "b"), ("out",), XOR2_EXPECTED_TABLE, 8),
    LogicGateCase("xor2_6t", LogicXor2_6T, ("a", "b"), ("out",), XOR2_EXPECTED_TABLE, 6),
)


def _inverter_timing_waveforms(
    table: DigitalTruthTable,
    time: np.ndarray,
    delays: tuple[float, ...],
) -> dict[str, np.ndarray]:
    input_points: list[tuple[float, float]] = []
    output_points: list[tuple[float, float]] = []
    previous_input: float | None = None
    previous_output: float | None = None
    for arc, delay in zip(table.propagation_delay_arcs(), delays):
        initial_input = float(arc.from_inputs[0])
        final_input = float(arc.to_inputs[0])
        initial_output = float(arc.from_outputs[0])
        final_output = float(arc.to_outputs[0])
        if previous_input is None:
            previous_input = initial_input
        if previous_output is None:
            previous_output = initial_output

        _append_test_point(input_points, arc.start, previous_input)
        _append_test_point(input_points, arc.reset_end, initial_input)
        _append_test_point(input_points, arc.trigger_start, initial_input)
        _append_test_point(input_points, arc.trigger_end, final_input)
        _append_test_point(input_points, arc.stop, final_input)

        input_crossing = (arc.trigger_start + arc.trigger_end) / 2.0
        output_crossing = input_crossing + delay
        _append_test_point(output_points, arc.start, previous_output)
        _append_test_point(output_points, arc.reset_end, initial_output)
        _append_test_point(output_points, output_crossing - table.transition / 2.0, initial_output)
        _append_test_point(output_points, output_crossing + table.transition / 2.0, final_output)
        _append_test_point(output_points, arc.stop, final_output)

        previous_input = final_input
        previous_output = final_output
    return {
        "vin": np.interp(time, [point[0] for point in input_points], [point[1] for point in input_points]),
        "out": np.interp(time, [point[0] for point in output_points], [point[1] for point in output_points]),
    }


def _append_test_point(points: list[tuple[float, float]], time: float, value: float) -> None:
    if points and points[-1][0] == time and points[-1][1] == value:
        return
    points.append((time, value))


def _pwl_points(values) -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(values[index]), float(values[index + 1]))
        for index in range(0, len(values), 2)
    )


def _has_pwl_point(points: tuple[tuple[float, float], ...], time: float, value: float) -> bool:
    return any(
        point_time == pytest.approx(time) and point_value == pytest.approx(value)
        for point_time, point_value in points
    )


def _digital_task_metadata(task_or_metadata) -> Mapping[str, Any]:
    metadata = getattr(task_or_metadata, "metadata", task_or_metadata)
    assert isinstance(metadata, Mapping)
    namespace = metadata["monata"]
    assert isinstance(namespace, Mapping)
    payload = namespace["digital_task_v1"]
    assert isinstance(payload, Mapping)
    return payload


def _sequence_result_for_task(
    table: DigitalTruthTable,
    task: SimTask,
    *,
    delay: float | tuple[float, ...] = 0.0,
):
    dummy = SimResult(
        status="ok",
        sweep_var=np.array([0.0, 1.0]),
        waveforms={},
        corner=None,
        metadata=task.metadata,
    )
    schedule = DigitalTruthTablePlan(table).sequence_for_result(dummy)
    arcs = schedule.arcs
    initial_settle = schedule.initial_settle
    slot_duration = schedule.slot_duration
    states = [arcs[0].from_inputs, *(arc.to_inputs for arc in arcs)]
    analysis_spec = task.analysis_spec
    assert isinstance(analysis_spec, TranSpec)
    stop = float(analysis_spec.stop)
    step = float(analysis_spec.step or slot_duration / 20.0)
    time = np.linspace(0.0, stop, max(2, int(stop / step) + 1))
    waveforms: dict[str, np.ndarray] = {}
    for input_index, input_name in enumerate(table.inputs):
        points = []
        previous = float(states[0][input_index]) * table.vdd
        _append_test_point(points, 0.0, previous)
        _append_test_point(points, initial_settle, previous)
        _append_test_point(points, initial_settle + slot_duration, previous)
        for state_index, state in enumerate(states[1:], start=1):
            boundary = initial_settle + state_index * slot_duration
            transition_start = boundary + table.skew_step * input_index
            level = float(state[input_index]) * table.vdd
            _append_test_point(points, transition_start, previous)
            _append_test_point(points, transition_start + table.transition, level)
            _append_test_point(points, initial_settle + (state_index + 1) * slot_duration, level)
            previous = level
        waveforms[input_name] = np.interp(
            time,
            [point[0] for point in points],
            [point[1] for point in points],
        )
    for output_index, output_name in enumerate(table.outputs):
        points = []
        initial = _sequence_expected_outputs(table, states[0])[output_index]
        previous = float(initial) * table.vdd
        _append_test_point(points, 0.0, previous)
        _append_test_point(points, initial_settle, previous)
        _append_test_point(points, initial_settle + slot_duration, previous)
        for arc_index, arc in enumerate(arcs):
            next_bit = _sequence_expected_outputs(table, arc.to_inputs)[output_index]
            level = float(next_bit) * table.vdd
            if level == previous:
                _append_test_point(points, arc.stop, level)
            else:
                delay_value = delay[arc_index] if isinstance(delay, tuple) else delay
                input_crossing = (arc.trigger_start + arc.trigger_end) / 2.0
                output_crossing = input_crossing + delay_value
                _append_test_point(points, arc.trigger_start, previous)
                _append_test_point(points, output_crossing - table.transition / 2.0, previous)
                _append_test_point(points, output_crossing + table.transition / 2.0, level)
                _append_test_point(points, arc.stop, level)
            previous = level
        waveforms[output_name] = np.interp(
            time,
            [point[0] for point in points],
            [point[1] for point in points],
        )
    return SimResult(
        status="ok",
        sweep_var=time,
        waveforms=waveforms,
        corner=None,
        metadata=task.metadata,
    )


def _sequence_expected_outputs(table: DigitalTruthTable, bits: tuple[int, ...]) -> tuple[int, ...]:
    expected = table.expected_for(bits)
    if expected is not None:
        return expected
    return tuple(bits[0] & bits[1] for _output in table.outputs)
