from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

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
from monata.digital.claims import (
    DigitalVerificationClaim,
)
from monata.digital.spec import ExpectedTable
from monata.digital.stim import DigitalStimulusConfig

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


def _make_stimulus(dut_cls, *, inputs, outputs, period=1e-9, step=None, transition=0.0):
    """Build a minimal DigitalStimulusConfig for tests."""
    return DigitalStimulusConfig(
        dut=dut_cls,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        period=float(period),
        step=float(step) if step is not None else None,
        transition=float(transition),
    )


def _sequence_result_for_task(
    stim: DigitalStimulusConfig,
    task: SimTask,
    *,
    delay: float | tuple[float, ...] = 0.0,
):
    payload = _digital_task_metadata(task)
    stimulus_meta = payload["stimulus"]
    assert isinstance(stimulus_meta, Mapping)
    kind = str(stimulus_meta.get("kind", ""))
    analysis_spec = task.analysis_spec
    assert isinstance(analysis_spec, TranSpec)
    stop = float(analysis_spec.stop)
    step = float(analysis_spec.step or 5e-11)
    time = np.linspace(0.0, stop, max(2, int(stop / step) + 1))

    if kind == "digital_sequence":
        return _clocked_sequence_waveforms(
            stim, task, time, stimulus_meta, delay=delay
        )

    raise AssertionError(f"unsupported digital stimulus kind in test helper: {kind}")


def _bits_from_text(text: str) -> tuple[int, ...]:
    return cast(tuple[int, ...], tuple(int(bit) for bit in text.strip()))


def _clocked_sequence_waveforms(
    stim: DigitalStimulusConfig,
    task: SimTask,
    time: np.ndarray,
    stimulus: Mapping[str, object],
    *,
    delay: float | tuple[float, ...] = 0.0,
) -> SimResult:
    import numpy as np

    initial_settle = float(_scalar_metadata_value(stimulus.get("initial_settle", 0.0)))
    clock_period = float(_scalar_metadata_value(stimulus.get("clock_period", stim.period)))
    raw_states = tuple(_iterable_metadata_value(stimulus.get("state_sequence", ())))
    states: list[tuple[int, ...]] = [_bits_from_text(str(text)) for text in raw_states]
    if not states:
        raise ValueError("clocked sequence stimulus requires at least one state")
    transition = max(stim.transition, 0.0)
    delay_value = float(delay[0] if isinstance(delay, tuple) else delay)

    waveforms: dict[str, np.ndarray] = {}
    for input_index, input_name in enumerate(stim.inputs):
        points = []
        previous = float(states[0][input_index]) * stim.vdd
        _append_test_point(points, 0.0, previous)
        if len(states) > 1:
            _append_test_point(points, initial_settle, previous)
            for cycle, next_state in enumerate(states[1:]):
                edge = initial_settle + float(cycle) * clock_period
                next_level = float(next_state[input_index]) * stim.vdd
                if next_level != previous:
                    _append_test_point(points, edge, previous)
                    _append_test_point(points, edge + transition, next_level)
                _append_test_point(points, edge + clock_period, next_level)
                previous = next_level
        waveforms[input_name] = np.interp(time, [p[0] for p in points], [p[1] for p in points])
    for output_index, output_name in enumerate(stim.outputs):
        points = []
        initial_exp = _sequence_expected_outputs_for_stim(stim, states[0])[output_index]
        previous = float(initial_exp) * stim.vdd
        _append_test_point(points, 0.0, previous)
        for cycle in range(1, len(states)):
            edge = initial_settle + float(cycle - 1) * clock_period
            next_exp = _sequence_expected_outputs_for_stim(stim, states[cycle])[output_index]
            level = float(next_exp) * stim.vdd
            if level == previous:
                _append_test_point(points, edge + clock_period, level)
            else:
                input_crossing = edge + transition / 2.0
                output_crossing = input_crossing + delay_value
                _append_test_point(points, edge, previous)
                _append_test_point(points, output_crossing - transition / 2.0, previous)
                _append_test_point(points, output_crossing + transition / 2.0, level)
                _append_test_point(points, edge + clock_period, level)
            previous = level
        waveforms[output_name] = np.interp(time, [p[0] for p in points], [p[1] for p in points])
    return SimResult(status="ok", sweep_var=time, waveforms=waveforms, corner=None, metadata=task.metadata)


def _sequence_expected_outputs_for_stim(stim: DigitalStimulusConfig, bits: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(bits[0] & bits[1] for _output in stim.outputs)


def _scalar_metadata_value(value: object) -> str | float | int:
    if isinstance(value, (str, int, float)):
        return value
    raise TypeError(f"expected scalar metadata value, got {type(value).__name__}")


def _iterable_metadata_value(value: object) -> Iterable[object]:
    if isinstance(value, str):
        raise TypeError("expected iterable metadata value, got str")
    if isinstance(value, Iterable):
        return value
    raise TypeError(f"expected iterable metadata value, got {type(value).__name__}")
