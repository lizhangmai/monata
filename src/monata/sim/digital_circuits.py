"""Internal circuit construction helpers for digital truth-table simulations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from monata.netlist import Circuit, SubCircuit
from monata.sim.digital_timing import (
    DigitalPropagationDelayArc,
    _state_bit,
    _states_for_arc_sequence,
)

if TYPE_CHECKING:
    from monata.sim.digital_table import DigitalTruthTable

SubCircuitInput = type[SubCircuit] | SubCircuit


@dataclass(frozen=True)
class DigitalTruthTableCircuitBuilder:
    table: DigitalTruthTable

    def digital_sequence_circuit(
        self,
        arcs: tuple[DigitalPropagationDelayArc, ...],
        *,
        initial_settle: float,
        slot_duration: float,
    ) -> Circuit:
        table = self.table
        circuit = self._base_circuit(f"{table.dut_name} digital single-bit arc sequence")
        dut = self._add_dut(circuit)
        states = _states_for_arc_sequence(arcs)

        for index, input_name in enumerate(table.inputs):
            self._add_digital_sequence_source(
                circuit,
                input_name,
                states,
                index,
                initial_settle=initial_settle,
                slot_duration=slot_duration,
            )
        for index, input_name in enumerate(table.complement_inputs):
            self._add_digital_sequence_source(
                circuit,
                input_name,
                states,
                index,
                initial_settle=initial_settle,
                slot_duration=slot_duration,
                inverted=True,
            )

        circuit.instance("dut", tuple(self._node_for(node) for node in dut.nodes), dut.name)
        self._add_loads(circuit)
        return self._project_circuit(circuit)

    def _base_circuit(self, title: str) -> Circuit:
        table = self.table
        circuit = Circuit(title)
        if table.setup is not None:
            table.setup(circuit)
        circuit.vdc("dd", table.rails[0], table.rails[1], table.vdd)
        for dependency in table.dependencies:
            circuit.subckt(_subckt_instance(dependency))
        return circuit

    def _add_dut(self, circuit: Circuit) -> SubCircuit:
        dut = _subckt_instance(self.table.dut)
        circuit.subckt(dut)
        return dut

    def _project_circuit(self, circuit: Circuit) -> Circuit:
        table = self.table
        if table.library is None:
            return circuit
        if table.model_flow is None:
            return table.library.project_pdk_instances(circuit, corner=table.corner)
        projected = table.library.project_pdk_instances(
            circuit,
            corner=table.corner,
            include_models=False,
        )
        table.model_flow.model_selection.apply_to_circuit(projected)
        return projected

    def _add_loads(self, circuit: Circuit, prefix: str | None = None) -> None:
        table = self.table
        if table.load_cap is None:
            return
        for output in table.outputs:
            node = self._node_for(output, prefix=prefix)
            name = f"load_{output}" if prefix is None else f"load_{prefix}_{output}"
            circuit.capacitor(name, node, table.rails[1], table.load_cap)

    def _add_digital_sequence_source(
        self,
        circuit: Circuit,
        source_name: str,
        states: tuple[tuple[int, ...], ...],
        input_index: int,
        *,
        initial_settle: float,
        slot_duration: float,
        inverted: bool = False,
    ) -> None:
        table = self.table
        circuit.vpwl(
            source_name,
            source_name,
            table.rails[1],
            *self._digital_sequence_source_points(
                states,
                input_index,
                initial_settle=initial_settle,
                slot_duration=slot_duration,
                inverted=inverted,
            ),
        )

    def _digital_sequence_source_points(
        self,
        states: tuple[tuple[int, ...], ...],
        input_index: int,
        *,
        initial_settle: float,
        slot_duration: float,
        inverted: bool = False,
    ) -> tuple[tuple[float, float], ...]:
        if len(states) < 2:
            raise RuntimeError(f"{self.table.dut_name} digital sequence requires at least two states")
        points: list[tuple[float, float]] = []
        transition = min(max(self.table.transition, 0.0), slot_duration)
        input_skew = self._input_skew(input_index, slot_duration=slot_duration)
        previous_level = self._level(_state_bit(states[0], input_index, inverted=inverted))
        _append_pwl_point(points, 0.0, previous_level)
        _append_pwl_point(points, initial_settle, previous_level)
        _append_pwl_point(points, initial_settle + slot_duration, previous_level)
        for state_index, state in enumerate(states[1:], start=1):
            boundary = initial_settle + state_index * slot_duration
            transition_start = boundary + input_skew
            stop = initial_settle + (state_index + 1) * slot_duration
            level = self._level(_state_bit(state, input_index, inverted=inverted))
            _append_pwl_point(points, transition_start, previous_level)
            _append_pwl_point(points, transition_start + transition, level)
            _append_pwl_point(points, stop, level)
            previous_level = level
        return tuple(points)

    def _node_for(self, node: str, prefix: str | None = None) -> str:
        table = self.table
        if node in table.inputs or node in table.complement_inputs or node in table.outputs:
            return node if prefix is None else f"{prefix}_{node}"

        lower = node.lower()
        if node == table.rails[0] or lower in {"vdd", "vcc"}:
            return table.rails[0]
        if node == table.rails[1] or lower in {"vss", "gnd"} or node == "0":
            return table.rails[1]
        return node if prefix is None else f"{prefix}_{node}"

    def _level(self, bit: int) -> float:
        return self.table.vdd if bit else 0.0

    def _input_skew(self, input_index: int, *, slot_duration: float) -> float:
        skew = max(self.table.skew_step, 0.0) * int(input_index)
        if skew + max(self.table.transition, 0.0) >= slot_duration:
            raise RuntimeError(
                f"{self.table.dut_name} digital sequence period is too short for skewed input timing: "
                f"slot_duration={slot_duration}, transition={self.table.transition}, "
                f"skew_step={self.table.skew_step}, input_index={input_index}"
            )
        return skew


def _append_pwl_point(points: list[tuple[float, float]], time: float, value: float) -> None:
    if points and points[-1][0] == time and points[-1][1] == value:
        return
    points.append((float(time), float(value)))


def _subckt_instance(value: SubCircuitInput) -> SubCircuit:
    if isinstance(value, type) and issubclass(value, SubCircuit):
        return value()
    if isinstance(value, SubCircuit):
        return deepcopy(value)
    raise TypeError("digital truth-table DUT/dependencies must be SubCircuit classes or instances")


def _subckt_name(value: SubCircuitInput) -> str:
    if isinstance(value, type) and issubclass(value, SubCircuit):
        return value.subckt_name()
    if isinstance(value, SubCircuit):
        return value.name
    raise TypeError("digital truth-table DUT must be a SubCircuit class or instance")
