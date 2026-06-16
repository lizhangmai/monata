"""Internal circuit construction helpers for digital truth-table simulations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from monata._home import monata_cache_dir
from monata.netlist import Circuit, SubCircuit

if TYPE_CHECKING:
    from monata.sim.digital_stim import DigitalStimulusConfig as _TableLike

SubCircuitInput = type[SubCircuit] | SubCircuit


def circuit_depot_dir() -> Path:
    """Persistent depot under ``$MONATA_HOME/cache/circuits``.

    Used to cache serialised SubCircuit definitions across runner
    invocations so dependency instantiation can skip rebuilds.
    """
    depot = monata_cache_dir() / "circuits"
    depot.mkdir(parents=True, exist_ok=True)
    return depot


@dataclass(frozen=True)
class DigitalTruthTableCircuitBuilder:
    table: _TableLike
    _dep_cache: dict[str, SubCircuit] = field(default_factory=dict, init=False, repr=False)

    def clocked_sequence_circuit(
        self,
        states: tuple[tuple[int, ...], ...],
        *,
        initial_settle: float,
        clock_period: float,
    ) -> Circuit:
        """Build a clock-driven circuit with PWL inputs from a state sequence.

        A ``vpulse`` clock starts after *initial_settle* and runs at
        *clock_period*.  Input PWL sources hold ``states[0]`` through the
        settle window, then transition to each subsequent state at the
        next clock rising edge.
        """
        table = self.table
        circuit = self._base_circuit(f"{table.dut_name} digital sequence")
        dut = self._add_dut(circuit)

        transition = max(table.transition, 0.0)
        clock_rise = clock_fall = min(transition, clock_period * 0.05)
        circuit.vpulse(
            "clk",
            "clk",
            table.rails[1],
            0.0,
            table.vdd,
            initial_settle,
            clock_rise,
            clock_fall,
            clock_period / 2.0,
            clock_period,
        )

        for index, input_name in enumerate(table.inputs):
            self._add_clocked_source(
                circuit,
                input_name,
                states,
                index,
                initial_settle=initial_settle,
                clock_period=clock_period,
            )
        for index, input_name in enumerate(table.complement_inputs):
            self._add_clocked_source(
                circuit,
                input_name,
                states,
                index,
                initial_settle=initial_settle,
                clock_period=clock_period,
                inverted=True,
            )

        circuit.instance("dut", tuple(self._node_for(node) for node in dut.nodes), dut.name)
        self._add_loads(circuit)
        return self._project_circuit(circuit)

    def _add_clocked_source(
        self,
        circuit: Circuit,
        source_name: str,
        states: tuple[tuple[int, ...], ...],
        input_index: int,
        *,
        initial_settle: float,
        clock_period: float,
        inverted: bool = False,
    ) -> None:
        table = self.table
        circuit.vpwl(
            source_name,
            source_name,
            table.rails[1],
            *self._clocked_source_points(
                states,
                input_index,
                initial_settle=initial_settle,
                clock_period=clock_period,
                inverted=inverted,
            ),
        )

    def _clocked_source_points(
        self,
        states: tuple[tuple[int, ...], ...],
        input_index: int,
        *,
        initial_settle: float,
        clock_period: float,
        inverted: bool = False,
    ) -> tuple[tuple[float, float], ...]:
        if not states:
            raise RuntimeError(f"{self.table.dut_name} clocked sequence requires at least one state")
        points: list[tuple[float, float]] = []
        transition = min(max(self.table.transition, 0.0), clock_period)
        previous = self._level(_state_bit(states[0], input_index, inverted=inverted))
        _append_pwl_point(points, 0.0, previous)
        if len(states) == 1:
            return tuple(points)
        _append_pwl_point(points, initial_settle, previous)
        for cycle, next_state in enumerate(states[1:]):
            edge = initial_settle + float(cycle) * clock_period
            next_level = self._level(_state_bit(next_state, input_index, inverted=inverted))
            if next_level != previous:
                _append_pwl_point(points, edge, previous)
                _append_pwl_point(points, edge + transition, next_level)
            _append_pwl_point(points, edge + clock_period, next_level)
            previous = next_level
        return tuple(points)

    def _base_circuit(self, title: str) -> Circuit:
        table = self.table
        circuit = Circuit(title)
        if table.setup is not None:
            table.setup(circuit)
        circuit.vdc("dd", table.rails[0], table.rails[1], table.vdd)
        for dependency in table.dependencies:
            dep_name = _subckt_name(dependency)
            if dep_name not in self._dep_cache:
                entry = _load_cached_dependency(dep_name, dependency)
                self._dep_cache[dep_name] = entry
            circuit.subckt(deepcopy(self._dep_cache[dep_name]))
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


def _state_bit(state: tuple[int, ...], input_index: int, *, inverted: bool = False) -> int:
    return 1 - state[input_index] if inverted else state[input_index]


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


def _load_cached_dependency(dep_name: str, dependency: SubCircuitInput) -> SubCircuit:
    """Return a dependency SubCircuit, preferring the in-process cache.

    On first access the instance is built and its SPICE text is persisted
    to the Monata circuit depot at ``$MONATA_HOME/cache/circuits``.
    Subsequent runner invocations can seed the cache from the depot
    (future enhancement), but the authoritative cache is the in-process
    dictionary shared across chunks within a single run.
    """
    instance = _subckt_instance(dependency)
    try:
        depot = circuit_depot_dir()
        depot_path = depot / f"{dep_name}.spice"
        if not depot_path.exists():
            depot_path.write_text(instance.to_spice(), encoding="utf-8")
    except Exception:
        pass
    return instance


def _subckt_name(value: SubCircuitInput) -> str:
    if isinstance(value, type) and issubclass(value, SubCircuit):
        return value.subckt_name()
    if isinstance(value, SubCircuit):
        return value.name
    raise TypeError("digital truth-table DUT must be a SubCircuit class or instance")
