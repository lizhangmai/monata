"""Internal digital timing primitives for truth-table simulations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from monata.sim._digital_bits import bits_to_text

_PROPAGATION_DELAY_TRIGGER_FRACTION = 0.35
_PROPAGATION_DELAY_CHUNK_SIZE = 128
_DIGITAL_SINGLE_BIT_ARC_CHUNK_SIZE = 128


@dataclass(frozen=True)
class DigitalPropagationDelayArc:
    index: int
    from_inputs: tuple[int, ...]
    to_inputs: tuple[int, ...]
    from_outputs: tuple[int, ...]
    to_outputs: tuple[int, ...]
    input_index: int
    output_indices: tuple[int, ...]
    start: float
    reset_end: float
    trigger_start: float
    trigger_end: float
    stop: float

    @property
    def input_edge(self) -> Literal["rise", "fall"]:
        return "rise" if self.to_inputs[self.input_index] > self.from_inputs[self.input_index] else "fall"

    def output_edge(self, output_index: int) -> Literal["rise", "fall"]:
        return "rise" if self.to_outputs[output_index] > self.from_outputs[output_index] else "fall"


def _states_for_arc_sequence(
    arcs: tuple[DigitalPropagationDelayArc, ...],
) -> tuple[tuple[int, ...], ...]:
    if not arcs:
        raise RuntimeError("digital sequence requires at least one arc")
    states = [arcs[0].from_inputs]
    states.extend(arc.to_inputs for arc in arcs)
    return tuple(states)


def _state_bit(state: tuple[int, ...], input_index: int, *, inverted: bool = False) -> int:
    bit = state[input_index]
    return 1 - bit if inverted else bit


def _digital_sequence_stop_time(
    arcs: tuple[DigitalPropagationDelayArc, ...],
    *,
    initial_settle: float,
    slot_duration: float,
) -> float:
    return initial_settle + (len(arcs) + 1) * slot_duration


def _truth_vector_indices_for_arcs(arcs: Iterable[DigitalPropagationDelayArc]) -> list[int]:
    indices = set()
    for arc in arcs:
        indices.add(_bits_index(arc.from_inputs))
        indices.add(_bits_index(arc.to_inputs))
    return sorted(indices)


def _order_propagation_delay_arcs(
    arcs: Iterable[DigitalPropagationDelayArc],
) -> tuple[DigitalPropagationDelayArc, ...]:
    adjacency: dict[tuple[int, ...], list[DigitalPropagationDelayArc]] = {}
    for arc in arcs:
        adjacency.setdefault(arc.from_inputs, []).append(arc)
    for outgoing in adjacency.values():
        outgoing.sort(key=_propagation_delay_arc_sort_key, reverse=True)

    ordered: list[DigitalPropagationDelayArc] = []
    for start in sorted(adjacency):
        while adjacency.get(start):
            component: list[DigitalPropagationDelayArc] = []
            _visit_propagation_delay_arcs(start, adjacency, component)
            ordered.extend(reversed(component))
    return tuple(ordered)


def _bits_index(bits: Iterable[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _visit_propagation_delay_arcs(
    node: tuple[int, ...],
    adjacency: dict[tuple[int, ...], list[DigitalPropagationDelayArc]],
    component: list[DigitalPropagationDelayArc],
) -> None:
    outgoing = adjacency.get(node)
    while outgoing:
        arc = outgoing.pop()
        _visit_propagation_delay_arcs(arc.to_inputs, adjacency, component)
        component.append(arc)


def _propagation_delay_arc_sort_key(arc: DigitalPropagationDelayArc) -> tuple[int, str, str]:
    return (arc.input_index, bits_to_text(arc.from_inputs), bits_to_text(arc.to_inputs))
