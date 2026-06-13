"""Optional editable topology graph for native netlists."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable

from monata.netlist.device_schema import DeviceSchemaError, element_spec
from monata.netlist.ir import Circuit, Element


class TopologyError(ValueError):
    """Raised for invalid topology mutations."""


class Node:
    """A named topology node with connected pins."""

    def __init__(self, topology: Topology, name: str) -> None:
        if not name:
            raise TopologyError("node name is required")
        self.topology = topology
        self.name = str(name)
        self._pins: list[Pin] = []

    @property
    def pins(self) -> tuple[Pin, ...]:
        return tuple(self._pins)

    @property
    def is_ground_node(self) -> bool:
        return self.name == "0" or self.name.lower() == "gnd"

    def __bool__(self) -> bool:
        return bool(self._pins)

    def __len__(self) -> int:
        return len(self._pins)

    def __iter__(self):
        return iter(self.pins)

    def __contains__(self, pin: Pin) -> bool:
        return pin in self._pins

    def connect(self, pin: Pin) -> None:
        if pin not in self._pins:
            self._pins.append(pin)

    def disconnect(self, pin: Pin) -> None:
        if pin in self._pins:
            self._pins.remove(pin)

    def merge(self, other: Node) -> Node:
        if self.topology is not other.topology:
            raise TopologyError("cannot merge nodes from different topologies")
        if other is self:
            return self
        for pin in list(other.pins):
            pin.connect(self)
        self.topology._delete_node(other.name)
        return self

    def __iadd__(self, other: Node | Pin | Iterable[Node | Pin]) -> Node:
        items = other if isinstance(other, (list, tuple)) else (other,)
        for item in items:
            if isinstance(item, Node):
                self.merge(item)
            elif isinstance(item, Pin):
                node = item.node
                if node is not None and node is not self:
                    self.merge(node)
                else:
                    item.connect(self)
            else:
                raise TopologyError(f"cannot connect node to {type(item).__name__}")
        return self

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Node({self.name!r})"


class Pin:
    """A topology element pin that can be reconnected."""

    def __init__(self, element: TopologyElement, index: int, name: str, node: Node | None) -> None:
        self.element = element
        self.index = index
        self.name = name
        self._node: Node | None = None
        if node is not None:
            self.connect(node)

    @property
    def node(self) -> Node | None:
        return self._node

    @property
    def connected(self) -> bool:
        return self._node is not None

    @property
    def dangling(self) -> bool:
        return self._node is None

    def connect(self, node: Node | str) -> None:
        target = self.element.topology.node(node) if isinstance(node, str) else node
        if target.topology is not self.element.topology:
            raise TopologyError("cannot connect pin to a node from another topology")
        self.disconnect()
        target.connect(self)
        self._node = target

    def disconnect(self) -> None:
        if self._node is not None:
            self._node.disconnect(self)
            self._node = None

    def add_current_probe(self, name: str | None = None) -> TopologyElement:
        probe_name = name or f"{self.element.name}_{self.name}_probe"
        return self.element.topology.insert_series(self, "V", probe_name, value=0)

    def add_esr(self, name: str | None = None, value: Any = "1e-3") -> TopologyElement:
        resistor_name = name or f"{self.element.name}_{self.name}_esr"
        return self.element.topology.insert_series(self, "R", resistor_name, value=value)

    def __iadd__(self, other: Node | Pin) -> Pin:
        if isinstance(other, Node):
            self.connect(other)
            return self
        if isinstance(other, Pin):
            if self.connected and other.connected:
                self.node.merge(other.node)  # type: ignore[union-attr]
            elif self.connected:
                other.connect(self.node)  # type: ignore[arg-type]
            elif other.connected:
                self.connect(other.node)  # type: ignore[arg-type]
            else:
                shared = self.element.topology.node(f"{self.element.name}_{self.name}")
                self.connect(shared)
                other.connect(shared)
            return self
        raise TopologyError(f"cannot connect pin to {type(other).__name__}")

    def __repr__(self) -> str:
        node = self._node.name if self._node is not None else None
        return f"Pin({self.element.name!r}, {self.name!r}, node={node!r})"


class TopologyElement:
    """Editable topology element with named pins."""

    def __init__(
        self,
        topology: Topology,
        kind: str,
        name: str,
        nodes: Iterable[str | Node],
        *,
        value: Any = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
        pin_names: Iterable[str] | None = None,
    ) -> None:
        self.topology = topology
        self.kind = kind.upper()
        self.name = str(name)
        self.value = value
        self.model = model
        self.params = OrderedDict(params or {})
        node_objects = tuple(topology.node(node) if isinstance(node, str) else node for node in nodes)
        self._pins = tuple(
            Pin(self, index, pin_name, node)
            for index, (pin_name, node) in enumerate(zip(_pin_names(self.kind, len(node_objects), pin_names), node_objects))
        )

    @property
    def pins(self) -> tuple[Pin, ...]:
        return self._pins

    @property
    def nodes(self) -> tuple[str, ...]:
        if any(pin.node is None for pin in self._pins):
            raise TopologyError(f"element {self.name} has dangling pins")
        return tuple(pin.node.name for pin in self._pins if pin.node is not None)

    def pin(self, name_or_index: str | int) -> Pin:
        if isinstance(name_or_index, int):
            return self._pins[name_or_index]
        for pin in self._pins:
            if pin.name == name_or_index:
                return pin
        raise TopologyError(f"unknown pin {name_or_index!r} on element {self.name}")

    def series_with(self, other: TopologyElement, *, node: str | Node | None = None) -> TopologyElement:
        """Connect this two-pin element in series with another two-pin element."""
        self._require_same_topology(other)
        _, output_pin = self._dipole_pins()
        input_pin, _ = other._dipole_pins()
        if node is None:
            output_pin += input_pin
        else:
            shared_node = self.topology.node(node)
            output_pin += shared_node
            input_pin += shared_node
        return other

    def parallel_with(self, other: TopologyElement) -> TopologyElement:
        """Connect another two-pin element across the same endpoints as this element."""
        self._require_same_topology(other)
        positive_pin, negative_pin = self._dipole_pins()
        other_positive_pin, other_negative_pin = other._dipole_pins()
        positive_pin += other_positive_pin
        negative_pin += other_negative_pin
        return self

    def to_element(self) -> Element:
        return Element(self.kind, self.name, self.nodes, value=self.value, model=self.model, params=self.params)

    def _require_same_topology(self, other: TopologyElement) -> None:
        if self.topology is not other.topology:
            raise TopologyError("elements must belong to the same topology")

    def _dipole_pins(self) -> tuple[Pin, Pin]:
        if len(self._pins) != 2:
            raise TopologyError(f"element {self.name} is not a two-pin element")
        named_pins = {pin.name: pin for pin in self._pins}
        if "p" in named_pins and "n" in named_pins:
            return named_pins["p"], named_pins["n"]
        return self._pins[0], self._pins[1]


class Topology:
    """Mutable graph that can be projected back to Monata's record IR."""

    def __init__(self, title: str = "Monata topology") -> None:
        self.title = title
        self._nodes: OrderedDict[str, Node] = OrderedDict()
        self._elements: OrderedDict[str, TopologyElement] = OrderedDict()

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes.values())

    @property
    def elements(self) -> tuple[TopologyElement, ...]:
        return tuple(self._elements.values())

    def node(self, name: str | Node) -> Node:
        if isinstance(name, Node):
            if name.topology is not self:
                raise TopologyError("node belongs to another topology")
            return name
        key = str(name)
        if key not in self._nodes:
            self._nodes[key] = Node(self, key)
        return self._nodes[key]

    def element(self, name: str) -> TopologyElement:
        try:
            return self._elements[name]
        except KeyError as exc:
            raise TopologyError(f"element not found: {name}") from exc

    def add_element(
        self,
        kind: str,
        name: str,
        nodes: Iterable[str | Node],
        *,
        value: Any = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
        pin_names: Iterable[str] | None = None,
    ) -> TopologyElement:
        if name in self._elements:
            raise TopologyError(f"duplicate element name: {name}")
        element = TopologyElement(
            self,
            kind,
            name,
            nodes,
            value=value,
            model=model,
            params=params,
            pin_names=pin_names,
        )
        Element(element.kind, element.name, element.nodes, value=value, model=model, params=OrderedDict(params or {}))
        self._elements[element.name] = element
        return element

    def add(self, element: Element, *, pin_names: Iterable[str] | None = None) -> TopologyElement:
        return self.add_element(
            element.kind,
            element.name,
            element.nodes,
            value=element.value,
            model=element.model,
            params=element.params,
            pin_names=pin_names,
        )

    def insert_series(self, pin: Pin, kind: str, name: str, *, value: Any) -> TopologyElement:
        if pin.element.topology is not self:
            raise TopologyError("pin belongs to another topology")
        if pin.node is None:
            raise TopologyError("cannot insert series element on a dangling pin")
        original_node = pin.node
        inserted_node = self.node(self._unique_node_name(f"{pin.element.name}_{pin.name}"))
        pin.connect(inserted_node)
        return self.add_element(kind, name, (original_node, inserted_node), value=value)

    def dangling_pins(self) -> tuple[Pin, ...]:
        return tuple(pin for element in self._elements.values() for pin in element.pins if pin.dangling)

    def to_circuit(self) -> Circuit:
        if self.dangling_pins():
            names = ", ".join(f"{pin.element.name}.{pin.name}" for pin in self.dangling_pins())
            raise TopologyError(f"topology contains dangling pins: {names}")
        circuit = Circuit(self.title)
        for element in self._elements.values():
            circuit.add(element.to_element())
        return circuit

    @classmethod
    def from_scope(cls, scope: Any, *, title: str | None = None) -> Topology:
        topology = cls(title or getattr(scope, "title", "Monata topology"))
        for element in getattr(scope, "elements", ()):
            topology.add(element)
        return topology

    def _delete_node(self, name: str) -> None:
        self._nodes.pop(name, None)

    def _unique_node_name(self, prefix: str) -> str:
        candidate = prefix
        index = 1
        while candidate in self._nodes:
            index += 1
            candidate = f"{prefix}_{index}"
        return candidate


def _pin_names(kind: str, count: int, explicit: Iterable[str] | None) -> tuple[str, ...]:
    if explicit is not None:
        names = tuple(str(name) for name in explicit)
        if len(names) != count:
            raise TopologyError(f"expected {count} pin names, got {len(names)}")
        return names
    try:
        spec = element_spec(kind)
    except DeviceSchemaError:
        return tuple(f"n{index + 1}" for index in range(count))
    spec_names = tuple(pin.name for pin in spec.pins)
    if len(spec_names) == count:
        return spec_names
    return tuple(f"n{index + 1}" for index in range(count))


__all__ = [
    "Node",
    "Pin",
    "Topology",
    "TopologyElement",
    "TopologyError",
]
