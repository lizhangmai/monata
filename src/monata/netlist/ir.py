"""Simulator-neutral SPICE netlist IR for Monata."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

from monata.netlist.scope_state import ScopeState
from monata.netlist.scope_api import (
    ScopeDeviceApi,
    ScopeDirectiveApi,
    ScopeInstanceApi,
    ScopePassiveApi,
    ScopeSourceApi,
)


_ANY_NODES = -1
_AT_LEAST_TWO_NODES = -2
_UNSET = object()

_ELEMENT_DEFS = {
    "A": {"nodes": _AT_LEAST_TWO_NODES, "requires_model": True},
    "B": {"nodes": _AT_LEAST_TWO_NODES, "requires_value": True},
    "C": {"nodes": 2, "requires_value": True},
    "D": {"nodes": 2, "requires_model": True},
    "E": {"nodes": _ANY_NODES, "allowed_nodes": (2, 4), "requires_value": True},
    "F": {"nodes": 2, "requires_model": True, "requires_value": True},
    "G": {"nodes": _ANY_NODES, "allowed_nodes": (2, 4), "requires_value": True},
    "H": {"nodes": 2, "requires_model": True, "requires_value": True},
    "I": {"nodes": 2, "requires_value": True},
    "J": {"nodes": 3, "requires_model": True},
    "K": {"nodes": _ANY_NODES, "requires_value": True, "min_nodes": 2},
    "L": {"nodes": 2, "requires_value": True},
    "M": {"nodes": 4, "requires_model": True},
    "N": {"nodes": _ANY_NODES, "requires_model": True, "min_nodes": 1},
    "O": {"nodes": 4, "requires_model": True},
    "P": {"nodes": _ANY_NODES, "requires_model": True, "min_nodes": 2},
    "Q": {"nodes": _ANY_NODES, "requires_model": True, "min_nodes": 3},
    "R": {"nodes": 2, "requires_value": True},
    "S": {"nodes": 4, "requires_model": True},
    "T": {"nodes": 4, "requires_value": True},
    "U": {"nodes": 3, "requires_model": True},
    "V": {"nodes": 2, "requires_value": True},
    "W": {"nodes": 2, "requires_model": True, "requires_value": True},
    "X": {"nodes": _AT_LEAST_TWO_NODES, "requires_model": True},
    "Y": {"nodes": 4, "requires_model": True},
    "Z": {"nodes": 3, "requires_model": True},
}

SUPPORTED_KINDS = frozenset(_ELEMENT_DEFS)
_FIXED_SOURCE_VALUE_LENGTHS = {
    "DC": 1,
    "AC": 2,
    "PULSE": 7,
    "SIN": 5,
    "EXP": 6,
    "SFFM": 5,
    "AM": 5,
    "TRRANDOM": 5,
}
_TRRANDOM_DISTRIBUTIONS = {
    "uniform": 1,
    "gaussian": 2,
    "exponential": 3,
    "poisson": 4,
}


class NetlistError(ValueError):
    """Raised for invalid native netlist definitions."""


def _assert_single_line(value: Any, label: str) -> None:
    if isinstance(value, SourceValue):
        for item in value.values:
            _assert_single_line(item, label)
        for key, item in value.params.items():
            _assert_single_line(key, label)
            _assert_single_line(item, label)
        return
    text = str(value)
    if "\n" in text or "\r" in text:
        raise NetlistError(f"{label} cannot contain newlines")


def _validate_source_values(form: str, values: tuple[Any, ...]) -> None:
    if form == "PWL":
        if not values:
            raise NetlistError("PWL source value requires at least one time/value pair")
        if len(values) % 2:
            raise NetlistError("PWL source value expects time/value pairs")
        if any(isinstance(value, tuple | list) for value in values):
            raise NetlistError("PWL source value expects a flat time/value sequence")
        return
    expected = _FIXED_SOURCE_VALUE_LENGTHS.get(form)
    if expected is None:
        raise NetlistError(f"unsupported source value form: {form}")
    if len(values) != expected:
        raise NetlistError(f"{form} source value expects {expected} values, got {len(values)}")
    if form == "TRRANDOM" and (
        isinstance(values[0], bool)
        or not isinstance(values[0], int)
        or values[0] not in _TRRANDOM_DISTRIBUTIONS.values()
    ):
        supported = ", ".join(str(value) for value in _TRRANDOM_DISTRIBUTIONS.values())
        raise NetlistError(f"TRRANDOM source value expects distribution code {supported}")


def _validate_source_params(form: str, params: OrderedDict[str, Any]) -> None:
    if not params:
        return
    if form != "PWL":
        raise NetlistError(f"{form} source value does not support local parameters")
    supported = {"dc", "r", "td"}
    unsupported = [key for key in params if key.lower() not in supported]
    if unsupported:
        joined = ", ".join(unsupported)
        raise NetlistError(f"PWL source value has unsupported local parameter(s): {joined}")


def _pwl_values(points: tuple[Any, ...]) -> tuple[Any, ...]:
    if not points:
        raise NetlistError("PWL source value requires at least one time/value pair")
    has_pair_inputs = any(_is_pair(point) for point in points)
    if has_pair_inputs:
        if not all(_is_pair(point) for point in points):
            raise NetlistError("PWL source value expects time/value pairs")
        flattened: list[Any] = []
        for time, value in points:
            flattened.extend([time, value])
        return tuple(flattened)
    if len(points) % 2:
        raise NetlistError("PWL source value expects time/value pairs")
    return points


def _is_pair(value: Any) -> bool:
    return isinstance(value, tuple | list) and len(value) == 2


def _trrandom_values(
    distribution: str,
    duration: Any,
    delay: Any,
    parameter1: Any,
    parameter2: Any,
) -> tuple[Any, ...]:
    key = distribution.lower()
    try:
        distribution_code = _TRRANDOM_DISTRIBUTIONS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_TRRANDOM_DISTRIBUTIONS))
        raise NetlistError(f"unsupported TRRANDOM distribution: {distribution}; expected one of {supported}") from exc
    return (distribution_code, duration, delay, parameter1, parameter2)


@dataclass(frozen=True)
class SourceValue:
    """Structured source expression that keeps source metadata renderable."""

    form: str
    values: tuple[Any, ...]
    params: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        form = self.form.upper()
        _assert_single_line(form, "source value form")
        object.__setattr__(self, "form", form)
        _validate_source_values(form, self.values)
        for value in self.values:
            _assert_single_line(value, f"{form} source value")
        params = _params(dict(self.params))
        _validate_source_params(form, params)
        object.__setattr__(self, "params", params)
        for key, value in self.params.items():
            _assert_single_line(key, f"{form} source value parameter name")
            _assert_single_line(value, f"{form} source value parameter value")

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.params)

    def __getitem__(self, name: str) -> Any:
        key = name[:-1] if name.endswith("_") else name
        return self.params[key]

    def __getattr__(self, name: str) -> Any:
        key = name[:-1] if name.endswith("_") else name
        try:
            params = object.__getattribute__(self, "params")
            return params[key]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, name: str, default: Any = None) -> Any:
        key = name[:-1] if name.endswith("_") else name
        return self.params.get(key, default)

    def has_parameter(self, name: str) -> bool:
        key = name[:-1] if name.endswith("_") else name
        return key in self.params

    def clone(
        self,
        *,
        form: str | None = None,
        values: Iterable[Any] | None = None,
        **params: Any,
    ) -> "SourceValue":
        merged = OrderedDict(self.params)
        merged.update(_params(params))
        return type(self)(
            form=form if form is not None else self.form,
            values=tuple(values) if values is not None else self.values,
            params=merged,
        )

    def to_spice(self) -> str:
        """Return this source expression rendered as deterministic SPICE text."""

        from monata.netlist.ngspice import _format_value

        return _format_value(self)


def _pwl_source_value(
    points: tuple[Any, ...],
    *,
    repeat_time: Any | None = None,
    delay_time: Any | None = None,
    dc: Any | None = None,
) -> SourceValue:
    params: dict[str, Any] = {}
    if dc is not None:
        params["dc"] = dc
    if repeat_time is not None:
        params["r"] = repeat_time
    if delay_time is not None:
        params["td"] = delay_time
    return SourceValue("PWL", _pwl_values(points), _params(params))


@dataclass(frozen=True)
class Directive:
    """Simulator-neutral netlist directive record."""

    name: str
    args: tuple[Any, ...] = ()
    params: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    raw: bool = False

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise NetlistError("directive name is required")
        _assert_single_line(name, "directive name")
        if self.raw:
            object.__setattr__(self, "name", name)
            return
        name = name.lower().lstrip(".")
        if not name:
            raise NetlistError("directive name is required")
        object.__setattr__(self, "name", name)
        for arg in self.args:
            _assert_single_line(arg, f".{name} argument")
        for key, value in self.params.items():
            _assert_single_line(key, f".{name} parameter name")
            _assert_single_line(value, f".{name} parameter value")
        if name == "model" and (len(self.args) < 2 or not self.args[0] or not self.args[1]):
            raise NetlistError(".model requires name and type")
        if name == "lib" and (len(self.args) < 1 or not self.args[0]):
            raise NetlistError(".lib requires a path")
        if name in {"global", "save", "probe"} and not self.args:
            raise NetlistError(f".{name} requires at least one argument")
        if name in {"nodeset", "ic", "options"} and not self.params:
            raise NetlistError(f".{name} requires at least one parameter")
        if name == "print" and not self.args:
            raise NetlistError(".print requires an analysis")
        if name == "measure" and len(self.args) < 3:
            raise NetlistError(".measure requires analysis, name, and expression")

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.params)

    def __getitem__(self, name: str) -> Any:
        key = name[:-1] if name.endswith("_") else name
        return self.params[key]

    def __getattr__(self, name: str) -> Any:
        key = name[:-1] if name.endswith("_") else name
        try:
            params = object.__getattribute__(self, "params")
            return params[key]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, name: str, default: Any = None) -> Any:
        key = name[:-1] if name.endswith("_") else name
        return self.params.get(key, default)

    def has_parameter(self, name: str) -> bool:
        key = name[:-1] if name.endswith("_") else name
        return key in self.params

    def clone(
        self,
        *,
        name: str | None = None,
        args: Iterable[Any] | None = None,
        raw: bool | None = None,
        **params: Any,
    ) -> "Directive":
        merged = OrderedDict(self.params)
        merged.update(_params(params))
        return type(self)(
            name=self.name if name is None else name,
            args=tuple(args) if args is not None else self.args,
            params=merged,
            raw=self.raw if raw is None else raw,
        )

    def to_spice(self) -> str:
        """Return this directive rendered as one deterministic SPICE line."""

        from monata.netlist.ngspice import _render_directive

        return _render_directive(self)

    def __str__(self) -> str:
        return self.to_spice()


@dataclass(frozen=True)
class ModelCard:
    """Reusable SPICE model-card declaration."""

    name: str
    model_type: str
    params: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        model_type = str(self.model_type).strip()
        if not name or not model_type:
            raise NetlistError("model card requires name and type")
        _assert_single_line(name, "model card name")
        _assert_single_line(model_type, f"model card {name} type")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "params", _params(dict(self.params)))

    @classmethod
    def create(cls, name: str, model_type: str, **params: Any) -> "ModelCard":
        return cls(name, model_type, _params(params))

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.params)

    def clone(
        self,
        *,
        name: str | None = None,
        model_type: str | None = None,
        **params: Any,
    ) -> "ModelCard":
        merged = OrderedDict(self.params)
        merged.update(_params(params))
        return type(self)(name or self.name, model_type or self.model_type, merged)

    def __getitem__(self, name: str) -> Any:
        key = name[:-1] if name.endswith("_") else name
        return self.params[key]

    def __getattr__(self, name: str) -> Any:
        key = name[:-1] if name.endswith("_") else name
        try:
            params = object.__getattribute__(self, "params")
            return params[key]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, name: str, default: Any = None) -> Any:
        key = name[:-1] if name.endswith("_") else name
        return self.params.get(key, default)

    def has_parameter(self, name: str) -> bool:
        key = name[:-1] if name.endswith("_") else name
        return key in self.params

    def to_directive(self) -> Directive:
        return Directive("model", (self.name, self.model_type), OrderedDict(self.params))

    def to_spice(self) -> str:
        """Return this model card rendered as one deterministic SPICE line."""

        return self.to_directive().to_spice()

    def __str__(self) -> str:
        return self.to_spice()

    def apply(self, scope: Any) -> Directive:
        return scope.model(self)

    def copy_to(self, target: Any) -> Directive:
        """Copy this model card into a native netlist scope and return the directive."""

        if not hasattr(target, "model"):
            raise NetlistError("copy target must be a native netlist scope")
        return self.clone().apply(target)


@dataclass(frozen=True)
class Element:
    """Primitive SPICE element record."""

    kind: str
    name: str
    nodes: tuple[str, ...]
    value: str | int | float | SourceValue | None = None
    model: str | None = None
    params: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    comment: str | None = None
    enabled: bool = True
    raw_suffix: str | None = None

    def __post_init__(self) -> None:
        kind = self.kind.upper()
        object.__setattr__(self, "kind", kind)
        if kind not in SUPPORTED_KINDS:
            raise NetlistError(f"unsupported element kind: {self.kind}")
        if not self.name:
            raise NetlistError("element name is required")
        _assert_single_line(self.name, "element name")
        if any(not node for node in self.nodes):
            raise NetlistError(f"element {self.name} has an empty node")
        for node in self.nodes:
            _assert_single_line(node, f"{kind}{self.name} node")
        if self.value is not None:
            _assert_single_line(self.value, f"{kind}{self.name} value")
        if self.model is not None:
            _assert_single_line(self.model, f"{kind}{self.name} model")
        for key, value in self.params.items():
            _assert_single_line(key, f"{kind}{self.name} parameter name")
            _assert_single_line(value, f"{kind}{self.name} parameter value")
        if self.comment is not None:
            _assert_single_line(self.comment, f"{kind}{self.name} comment")
        if not isinstance(self.enabled, bool):
            raise NetlistError(f"{kind}{self.name} enabled flag must be a bool")
        if self.raw_suffix is not None:
            _assert_single_line(self.raw_suffix, f"{kind}{self.name} raw suffix")
            raw_suffix = str(self.raw_suffix).strip()
            object.__setattr__(self, "raw_suffix", raw_suffix or None)
        element_def = _ELEMENT_DEFS[kind]
        expected = element_def["nodes"]
        allowed = element_def.get("allowed_nodes")
        min_nodes = element_def.get("min_nodes")
        if allowed is not None and len(self.nodes) not in allowed:
            joined = " or ".join(str(count) for count in allowed)
            raise NetlistError(f"{kind}{self.name} expects {joined} nodes, got {len(self.nodes)}")
        if expected >= 0 and len(self.nodes) != expected:
            raise NetlistError(
                f"{kind}{self.name} expects {expected} nodes, got {len(self.nodes)}"
            )
        if expected == _AT_LEAST_TWO_NODES and len(self.nodes) < 2:
            raise NetlistError(f"{kind}{self.name} expects at least 2 nodes")
        if min_nodes is not None and len(self.nodes) < min_nodes:
            raise NetlistError(f"{kind}{self.name} expects at least {min_nodes} nodes")
        if element_def.get("requires_value") and self.value is None:
            raise NetlistError(f"{kind}{self.name} requires a value")
        if element_def.get("requires_model") and not self.model:
            raise NetlistError(f"{kind}{self.name} requires a model/subcircuit name")

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.params)

    def __getitem__(self, name: str) -> Any:
        key = name[:-1] if name.endswith("_") else name
        return self.params[key]

    def __getattr__(self, name: str) -> Any:
        key = name[:-1] if name.endswith("_") else name
        try:
            params = object.__getattribute__(self, "params")
            return params[key]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, name: str, default: Any = None) -> Any:
        key = name[:-1] if name.endswith("_") else name
        return self.params.get(key, default)

    def has_parameter(self, name: str) -> bool:
        key = name[:-1] if name.endswith("_") else name
        return key in self.params

    def to_spice(self) -> str:
        """Return this element rendered as one deterministic SPICE line."""

        from monata.netlist.ngspice import _render_element

        return _render_element(self)

    def __str__(self) -> str:
        return self.to_spice()

    def copy_to(self, target: Any) -> Element:
        """Copy this element into a native netlist scope and return the copy."""

        if not hasattr(target, "add"):
            raise NetlistError("copy target must be a native netlist scope")
        return target.add(self.clone())

    def clone(
        self,
        *,
        kind: str | None = None,
        name: str | None = None,
        nodes: Iterable[str] | None = None,
        value: Any = _UNSET,
        model: Any = _UNSET,
        comment: Any = _UNSET,
        enabled: Any = _UNSET,
        raw_suffix: Any = _UNSET,
        **params: Any,
    ) -> "Element":
        """Return a modified copy of this immutable element record."""

        merged = OrderedDict(self.params)
        merged.update(_params(params))
        return type(self)(
            kind=kind if kind is not None else self.kind,
            name=name if name is not None else self.name,
            nodes=tuple(str(node) for node in nodes) if nodes is not None else self.nodes,
            value=self.value if value is _UNSET else value,
            model=self.model if model is _UNSET else model,
            params=merged,
            comment=self.comment if comment is _UNSET else comment,
            enabled=self.enabled if enabled is _UNSET else enabled,
            raw_suffix=self.raw_suffix if raw_suffix is _UNSET else raw_suffix,
        )


class _Scope(ScopeDirectiveApi, ScopePassiveApi, ScopeSourceApi, ScopeDeviceApi, ScopeInstanceApi):
    def __init__(self) -> None:
        self._state = ScopeState(NetlistError)
        self.includes: list[str] = self._state.includes
        self.params: OrderedDict[str, Any] = self._state.params
        self.directives: list[Directive] = self._state.directives
        self.elements: list[Element] = self._state.elements
        self.explicit_nodes: list[str] = self._state.explicit_nodes
        self.pdk_instances: list[Any] = self._state.pdk_instances.items

    def add(self, element: Element) -> Element:
        return self._state.add_element(element)

    def get_element(self, name: str, kind: str | None = None) -> Element | None:
        return self._state.get_element(name, kind=kind)

    def element(self, name: str, kind: str | None = None) -> Element:
        element = self.get_element(name, kind=kind)
        if element is None:
            raise NetlistError(f"element not found in scope: {name}")
        return element

    def remove_element(self, element: str | Element, kind: str | None = None) -> Element:
        target = element if isinstance(element, Element) else self.element(element, kind=kind)
        return self._state.remove_element(target)

    def replace_element(
        self,
        element: str | Element,
        replacement: Element,
        kind: str | None = None,
    ) -> Element:
        if not isinstance(replacement, Element):
            raise NetlistError("replacement must be an Element")
        target = element if isinstance(element, Element) else self.element(element, kind=kind)
        return self._state.replace_element(target, replacement)

    def copy_to(self, target: Any) -> Any:
        if not isinstance(target, _Scope):
            raise NetlistError("copy target must be a native netlist scope")
        target.includes[:] = deepcopy(self.includes)
        target.params.clear()
        target.params.update(deepcopy(self.params))
        target.directives[:] = deepcopy(self.directives)
        target.elements[:] = deepcopy(self.elements)
        target.explicit_nodes[:] = deepcopy(self.explicit_nodes)
        target.pdk_instances[:] = deepcopy(self.pdk_instances)
        target._rebuild_element_indexes()
        return target

    def __getitem__(self, name: str) -> Element | ModelCard | str:
        element = self.get_element(name)
        if element is not None:
            return element
        model_cards = _model_cards_named(self.model_cards, name)
        if len(model_cards) == 1:
            return model_cards[0]
        if model_cards:
            raise NetlistError(f"ambiguous model card in scope: {name}")
        node = self.get_node(name)
        if node is not None:
            return node
        raise NetlistError(f"item not found in scope: {name}")

    def __getattr__(self, name: str) -> Element | ModelCard | str:
        if "_state" not in self.__dict__:
            raise AttributeError(name)
        try:
            return self[name]
        except NetlistError:
            pass
        raise AttributeError(name)

    @property
    def gnd(self) -> str:
        return "0"

    @property
    def ground_node(self) -> str:
        return self.gnd

    @property
    def node_names(self) -> tuple[str, ...]:
        return _scope_node_names(self)

    @property
    def node_objects(self) -> tuple[Any, ...]:
        topology = _scope_topology(self)
        return tuple(topology.node(node) for node in self.node_names)

    def get_node(self, name: Any, create: bool = False) -> str | None:
        key = str(name).lower()
        for node in self.node_names:
            if node.lower() == key:
                return node
        if create:
            return self._add_node(name)
        return None

    def get_node_object(self, name: Any, create: bool = False) -> Any | None:
        node = self.get_node(name, create=create)
        if node is None:
            return None
        return _scope_topology(self).node(node)

    def node(self, name: Any, create: bool = False) -> str:
        node = self.get_node(name, create=create)
        if node is None:
            raise NetlistError(f"node not found in scope: {name}")
        return node

    def node_object(self, name: Any, create: bool = False) -> Any:
        node = self.get_node_object(name, create=create)
        if node is None:
            raise NetlistError(f"node not found in scope: {name}")
        return node

    def has_node(self, name: Any) -> bool:
        return self.get_node(name) is not None

    @property
    def has_ground_node(self) -> bool:
        return any(_is_ground_node(node) for node in self.node_names)

    @property
    def element_names(self) -> tuple[str, ...]:
        return tuple(instance_name(element.kind, element.name) for element in self.elements)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(str(directive.args[0]) for directive in self.directives if _is_model_directive(directive))

    @property
    def model_cards(self) -> tuple[ModelCard, ...]:
        return tuple(
            ModelCard(str(directive.args[0]), str(directive.args[1]), OrderedDict(directive.params))
            for directive in self.directives
            if _is_representable_model_directive(directive)
        )

    @property
    def models(self) -> tuple[ModelCard, ...]:
        return self.model_cards

    def get_model_card(self, name: str) -> ModelCard | None:
        matches = _model_cards_named(self.model_cards, name)
        if len(matches) == 1:
            return matches[0]
        return None

    def model_card_by_name(self, name: str) -> ModelCard:
        matches = _model_cards_named(self.model_cards, name)
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise NetlistError(f"ambiguous model card in scope: {name}")
        raise NetlistError(f"model card not found in scope: {name}")

    def to_topology(self, *, title: str | None = None) -> Any:
        """Return an editable topology view of this scope's elements."""

        from monata.netlist.topology import Topology

        return Topology.from_scope(self, title=title)

    def apply_topology(self, topology: Any) -> Any:
        """Replace this scope's elements from an editable topology projection."""

        projected = topology.to_circuit()
        previous = list(self.elements)
        self.elements[:] = list(projected.elements)
        try:
            self._rebuild_element_indexes()
        except Exception:
            self.elements[:] = previous
            self._rebuild_element_indexes()
            raise
        return self

    def _rebuild_element_indexes(self) -> None:
        self._state.rebuild_element_index()

    def _add_pdk_instance(self, instance: Any) -> Any:
        return self._state.add_pdk_instance(instance)

    def _add_node(self, name: Any) -> str:
        text = str(name)
        if not text:
            raise NetlistError("node name is required")
        _assert_single_line(text, "node name")
        for node in self.node_names:
            if node.lower() == text.lower():
                return node
        self.explicit_nodes.append(text)
        return text


class SubCircuit(_Scope):
    """Reusable native subcircuit definition."""

    NAME: str | None = None
    NODES: tuple[str, ...] = ()

    def __init__(self, name: str | None = None, nodes: Iterable[str] | None = None) -> None:
        super().__init__()
        self.name = name or self.subckt_name()
        if not self.name:
            raise NetlistError("subcircuit name is required")
        _assert_single_line(self.name, "subcircuit name")
        self.nodes = _validate_subcircuit_nodes(self.name, nodes if nodes is not None else self.NODES)
        self._built = False

    @property
    def external_nodes(self) -> tuple[str, ...]:
        """Return this subcircuit's external port names."""

        return self.nodes

    @classmethod
    def subckt_name(cls) -> str:
        return cls.NAME or cls.__name__.lower()

    def build(self) -> None:
        """Override in subclasses to populate the subcircuit."""

    def ensure_built(self) -> "SubCircuit":
        if not self._built:
            self.build()
            self._built = True
        return self

    def to_spice(self) -> str:
        """Return this subcircuit rendered as deterministic SPICE text."""

        from monata.netlist.ngspice import render_ngspice

        return render_ngspice(self)

    def __str__(self) -> str:
        return self.to_spice()

    def clone(
        self,
        *,
        name: str | None = None,
        nodes: Iterable[str] | None = None,
    ) -> "SubCircuit":
        """Return an independent copy of this subcircuit definition."""

        clone = deepcopy(self)
        if name is not None:
            name_text = str(name)
            if not name_text:
                raise NetlistError("subcircuit name is required")
            _assert_single_line(name_text, "subcircuit name")
            clone.name = name_text
        if nodes is not None:
            clone.nodes = _validate_subcircuit_nodes(clone.name, nodes)
        return clone


class Circuit(_Scope):
    """Top-level native circuit definition."""

    def __init__(self, title: str = "Monata circuit") -> None:
        super().__init__()
        _assert_single_line(title, "circuit title")
        self.title = title
        self.subcircuits: list[SubCircuit] = []
        self.outputs: list[str] = []

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.node_names

    def copy_to(self, target: Any) -> Any:
        copied = super().copy_to(target)
        if isinstance(target, Circuit):
            target.subcircuits[:] = deepcopy(self.subcircuits)
            target.outputs[:] = list(self.outputs)
        return copied

    def subckt(self, subcircuit: type[SubCircuit] | SubCircuit) -> SubCircuit:
        if isinstance(subcircuit, type) and issubclass(subcircuit, SubCircuit):
            instance = subcircuit()
        elif isinstance(subcircuit, SubCircuit):
            instance = subcircuit
        else:
            raise NetlistError("subcircuit must be a SubCircuit class or instance")
        instance.ensure_built()
        names = {subckt.name.lower() for subckt in self.subcircuits}
        if instance.name.lower() in names:
            raise NetlistError(f"duplicate subcircuit name: {instance.name}")
        self.subcircuits.append(instance)
        return instance

    @property
    def subcircuit_names(self) -> tuple[str, ...]:
        return tuple(subcircuit.name for subcircuit in self.subcircuits)

    def get_subcircuit(self, name: str) -> SubCircuit | None:
        key = str(name).lower()
        for subcircuit in self.subcircuits:
            if subcircuit.name.lower() == key:
                return subcircuit
        return None

    def subcircuit(self, subcircuit: str | type[SubCircuit] | SubCircuit) -> SubCircuit:
        if isinstance(subcircuit, SubCircuit) or (isinstance(subcircuit, type) and issubclass(subcircuit, SubCircuit)):
            return self.subckt(subcircuit)
        name = str(subcircuit)
        found = self.get_subcircuit(name)
        if found is None:
            raise NetlistError(f"subcircuit not found in circuit: {name}")
        return found

    def to_spice(self) -> str:
        """Return this circuit rendered as deterministic SPICE text."""

        from monata.netlist.ngspice import render_ngspice

        return render_ngspice(self)

    def __str__(self) -> str:
        return self.to_spice()

    def simulator(
        self,
        simulator: str | None = None,
        *,
        output_names: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        backend_options: dict[str, Any] | None = None,
        artifacts: Any = None,
        snapshot_tasks: bool | None = None,
        timeout: Any = _UNSET,
        temperature: Any = _UNSET,
        nominal_temperature: Any = _UNSET,
    ) -> Any:
        """Return a backend-neutral simulation session for this circuit."""

        from monata.sim.session import SimulationSession

        kwargs: dict[str, Any] = {"output_names": output_names}
        if simulator is not None:
            kwargs["simulator"] = simulator
        if metadata is not None:
            kwargs["metadata"] = metadata
        if backend_options is not None:
            kwargs["backend_options"] = backend_options
        if artifacts is not None:
            kwargs["artifacts"] = artifacts
        if snapshot_tasks is not None:
            kwargs["snapshot_tasks"] = snapshot_tasks
        if timeout is not _UNSET:
            kwargs["timeout"] = timeout
        if temperature is not _UNSET:
            kwargs["temperature"] = temperature
        if nominal_temperature is not _UNSET:
            kwargs["nominal_temperature"] = nominal_temperature
        return SimulationSession(self, **kwargs)

    def output(self, line: str) -> None:
        if not line:
            raise NetlistError("output line is required")
        _assert_single_line(line, "output line")
        self.outputs.append(line)

    def clone(self, *, title: str | None = None) -> "Circuit":
        """Return an independent copy of this circuit definition."""

        clone = deepcopy(self)
        if title is not None:
            _assert_single_line(title, "circuit title")
            clone.title = title
        return clone


def _element(
    kind: str,
    name: str,
    nodes: tuple[str, ...],
    value: Any = None,
    model: str | None = None,
    params: dict[str, Any] | None = None,
) -> Element:
    return Element(
        kind=kind,
        name=name,
        nodes=tuple(str(node) for node in nodes),
        value=value,
        model=model,
        params=_params(params or {}),
    )


def _params(params: dict[str, Any]) -> OrderedDict[str, Any]:
    result: OrderedDict[str, Any] = OrderedDict()
    for key, value in params.items():
        text = str(key)
        if text.endswith("_"):
            text = text[:-1]
        _assert_single_line(text, "parameter name")
        _assert_single_line(value, f"parameter {text} value")
        result[text] = value
    return result


def _validate_subcircuit_nodes(name: str, nodes: Iterable[Any]) -> tuple[str, ...]:
    external_nodes = tuple(str(node) for node in nodes)
    if not external_nodes:
        raise NetlistError(f"subcircuit {name} requires external nodes")
    if any(not node for node in external_nodes):
        raise NetlistError(f"subcircuit {name} has an empty external node")
    seen: set[str] = set()
    for node in external_nodes:
        _assert_single_line(node, f"subcircuit {name} external node")
        key = node.lower()
        if key in seen:
            raise NetlistError(f"subcircuit {name} has duplicate external node: {node}")
        seen.add(key)
    return external_nodes


def instance_name(kind: str, name: str) -> str:
    if name.startswith(kind):
        return name
    return f"{kind}{name}"


def _is_model_directive(directive: Directive) -> bool:
    return not directive.raw and directive.name == "model" and len(directive.args) >= 2


def _is_representable_model_directive(directive: Directive) -> bool:
    return _is_model_directive(directive) and len(directive.args) == 2


def _model_cards_named(cards: tuple[ModelCard, ...], name: str) -> tuple[ModelCard, ...]:
    key = str(name).lower()
    return tuple(card for card in cards if card.name.lower() == key)


def _scope_node_names(scope: Any) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    if isinstance(scope, SubCircuit):
        for node in scope.nodes:
            _append_node_name(names, seen, node)
    for element in scope.elements:
        for node in element.nodes:
            _append_node_name(names, seen, node)
    for node in getattr(scope, "explicit_nodes", ()):
        _append_node_name(names, seen, node)
    return tuple(names)


def _scope_topology(scope: Any) -> Any:
    from monata.netlist.topology import Topology

    topology = Topology.from_scope(scope, title=getattr(scope, "title", None))
    for node in scope.node_names:
        topology.node(node)
    return topology


def _append_node_name(names: list[str], seen: set[str], node: Any) -> None:
    text = str(node)
    key = text.lower()
    if key in seen:
        return
    seen.add(key)
    names.append(text)


def _is_ground_node(node: str) -> bool:
    text = str(node)
    return text == "0" or text.lower() == "gnd"
