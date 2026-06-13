"""Device and element schema records for native netlist authoring."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable


class DeviceSchemaError(ValueError):
    """Raised for invalid element schema usage."""


@dataclass(frozen=True)
class PinSpec:
    """Named element pin metadata."""

    name: str
    aliases: tuple[str, ...] = ()
    optional: bool = False


@dataclass(frozen=True)
class ElementParameterSpec:
    """Named element parameter metadata."""

    name: str
    spice_name: str | None = None
    aliases: tuple[str, ...] = ()
    unit: str | None = None
    required: bool = False
    default: Any = None
    description: str | None = None

    def matches(self, name: str) -> bool:
        candidates = {self.name, *(self.aliases)}
        if self.spice_name is not None:
            candidates.add(self.spice_name)
        return name.lower() in {candidate.lower() for candidate in candidates}


@dataclass(frozen=True)
class ElementSpec:
    """Element shape plus parameter metadata."""

    kind: str
    description: str
    pins: tuple[PinSpec, ...] = ()
    min_nodes: int | None = None
    max_nodes: int | None = None
    requires_value: bool = False
    requires_model: bool = False
    value_name: str | None = None
    model_name: str | None = None
    parameters: tuple[ElementParameterSpec, ...] = ()

    def __post_init__(self) -> None:
        kind = self.kind.upper()
        if len(kind) != 1 or not kind.isalpha():
            raise DeviceSchemaError(f"invalid element kind: {self.kind}")
        object.__setattr__(self, "kind", kind)
        if self.min_nodes is None and self.pins:
            object.__setattr__(self, "min_nodes", sum(not pin.optional for pin in self.pins))
        if self.max_nodes is None and self.pins:
            object.__setattr__(self, "max_nodes", len(self.pins))

    @property
    def fixed_node_count(self) -> int | None:
        if self.min_nodes is not None and self.min_nodes == self.max_nodes:
            return self.min_nodes
        return None

    def validate_nodes(self, nodes: Iterable[str]) -> tuple[str, ...]:
        result = tuple(str(node) for node in nodes)
        if self.min_nodes is not None and len(result) < self.min_nodes:
            raise DeviceSchemaError(f"{self.kind} expects at least {self.min_nodes} nodes")
        if self.max_nodes is not None and len(result) > self.max_nodes:
            raise DeviceSchemaError(f"{self.kind} expects at most {self.max_nodes} nodes")
        if any(not node for node in result):
            raise DeviceSchemaError(f"{self.kind} contains an empty node")
        return result

    def parameter(self, name: str) -> ElementParameterSpec | None:
        for parameter in self.parameters:
            if parameter.matches(name):
                return parameter
        return None

    def canonical_parameter_name(self, name: str) -> str:
        parameter = self.parameter(name)
        return parameter.name if parameter is not None else name

    def normalize_params(self, params: dict[str, Any]) -> OrderedDict[str, Any]:
        result: OrderedDict[str, Any] = OrderedDict()
        for key, value in params.items():
            result[self.canonical_parameter_name(str(key))] = value
        missing = [
            parameter.name
            for parameter in self.parameters
            if parameter.required and parameter.name not in result and parameter.default is None
        ]
        if missing:
            joined = ", ".join(missing)
            raise DeviceSchemaError(f"{self.kind} missing required parameter(s): {joined}")
        return result


@dataclass
class DeviceSchemaRegistry:
    """Registry for element schema records."""

    _specs: dict[str, ElementSpec] = field(default_factory=dict)

    def register(self, spec: ElementSpec, *, replace: bool = False) -> None:
        if spec.kind in self._specs and not replace:
            raise DeviceSchemaError(f"element schema already registered: {spec.kind}")
        self._specs[spec.kind] = spec

    def get(self, kind: str) -> ElementSpec:
        key = kind.upper()
        try:
            return self._specs[key]
        except KeyError as exc:
            raise DeviceSchemaError(f"unknown element kind: {kind}") from exc

    def supports(self, kind: str) -> bool:
        return kind.upper() in self._specs

    def list(self) -> list[str]:
        return sorted(self._specs)


def default_device_schema() -> DeviceSchemaRegistry:
    registry = DeviceSchemaRegistry()
    for spec in DEFAULT_ELEMENT_SPECS:
        registry.register(spec)
    return registry


def element_spec(kind: str) -> ElementSpec:
    return _DEFAULT_SCHEMA.get(kind)


def normalize_element_params(kind: str, params: dict[str, Any]) -> OrderedDict[str, Any]:
    return element_spec(kind).normalize_params(params)


def _pins(*names: str) -> tuple[PinSpec, ...]:
    return tuple(PinSpec(name) for name in names)


def _params(*params: ElementParameterSpec) -> tuple[ElementParameterSpec, ...]:
    return params


DEFAULT_ELEMENT_SPECS = (
    ElementSpec("A", "XSPICE code model", min_nodes=2, requires_model=True, model_name="model"),
    ElementSpec(
        "B",
        "Behavioral arbitrary source",
        min_nodes=2,
        requires_value=True,
        value_name="expression",
        parameters=_params(
            ElementParameterSpec("current_expression", "i"),
            ElementParameterSpec("voltage_expression", "v"),
            ElementParameterSpec("temperature_coefficient_1", "tc1"),
            ElementParameterSpec("temperature_coefficient_2", "tc2"),
            ElementParameterSpec("temperature", "temp", unit="degC"),
            ElementParameterSpec("device_temperature", "dtemp", unit="degC"),
        ),
    ),
    ElementSpec(
        "C",
        "Capacitor",
        pins=_pins("p", "n"),
        requires_value=True,
        value_name="capacitance",
        model_name="model",
        parameters=_params(
            ElementParameterSpec("length", "l", unit="m"),
            ElementParameterSpec("width", "w", unit="m"),
            ElementParameterSpec("multiplier", "m"),
            ElementParameterSpec("scale", "scale"),
            ElementParameterSpec("temperature", "temp", unit="degC"),
            ElementParameterSpec("device_temperature", "dtemp", unit="degC"),
            ElementParameterSpec("initial_condition", "ic", unit="V"),
            ElementParameterSpec("temperature_coefficient_1", "tc1"),
            ElementParameterSpec("temperature_coefficient_2", "tc2"),
        ),
    ),
    ElementSpec(
        "D",
        "Diode",
        pins=_pins("anode", "cathode"),
        requires_model=True,
        model_name="model",
        parameters=_params(
            ElementParameterSpec("area", "area"),
            ElementParameterSpec("multiplier", "m"),
            ElementParameterSpec("junction_perimeter", "pj"),
            ElementParameterSpec("off", "off"),
            ElementParameterSpec("initial_condition", "ic"),
            ElementParameterSpec("temperature", "temp", unit="degC"),
            ElementParameterSpec("device_temperature", "dtemp", unit="degC"),
        ),
    ),
    ElementSpec("E", "Voltage/nonlinear controlled voltage source", min_nodes=2, max_nodes=4, requires_value=True),
    ElementSpec(
        "F",
        "Current-controlled current source",
        pins=_pins("p", "n"),
        requires_model=True,
        requires_value=True,
        parameters=_params(ElementParameterSpec("multiplier", "m")),
    ),
    ElementSpec(
        "G",
        "Voltage-controlled current source",
        min_nodes=2,
        max_nodes=4,
        requires_value=True,
        parameters=_params(ElementParameterSpec("multiplier", "m")),
    ),
    ElementSpec("H", "Current-controlled voltage source", pins=_pins("p", "n"), requires_model=True, requires_value=True),
    ElementSpec("I", "Current source", pins=_pins("p", "n"), requires_value=True, value_name="current"),
    ElementSpec(
        "J",
        "JFET",
        pins=_pins("drain", "gate", "source"),
        requires_model=True,
        model_name="model",
        parameters=_params(
            ElementParameterSpec("area", "area"),
            ElementParameterSpec("multiplier", "m"),
            ElementParameterSpec("off", "off"),
            ElementParameterSpec("initial_condition", "ic"),
            ElementParameterSpec("temperature", "temp", unit="degC"),
        ),
    ),
    ElementSpec(
        "K",
        "Coupled inductors",
        pins=(PinSpec("inductor1"), PinSpec("inductor2")),
        requires_value=True,
        value_name="coupling",
    ),
    ElementSpec(
        "L",
        "Inductor",
        pins=_pins("p", "n"),
        requires_value=True,
        value_name="inductance",
        model_name="model",
        parameters=_params(
            ElementParameterSpec("turns_ratio", "nt"),
            ElementParameterSpec("multiplier", "m"),
            ElementParameterSpec("scale", "scale"),
            ElementParameterSpec("temperature", "temp", unit="degC"),
            ElementParameterSpec("device_temperature", "dtemp", unit="degC"),
            ElementParameterSpec("initial_condition", "ic", unit="A"),
            ElementParameterSpec("temperature_coefficient_1", "tc1"),
            ElementParameterSpec("temperature_coefficient_2", "tc2"),
        ),
    ),
    ElementSpec(
        "M",
        "MOSFET",
        pins=_pins("drain", "gate", "source", "bulk"),
        requires_model=True,
        model_name="model",
        parameters=_params(
            ElementParameterSpec("width", "w", unit="m"),
            ElementParameterSpec("length", "l", unit="m"),
            ElementParameterSpec("multiplier", "m"),
            ElementParameterSpec("area_drain", "ad", unit="m^2"),
            ElementParameterSpec("area_source", "as", unit="m^2"),
            ElementParameterSpec("perimeter_drain", "pd", unit="m"),
            ElementParameterSpec("perimeter_source", "ps", unit="m"),
            ElementParameterSpec("drain_squares", "nrd"),
            ElementParameterSpec("source_squares", "nrs"),
            ElementParameterSpec("off", "off"),
            ElementParameterSpec("initial_condition", "ic"),
            ElementParameterSpec("temperature", "temp", unit="degC"),
            ElementParameterSpec("fins", "nfin"),
        ),
    ),
    ElementSpec("N", "Numerical/GSS device", min_nodes=1, requires_model=True, model_name="model"),
    ElementSpec("O", "Lossy transmission line", pins=_pins("p1", "n1", "p2", "n2"), requires_model=True),
    ElementSpec("P", "Coupled multiconductor line", min_nodes=2, requires_model=True, model_name="model"),
    ElementSpec(
        "Q",
        "BJT",
        pins=(
            PinSpec("collector"),
            PinSpec("base"),
            PinSpec("emitter"),
            PinSpec("substrate", optional=True),
        ),
        requires_model=True,
        model_name="model",
        parameters=_params(
            ElementParameterSpec("area", "area"),
            ElementParameterSpec("area_collector", "areac"),
            ElementParameterSpec("area_base", "areab"),
            ElementParameterSpec("multiplier", "m"),
            ElementParameterSpec("off", "off"),
            ElementParameterSpec("initial_condition", "ic"),
            ElementParameterSpec("temperature", "temp", unit="degC"),
            ElementParameterSpec("device_temperature", "dtemp", unit="degC"),
        ),
    ),
    ElementSpec(
        "R",
        "Resistor",
        pins=_pins("p", "n"),
        requires_value=True,
        value_name="resistance",
        model_name="model",
        parameters=_params(
            ElementParameterSpec("length", "l", unit="m"),
            ElementParameterSpec("width", "w", unit="m"),
            ElementParameterSpec("ac_resistance", "ac", unit="Ohm"),
            ElementParameterSpec("multiplier", "m"),
            ElementParameterSpec("scale", "scale"),
            ElementParameterSpec("temperature", "temp", unit="degC"),
            ElementParameterSpec("device_temperature", "dtemp", unit="degC"),
            ElementParameterSpec("noisy", "noisy"),
            ElementParameterSpec("temperature_coefficient_1", "tc1"),
            ElementParameterSpec("temperature_coefficient_2", "tc2"),
        ),
    ),
    ElementSpec(
        "S",
        "Voltage-controlled switch",
        pins=_pins("p", "n", "cp", "cn"),
        requires_model=True,
        parameters=_params(ElementParameterSpec("initial_state")),
    ),
    ElementSpec(
        "T",
        "Lossless transmission line",
        pins=_pins("p1", "n1", "p2", "n2"),
        requires_value=True,
        value_name="line_parameters",
        parameters=_params(
            ElementParameterSpec("impedance", "z0", unit="Ohm"),
            ElementParameterSpec("time_delay", "td", unit="s"),
            ElementParameterSpec("frequency", "f", unit="Hz"),
            ElementParameterSpec("normalized_length", "nl"),
            ElementParameterSpec("initial_condition", "ic"),
        ),
    ),
    ElementSpec("U", "Uniform distributed RC line", pins=_pins("output", "input", "capacitance"), requires_model=True),
    ElementSpec("V", "Voltage source", pins=_pins("p", "n"), requires_value=True, value_name="voltage"),
    ElementSpec(
        "W",
        "Current-controlled switch",
        pins=_pins("p", "n"),
        requires_model=True,
        requires_value=True,
        parameters=_params(ElementParameterSpec("initial_state")),
    ),
    ElementSpec("X", "Subcircuit instance", min_nodes=2, requires_model=True, model_name="subcircuit"),
    ElementSpec("Y", "Single lossy transmission line", pins=_pins("p1", "n1", "p2", "n2"), requires_model=True),
    ElementSpec(
        "Z",
        "MESFET",
        pins=_pins("drain", "gate", "source"),
        requires_model=True,
        model_name="model",
        parameters=_params(
            ElementParameterSpec("area", "area"),
            ElementParameterSpec("multiplier", "m"),
            ElementParameterSpec("off", "off"),
            ElementParameterSpec("initial_condition", "ic"),
        ),
    ),
)

_DEFAULT_SCHEMA = default_device_schema()


__all__ = [
    "DEFAULT_ELEMENT_SPECS",
    "DeviceSchemaError",
    "DeviceSchemaRegistry",
    "ElementParameterSpec",
    "ElementSpec",
    "PinSpec",
    "default_device_schema",
    "element_spec",
    "normalize_element_params",
]
