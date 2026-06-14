"""Declarative, non-executing Monata view formats."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import re
from typing import Any, cast

from monata.errors import ViewNotGeneratedError
from monata.netlist import Circuit, SubCircuit
from monata.views.base import View

MONATA_SCHEMATIC_JSON = "monata-schematic-json"
MONATA_SYMBOL_JSON = "monata-symbol-json"
MONATA_TESTBENCH_JSON = "monata-testbench-json"
PYTHON_SCHEMATIC = "python-schematic"
PYTHON_TESTBENCH = "python-testbench"
SCHEMA_VERSION = 1

_NUMBER_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-zµ]*)\s*$")
_UNIT_FACTORS = {
    "": 1.0,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9,
    "t": 1e12,
}


class SchematicJsonView(View):
    """Safe JSON schematic cellview."""

    def __init__(
        self,
        cell,
        entry: str,
        *,
        view_type: str = "schematic",
        generated: bool = False,
        schema_version: int | None = SCHEMA_VERSION,
    ):
        super().__init__(
            view_type=view_type,
            cell=cell,
            entry=entry,
            generated=generated,
            format=MONATA_SCHEMATIC_JSON,
            trusted=False,
            schema_version=schema_version,
        )

    def read(self) -> dict[str, Any]:
        return _validate_schematic_payload(
            _read_json_object(self, generated=False),
            cell_name=getattr(self.cell, "name", None),
        )

    def load(self) -> dict[str, Any]:
        return self.read()

    def to_circuit(self) -> SubCircuit:
        return schematic_payload_to_circuit(self.read(), default_name=str(self.cell.name))

    def pin_names(self) -> tuple[str, ...]:
        return tuple(pin["name"] for pin in self.read()["pins"])


class SymbolJsonView(View):
    """Safe JSON symbol cellview."""

    def __init__(
        self,
        cell,
        entry: str,
        *,
        view_type: str = "symbol",
        generated: bool = True,
        schema_version: int | None = SCHEMA_VERSION,
    ):
        super().__init__(
            view_type=view_type,
            cell=cell,
            entry=entry,
            generated=generated,
            format=MONATA_SYMBOL_JSON,
            trusted=False,
            schema_version=schema_version,
        )

    def read(self) -> dict[str, Any]:
        return _validate_symbol_payload(
            _read_json_object(self, generated=self.generated),
            cell_name=getattr(self.cell, "name", None),
        )

    def load(self) -> dict[str, Any]:
        payload = self.read()
        return {
            "name": payload.get("name", getattr(self.cell, "name", "")),
            "pins": payload["pins"],
        }

    def pin_names(self) -> tuple[str, ...]:
        return tuple(pin["name"] for pin in self.read()["pins"])


class TestbenchJsonView(View):
    """Safe JSON simulation task cellview."""

    __test__ = False

    def __init__(
        self,
        cell,
        entry: str,
        *,
        view_type: str = "testbench",
        generated: bool = False,
        schema_version: int | None = SCHEMA_VERSION,
    ):
        super().__init__(
            view_type=view_type,
            cell=cell,
            entry=entry,
            generated=generated,
            format=MONATA_TESTBENCH_JSON,
            trusted=False,
            schema_version=schema_version,
        )

    def read(self) -> dict[str, Any]:
        return _validate_testbench_payload(
            _read_json_object(self, generated=False),
            default_dut=getattr(self.cell, "name", None),
        )

    def load(self) -> dict[str, Any]:
        return self.read()

    def to_sim_task(
        self,
        *,
        library: Any = None,
        allow_trusted_python: bool = False,
        simulator: str | None = None,
    ):
        from monata.sim.core import SimTask

        payload = self.read()
        dut_cell = _resolve_dut_cell(self.cell, library, payload["dut"])
        dut = schematic_view_to_circuit(
            dut_cell["schematic"],
            allow_trusted_python=allow_trusted_python,
            reason="testbench JSON DUT",
        )
        circuit = _testbench_circuit_for(dut, dut_name=payload["dut"])
        _add_testbench_sources(circuit, payload["sources"])
        metadata = {
            "view_type": "testbench",
            "schema_version": SCHEMA_VERSION,
            "dut": payload["dut"],
            "measurements": tuple(payload["measurements"]),
        }
        return SimTask(
            circuit=circuit,
            analysis_spec=_analysis_spec(payload["analysis"]),
            simulator=simulator or "ngspice-subprocess",
            output_names=payload["outputs"],
            metadata=metadata,
        )

    def to_sim_tasks(self, **kwargs: Any) -> tuple[Any, ...]:
        return (self.to_sim_task(**kwargs),)


def schematic_view_to_circuit(
    view: Any,
    *,
    allow_trusted_python: bool,
    reason: str,
) -> Circuit | SubCircuit:
    """Resolve a schematic view to native netlist IR without hidden Python fallback."""

    view_format = getattr(view, "format", None)
    if view_format == MONATA_SCHEMATIC_JSON or isinstance(view, SchematicJsonView):
        to_circuit = getattr(view, "to_circuit", None)
        if not callable(to_circuit):
            raise TypeError(f"{reason}: schematic data view is not convertible to a circuit")
        return _require_native_netlist(to_circuit(), reason=reason)

    if view_format == PYTHON_SCHEMATIC:
        if not allow_trusted_python:
            raise TypeError(f"{reason}: refusing to execute trusted Python schematic view")
        if not getattr(view, "trusted", False):
            raise ValueError(f"{reason}: python-schematic views require trusted = true")
        load_trusted = getattr(view, "load_trusted", None)
        if not callable(load_trusted):
            raise TypeError(f"{reason}: python-schematic view does not implement load_trusted()")
        return _require_native_netlist(_materialize_native_netlist(load_trusted()), reason=reason)

    raise TypeError(f"{reason}: unsupported schematic view format {view_format!r}")


def schematic_pin_names(
    view: Any,
    *,
    allow_trusted_python: bool,
    reason: str,
) -> tuple[str, ...]:
    pin_names = getattr(view, "pin_names", None)
    if callable(pin_names):
        return tuple(str(pin) for pin in cast(Iterable[Any], pin_names()))
    circuit = schematic_view_to_circuit(
        view,
        allow_trusted_python=allow_trusted_python,
        reason=reason,
    )
    return tuple(str(node) for node in getattr(circuit, "nodes", ()))


def schematic_view_to_subcircuit(
    view: Any,
    *,
    allow_trusted_python: bool,
    reason: str,
) -> SubCircuit:
    circuit = schematic_view_to_circuit(
        view,
        allow_trusted_python=allow_trusted_python,
        reason=reason,
    )
    if not isinstance(circuit, SubCircuit):
        raise TypeError(f"{reason}: digital truth-table schematics must resolve to a SubCircuit")
    return circuit


def schematic_payload_to_circuit(payload: Mapping[str, Any], *, default_name: str) -> SubCircuit:
    data = _validate_schematic_payload(payload, cell_name=default_name)
    circuit = SubCircuit(name=str(data.get("cell") or default_name), nodes=tuple(pin["name"] for pin in data["pins"]))
    for instance in data["instances"]:
        _add_schematic_instance(circuit, instance)
    circuit.ensure_built()
    return circuit


def parse_metric_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric, not bool")
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"{field} must be numeric or metric-suffixed string")
    match = _NUMBER_RE.match(value)
    if match is None:
        raise ValueError(f"{field} is not a numeric metric value: {value!r}")
    number, suffix = match.groups()
    normalized = suffix.lower()
    factor = _UNIT_FACTORS.get(normalized)
    if factor is None:
        raise ValueError(f"{field} has unsupported metric suffix: {suffix!r}")
    return float(number) * factor


def _read_json_object(view: View, *, generated: bool) -> dict[str, Any]:
    file_path = view.path() / view.entry
    if not file_path.exists():
        if generated:
            raise ViewNotGeneratedError(view.view_type, view.cell.name)
        raise FileNotFoundError(f"{view.view_type.capitalize()} file not found: {file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{view.entry} must contain a JSON object")
    return data


def _validate_schematic_payload(payload: Mapping[str, Any], *, cell_name: str | None) -> dict[str, Any]:
    data = _expect_mapping(payload, "schematic")
    _reject_unknown_fields(data, {"schema_version", "view_type", "cell", "pins", "instances", "nets"})
    _require_schema(data, view_type="schematic")
    cell = _optional_string(data.get("cell"), "schematic.cell") or cell_name
    if not cell:
        raise ValueError("schematic.cell is required")
    pins = _validate_pins(data.get("pins"), "schematic.pins", allowed={"name", "direction"})
    instance_items = _list_or_empty(data.get("instances"))
    instances = [_validate_instance(item, index) for index, item in enumerate(instance_items)]
    nets = tuple(_validate_string_list(data.get("nets"), "schematic.nets")) if "nets" in data else _infer_nets(pins, instances)
    _validate_schematic_nets(pins, instances, nets)
    return {
        "schema_version": SCHEMA_VERSION,
        "view_type": "schematic",
        "cell": cell,
        "pins": pins,
        "instances": instances,
        "nets": list(nets),
    }


def _validate_symbol_payload(payload: Mapping[str, Any], *, cell_name: str | None) -> dict[str, Any]:
    data = _expect_mapping(payload, "symbol")
    _reject_unknown_fields(data, {"schema_version", "view_type", "name", "pins"})
    _require_schema(data, view_type="symbol")
    pins = _validate_pins(data.get("pins"), "symbol.pins", allowed={"name", "side", "direction"})
    name = _optional_string(data.get("name"), "symbol.name") or cell_name or ""
    return {
        "schema_version": SCHEMA_VERSION,
        "view_type": "symbol",
        "name": name,
        "pins": pins,
    }


def _validate_testbench_payload(payload: Mapping[str, Any], *, default_dut: str | None) -> dict[str, Any]:
    data = _expect_mapping(payload, "testbench")
    _reject_unknown_fields(data, {"schema_version", "view_type", "dut", "analysis", "sources", "outputs", "measurements"})
    _require_schema(data, view_type="testbench")
    dut = _optional_string(data.get("dut"), "testbench.dut") or default_dut
    if not dut:
        raise ValueError("testbench.dut is required")
    analysis = _expect_mapping(data.get("analysis"), "testbench.analysis")
    source_items = _list_or_empty(data.get("sources"))
    sources = [_validate_source(item, index) for index, item in enumerate(source_items)]
    return {
        "schema_version": SCHEMA_VERSION,
        "view_type": "testbench",
        "dut": dut,
        "analysis": dict(analysis),
        "sources": sources,
        "outputs": _validate_string_list(data.get("outputs", []), "testbench.outputs"),
        "measurements": _validate_string_list(data.get("measurements", []), "testbench.measurements"),
    }


def _require_schema(data: Mapping[str, Any], *, view_type: str) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{view_type}.schema_version must be {SCHEMA_VERSION}")
    if data.get("view_type") != view_type:
        raise ValueError(f"{view_type}.view_type must be {view_type!r}")


def _validate_pins(value: Any, label: str, *, allowed: set[str]) -> list[dict[str, str]]:
    pins = _optional_list(value, None)
    if pins is None:
        raise ValueError(f"{label} is required")
    result = []
    for index, pin in enumerate(pins):
        item = _expect_mapping(pin, f"{label}[{index}]")
        _reject_unknown_fields(item, allowed)
        name = _required_string(item.get("name"), f"{label}[{index}].name")
        normalized = {"name": name}
        for key in sorted(allowed - {"name"}):
            if key in item:
                normalized[key] = _required_string(item[key], f"{label}[{index}].{key}")
        result.append(normalized)
    return result


def _validate_instance(value: Any, index: int) -> dict[str, Any]:
    label = f"schematic.instances[{index}]"
    item = _expect_mapping(value, label)
    _reject_unknown_fields(
        item,
        {
            "name",
            "kind",
            "device",
            "model",
            "subckt",
            "lib",
            "cell",
            "view",
            "connections",
            "pin_order",
            "nodes",
            "parameters",
            "value",
        },
    )
    connections = _validate_connections(item.get("connections"), f"{label}.connections")
    parameters = item.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{label}.parameters must be an object")
    normalized = {
        "name": _required_string(item.get("name"), f"{label}.name"),
        "connections": connections,
        "parameters": dict(parameters),
    }
    for key in ("kind", "device", "model", "subckt", "lib", "cell", "view", "value"):
        if key in item:
            normalized[key] = item[key]
    for key in ("pin_order", "nodes"):
        if key in item:
            normalized[key] = _validate_string_list(item[key], f"{label}.{key}")
    return normalized


def _validate_source(value: Any, index: int) -> dict[str, Any]:
    label = f"testbench.sources[{index}]"
    item = _expect_mapping(value, label)
    _reject_unknown_fields(item, {"kind", "name", "node", "ref", "p", "n", "value", "values", "parameters"})
    kind = _required_string(item.get("kind"), f"{label}.kind").lower()
    pulse_kinds = {"vpulse", "ipulse"}
    scalar_kinds = {"vdc", "dc_voltage", "idc", "dc_current", "voltage", "vsource", "v", "current", "isource", "i"}
    if kind not in pulse_kinds | scalar_kinds:
        raise ValueError(f"{label}.kind is unsupported: {kind}")
    parameters = item.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{label}.parameters must be an object")
    normalized = {
        "kind": kind,
        "name": _required_string(item.get("name"), f"{label}.name"),
        "parameters": dict(parameters),
    }
    _copy_source_endpoint(item, normalized, label)
    if kind in pulse_kinds:
        values = item.get("values")
        if not isinstance(values, list):
            raise ValueError(f"{label}.values must be an array")
        if len(values) != 7:
            raise ValueError(f"{label}.values must contain 7 values")
        normalized["values"] = list(values)
    else:
        if "value" not in item or item["value"] is None:
            raise ValueError(f"{label}.value is required")
        normalized["value"] = item["value"]
    return normalized


def _copy_source_endpoint(source: Mapping[str, Any], target: dict[str, Any], label: str) -> None:
    if "p" in source:
        target["p"] = _required_string(source["p"], f"{label}.p")
    elif "node" in source:
        target["node"] = _required_string(source["node"], f"{label}.node")
    else:
        raise ValueError(f"{label} requires node or p")
    if "n" in source:
        target["n"] = _required_string(source["n"], f"{label}.n")
    if "ref" in source:
        target["ref"] = _required_string(source["ref"], f"{label}.ref")


def _validate_connections(value: Any, label: str) -> dict[str, str]:
    mapping = _expect_mapping(value, label)
    result = {}
    for key, node in mapping.items():
        result[_required_string(key, f"{label} key")] = _required_string(node, f"{label}.{key}")
    return result


def _validate_schematic_nets(pins: Sequence[Mapping[str, str]], instances: Sequence[Mapping[str, Any]], nets: Sequence[str]) -> None:
    known = set(nets)
    missing = [pin["name"] for pin in pins if pin["name"] not in known]
    for instance in instances:
        for node in instance["connections"].values():
            if node not in known:
                missing.append(str(node))
    if missing:
        raise ValueError("schematic.nets is missing referenced net(s): " + ", ".join(sorted(set(missing))))


def _infer_nets(pins: Sequence[Mapping[str, str]], instances: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names = [pin["name"] for pin in pins]
    for instance in instances:
        for node in instance["connections"].values():
            if node not in names:
                names.append(str(node))
    return tuple(names)


def _add_schematic_instance(scope: SubCircuit, instance: Mapping[str, Any]) -> None:
    name = str(instance["name"])
    connections = dict(instance["connections"])
    params = dict(instance.get("parameters", {}))
    kind = str(instance.get("kind") or instance.get("device") or "").lower()
    device = instance.get("device")
    if _is_pdk_instance(kind, instance):
        scope.pdk_instance(
            name,
            lib=str(instance["lib"]),
            cell=str(instance["cell"]),
            view=str(instance["view"]),
            pins=connections,
            params=params,
        )
        return
    if _is_mos_instance(kind, connections):
        model = str(instance.get("model") or device or kind)
        scope.mos(
            name,
            d=connections["d"],
            g=connections["g"],
            s=connections["s"],
            b=connections["b"],
            model=model,
            **params,
        )
        return
    if kind in {"resistor", "res", "r"}:
        scope.resistor(name, connections["n1"], connections["n2"], _instance_value(instance, params), **params)
        return
    if kind in {"capacitor", "cap", "c"}:
        scope.capacitor(name, connections["n1"], connections["n2"], _instance_value(instance, params), **params)
        return
    if kind in {"inductor", "ind", "l"}:
        scope.inductor(name, connections["n1"], connections["n2"], _instance_value(instance, params), **params)
        return
    if kind in {"voltage", "vsource", "v"}:
        p, n = _source_nodes(connections)
        scope.voltage(name, p, n, _instance_value(instance, params), **params)
        return
    if kind in {"current", "isource", "i"}:
        p, n = _source_nodes(connections)
        scope.current(name, p, n, _instance_value(instance, params), **params)
        return
    if kind in {"vpulse", "ipulse"}:
        p, n = _source_nodes(connections)
        values = _source_values(instance, count=7)
        if kind == "vpulse":
            scope.vpulse(name, p, n, *values, **params)
        else:
            scope.ipulse(name, p, n, *values, **params)
        return
    if _is_subckt_instance(kind, instance):
        subckt = str(instance.get("subckt") or instance.get("device") or instance.get("cell"))
        if "nodes" in instance:
            scope.instance(name, instance["nodes"], subckt, **params)
        else:
            scope.instance_pins(name, subckt, connections, pin_order=instance.get("pin_order"), **params)
        return
    raise ValueError(f"unsupported schematic instance kind for {name!r}: {kind or device!r}")


def _is_pdk_instance(kind: str, instance: Mapping[str, Any]) -> bool:
    return kind == "pdk" or all(key in instance for key in ("lib", "cell", "view"))


def _is_mos_instance(kind: str, connections: Mapping[str, str]) -> bool:
    return kind in {"mos", "mosfet", "nmos", "pmos"} or {"d", "g", "s", "b"} <= set(connections)


def _is_subckt_instance(kind: str, instance: Mapping[str, Any]) -> bool:
    return kind in {"instance", "subckt", "subcircuit", "x"} or "subckt" in instance


def _instance_value(instance: Mapping[str, Any], params: dict[str, Any]) -> Any:
    if "value" in instance:
        return instance["value"]
    try:
        return params.pop("value")
    except KeyError as exc:
        raise ValueError(f"schematic instance {instance['name']!r} is missing value") from exc


def _source_values(instance: Mapping[str, Any], *, count: int) -> list[Any]:
    values = instance.get("values") or instance.get("value")
    if not isinstance(values, list):
        raise ValueError(f"schematic instance {instance['name']!r} requires values array")
    if len(values) != count:
        raise ValueError(f"schematic instance {instance['name']!r} requires {count} values")
    return list(values)


def _source_nodes(connections: Mapping[str, str]) -> tuple[str, str]:
    if "p" in connections and "n" in connections:
        return connections["p"], connections["n"]
    if "node" in connections and "ref" in connections:
        return connections["node"], connections["ref"]
    raise ValueError("source connections require p/n or node/ref")


def _resolve_dut_cell(cell: Any, library: Any, dut_name: str) -> Any:
    if dut_name == getattr(cell, "name", None):
        return cell
    resolved_library = library or getattr(cell, "library", None)
    if resolved_library is None:
        raise ValueError("testbench JSON requires a library to resolve dut")
    return resolved_library[dut_name]


def _testbench_circuit_for(dut: Circuit | SubCircuit, *, dut_name: str) -> Circuit:
    if isinstance(dut, Circuit):
        return dut
    circuit = Circuit(f"{dut_name} testbench")
    circuit.subckt(dut)
    circuit.instance("dut", dut.nodes, dut)
    return circuit


def _add_testbench_sources(circuit: Circuit, sources: Iterable[Mapping[str, Any]]) -> None:
    for source in sources:
        kind = str(source["kind"]).lower()
        name = str(source["name"])
        params = dict(source.get("parameters", {}))
        p = str(source.get("p") or source.get("node"))
        n = str(source.get("n") or source.get("ref", "0"))
        if kind == "vpulse":
            values = source.get("values")
            if not isinstance(values, list) or len(values) != 7:
                raise ValueError(f"vpulse source {name!r} requires 7 values")
            circuit.vpulse(name, p, n, *values, **params)
        elif kind == "ipulse":
            values = source.get("values")
            if not isinstance(values, list) or len(values) != 7:
                raise ValueError(f"ipulse source {name!r} requires 7 values")
            circuit.ipulse(name, p, n, *values, **params)
        elif kind in {"vdc", "dc_voltage"}:
            circuit.vdc(name, p, n, source["value"], **params)
        elif kind in {"idc", "dc_current"}:
            circuit.idc(name, p, n, source["value"], **params)
        elif kind in {"voltage", "vsource", "v"}:
            circuit.voltage(name, p, n, source["value"], **params)
        elif kind in {"current", "isource", "i"}:
            circuit.current(name, p, n, source["value"], **params)
        else:
            raise ValueError(f"unsupported testbench source kind: {kind}")


def _analysis_spec(analysis: Mapping[str, Any]) -> Any:
    from monata.sim.core import ACSpec, DCSpec, OPSpec, TranSpec

    kind = _required_string(analysis.get("kind"), "testbench.analysis.kind").lower()
    if kind == "tran":
        return TranSpec(
            stop=parse_metric_number(analysis.get("stop"), field="tran.stop"),
            step=_optional_metric_number(analysis.get("step"), field="tran.step"),
            start=parse_metric_number(analysis.get("start", 0), field="tran.start"),
            max_step=_optional_metric_number(analysis.get("max_step"), field="tran.max_step"),
            uic=bool(analysis.get("uic", False)),
        )
    if kind == "dc":
        return DCSpec(
            source=_required_string(analysis.get("source"), "dc.source"),
            start=parse_metric_number(analysis.get("start"), field="dc.start"),
            stop=parse_metric_number(analysis.get("stop"), field="dc.stop"),
            step=parse_metric_number(analysis.get("step"), field="dc.step"),
        )
    if kind == "ac":
        return ACSpec(
            start=parse_metric_number(analysis.get("start"), field="ac.start"),
            stop=parse_metric_number(analysis.get("stop"), field="ac.stop"),
            points=_required_int(analysis.get("points"), "ac.points"),
            variation=str(analysis.get("variation", "dec")),
        )
    if kind == "op":
        return OPSpec()
    raise ValueError(f"unsupported analysis kind: {kind}")


def _optional_metric_number(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return parse_metric_number(value, field=field)


def _materialize_native_netlist(value: Any) -> Circuit | SubCircuit:
    if isinstance(value, type):
        value = value()
    return value


def _require_native_netlist(value: Any, *, reason: str) -> Circuit | SubCircuit:
    if not isinstance(value, Circuit | SubCircuit):
        raise TypeError(
            f"{reason}: schematic must define a monata.netlist Circuit or SubCircuit"
        )
    return value


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _reject_unknown_fields(data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(str(key) for key in data if str(key) not in allowed)
    if unknown:
        raise ValueError("unknown field(s): " + ", ".join(unknown))


def _optional_list(value: Any, default: list[Any] | None) -> list[Any] | None:
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError("expected an array")
    return value


def _list_or_empty(value: Any) -> list[Any]:
    result = _optional_list(value, [])
    return [] if result is None else result


def _validate_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [_required_string(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} cannot contain newlines")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, label)


def _required_int(value: Any, label: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return int(value)
