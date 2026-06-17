"""Declarative, non-executing Monata view formats."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import re
from typing import Any, cast

from monata.errors import ViewNotGeneratedError
from monata.netlist import Circuit, SubCircuit
from monata.schematic import SchematicData, load_schematic, schematic_to_subcircuit
from monata.views.base import View
from monata.views.path_safety import resolve_cell_relative_path

MONATA_SCHEMATIC_JSON = "monata-schematic-json"
MONATA_SYMBOL_JSON = "monata-symbol-json"
MONATA_TESTBENCH_JSON = "monata-testbench-json"
SCHEMA_VERSION = 1
SCHEMATIC_SCHEMA_VERSION = 2

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
        schema_version: int | None = SCHEMATIC_SCHEMA_VERSION,
    ):
        super().__init__(
            view_type=view_type,
            cell=cell,
            entry=entry,
            generated=generated,
            format=MONATA_SCHEMATIC_JSON,
            schema_version=schema_version,
        )

    def read(self) -> SchematicData:
        return load_schematic(
            _read_json_object(self, generated=False),
            default_cell=getattr(self.cell, "name", None),
        )

    def load(self) -> SchematicData:
        return self.read()

    def to_circuit(self) -> SubCircuit:
        return schematic_to_subcircuit(self.read())

    def pin_names(self) -> tuple[str, ...]:
        return self.read().pin_names


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
        simulator: str | None = None,
    ):
        from monata.sim.core import SimTask

        payload = self.read()
        dut_cell = _resolve_dut_cell(self.cell, library, payload["dut"])
        dut = schematic_view_to_circuit(
            dut_cell["schematic"],
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
    reason: str,
) -> Circuit | SubCircuit:
    """Resolve a schematic view to native netlist IR without hidden Python fallback."""

    view_format = getattr(view, "format", None)
    if view_format == MONATA_SCHEMATIC_JSON or isinstance(view, SchematicJsonView):
        to_circuit = getattr(view, "to_circuit", None)
        if not callable(to_circuit):
            raise TypeError(f"{reason}: schematic data view is not convertible to a circuit")
        return _require_native_netlist(to_circuit(), reason=reason)

    raise TypeError(f"{reason}: unsupported schematic view format {view_format!r}")


def schematic_pin_names(
    view: Any,
    *,
    reason: str,
) -> tuple[str, ...]:
    pin_names = getattr(view, "pin_names", None)
    if callable(pin_names):
        return tuple(str(pin) for pin in cast(Iterable[Any], pin_names()))
    circuit = schematic_view_to_circuit(
        view,
        reason=reason,
    )
    return tuple(str(node) for node in getattr(circuit, "nodes", ()))


def schematic_view_to_subcircuit(
    view: Any,
    *,
    reason: str,
) -> SubCircuit:
    circuit = schematic_view_to_circuit(
        view,
        reason=reason,
    )
    if not isinstance(circuit, SubCircuit):
        raise TypeError(f"{reason}: digital truth-table schematics must resolve to a SubCircuit")
    return circuit


def schematic_payload_to_circuit(payload: Mapping[str, Any], *, default_name: str) -> SubCircuit:
    return schematic_to_subcircuit(load_schematic(payload, default_cell=default_name))


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
    file_path = resolve_cell_relative_path(
        view.path(),
        view.entry,
        label=f"{view.view_type}.entry",
    )
    if not file_path.exists():
        if generated:
            raise ViewNotGeneratedError(view.view_type, view.cell.name)
        raise FileNotFoundError(f"{view.view_type.capitalize()} file not found: {file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{view.entry} must contain a JSON object")
    return data


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
