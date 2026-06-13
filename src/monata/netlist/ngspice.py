"""ngspice renderer for the native Monata netlist IR."""

from __future__ import annotations

from typing import Any

from monata.netlist.ir import Circuit, Directive, Element, SourceValue, SubCircuit, instance_name
from monata.units import format_spice_value, join_spice_values

_MODEL_REFERENCE_DIRECTIVES = {"lib", "monata_model_ref"}


def render_ngspice(netlist: Circuit | SubCircuit) -> str:
    """Render a native netlist object to deterministic ngspice text."""

    if isinstance(netlist, Circuit):
        return _render_circuit(netlist)
    if isinstance(netlist, SubCircuit):
        return _render_subcircuit(netlist.ensure_built())
    raise TypeError("render_ngspice expects Circuit or SubCircuit")


def _render_circuit(circuit: Circuit) -> str:
    lines = [circuit.title]
    lines.extend(_render_includes(circuit.includes))
    lines.extend(_render_params(circuit.params))
    lines.extend(_render_directive(directive) for directive in circuit.directives)
    subcircuits = [subcircuit.ensure_built() for subcircuit in circuit.subcircuits]
    lines.extend(_render_model_references(subcircuits))
    for subcircuit in subcircuits:
        if len(lines) > 1:
            lines.append("")
        lines.extend(_render_subcircuit_lines(subcircuit))
    elements = _enabled_elements(circuit.elements)
    if elements:
        if len(lines) > 1:
            lines.append("")
        for element in elements:
            lines.extend(_render_element_lines(element))
    if circuit.outputs:
        if len(lines) > 1:
            lines.append("")
        lines.extend(circuit.outputs)
    lines.append(".end")
    return "\n".join(lines) + "\n"


def _render_subcircuit(subcircuit: SubCircuit) -> str:
    subcircuit = subcircuit.ensure_built()
    return "\n".join([
        *_render_model_references([subcircuit]),
        *_render_subcircuit_lines(subcircuit),
    ]) + "\n"


def _render_subcircuit_lines(subcircuit: SubCircuit) -> list[str]:
    signature = [".subckt", subcircuit.name, *subcircuit.nodes]
    signature.extend(f"{name}={_format_value(value)}" for name, value in subcircuit.params.items())
    lines = [" ".join(signature)]
    lines.extend(_render_includes(_body_includes(subcircuit)))
    lines.extend(_render_directive(directive) for directive in _body_directives(subcircuit))
    for element in _enabled_elements(subcircuit.elements):
        lines.extend(_render_element_lines(element))
    lines.append(f".ends {subcircuit.name}")
    return lines


def _render_model_references(subcircuits: list[SubCircuit]) -> list[str]:
    lines = []
    seen = set()
    for subcircuit in subcircuits:
        for directive in subcircuit.directives:
            if not _is_model_reference_directive(directive):
                continue
            line = _render_directive(directive)
            if line not in seen:
                seen.add(line)
                lines.append(line)
    return lines


def _body_includes(subcircuit: SubCircuit) -> list[str]:
    return subcircuit.includes


def _body_directives(subcircuit: SubCircuit) -> list[Directive]:
    return [
        directive
        for directive in subcircuit.directives
        if not _is_model_reference_directive(directive)
    ]


def _is_model_reference_directive(directive: Directive) -> bool:
    return not directive.raw and directive.name in _MODEL_REFERENCE_DIRECTIVES


def _render_includes(includes: list[str]) -> list[str]:
    return [f'.include "{path}"' for path in includes]


def _render_params(params: dict[str, Any]) -> list[str]:
    return [f".param {name}={_format_value(value)}" for name, value in params.items()]


def _render_directive(directive: Directive) -> str:
    if directive.raw:
        return directive.name
    if directive.name == "model":
        name, model_type, *rest = directive.args
        parts = [".model", str(name), str(model_type), *(_format_value(value) for value in rest)]
        if directive.params:
            params = " ".join(
                f"{key}={_format_value(value)}" for key, value in directive.params.items()
            )
            parts.append(f"({params})")
        return " ".join(parts)
    parts = [f".{directive.name}", *(_format_value(value) for value in directive.args)]
    if directive.name in {"nodeset", "ic"}:
        parts.extend(f"v({key})={_format_value(value)}" for key, value in directive.params.items())
    elif directive.name == "options":
        parts.extend(
            str(key) if value is True else f"{key}={_format_value(value)}"
            for key, value in directive.params.items()
        )
    else:
        parts.extend(f"{key}={_format_value(value)}" for key, value in directive.params.items())
    return " ".join(parts)


def _render_element(element: Element) -> str:
    return "\n".join(_render_element_lines(element))


def _render_element_lines(element: Element) -> list[str]:
    if not element.enabled:
        return []
    lines = []
    if element.comment:
        lines.append(f"* {element.comment}")
    name = instance_name(element.kind, element.name)
    parts = [name, *element.nodes]
    consumed_params: set[str] = set()
    if element.kind in {"R", "C", "L"}:
        parts.append(_format_value(element.value))
        if element.model is not None:
            parts.append(str(element.model))
    elif element.kind in {"V", "I", "B", "E", "G", "K", "T"}:
        parts.append(_format_value(element.value))
    elif element.kind in {"A", "D", "J", "M", "N", "O", "P", "Q", "U", "X", "Y", "Z"}:
        parts.append(str(element.model))
    elif element.kind == "S":
        parts.append(str(element.model))
        if (initial_state := _element_param(element, "initial_state")) is not None:
            parts.append(_format_value(initial_state))
            consumed_params.add("initial_state")
    elif element.kind in {"F", "H"}:
        parts.extend([str(element.model), _format_value(element.value)])
    elif element.kind == "W":
        parts.extend([_format_value(element.value), str(element.model)])
        if (initial_state := _element_param(element, "initial_state")) is not None:
            parts.append(_format_value(initial_state))
            consumed_params.add("initial_state")
    parts.extend(
        rendered
        for key, value in element.params.items()
        if key.lower() not in consumed_params
        if (rendered := _render_element_param(key, value))
    )
    if element.raw_suffix:
        parts.append(element.raw_suffix)
    lines.append(" ".join(parts))
    return lines


def _enabled_elements(elements: list[Element]) -> list[Element]:
    return [element for element in elements if element.enabled]


def _element_param(element: Element, name: str) -> Any | None:
    key = name.lower()
    for param_name, value in element.params.items():
        if param_name.lower() == key:
            return value
    return None


def _render_element_param(key: str, value: Any) -> str:
    if key.lower() == "off":
        return "off" if bool(value) else ""
    return f"{key}={_format_value(value)}"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, SourceValue):
        if value.form == "PWL":
            return _format_pwl_source_value(value)
        values = join_spice_values(*value.values)
        if value.form == "DC":
            return f"DC {values}"
        if value.form == "AC":
            dc, ac = (_format_value(item) for item in value.values)
            return f"DC {dc} AC {ac}"
        return f"{value.form}({values})"
    return format_spice_value(value)


def _format_pwl_source_value(value: SourceValue) -> str:
    body = [token for item in value.values if (token := _format_value(item))]
    dc_prefix = ""
    for key, item in value.params.items():
        rendered = _format_value(item)
        if not rendered:
            continue
        if key.lower() == "dc":
            dc_prefix = f"DC {rendered} "
        else:
            body.append(f"{key}={rendered}")
    return f"{dc_prefix}PWL({' '.join(body)})"
