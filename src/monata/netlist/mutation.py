"""Structured mutation helpers for native and imported netlists."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
import re
from typing import Any

from monata.netlist.ir import Circuit, Directive, Element, SourceValue, SubCircuit


class MutationError(ValueError):
    """Raised when a structured netlist mutation cannot be resolved."""


@dataclass(frozen=True)
class MutationProjection:
    """Backend-ready mutation projection."""

    circuit: Circuit
    param_overrides: dict[str, Any]
    metadata: dict[str, Any]


_RAW_DIRECTIVE_TARGET_RE = re.compile(r"^raw\.(\d+)$")


def apply_mutation(circuit: Circuit, target: str, value: Any) -> Circuit:
    """Return a copied circuit with one structured mutation applied."""

    projected = deepcopy(circuit)
    _apply_to_scope(projected, target, value)
    return projected


def project_param_overrides(circuit: Circuit, overrides: dict | None) -> MutationProjection:
    """Project SimTask-style overrides into structured mutations plus legal params."""

    if not overrides:
        return MutationProjection(circuit, {}, {"structured_mutations": []})
    projected = deepcopy(circuit)
    param_overrides: dict[str, Any] = {}
    mutation_metadata: list[dict[str, Any]] = []
    for target, value in overrides.items():
        target_text = str(target)
        if "." not in target_text:
            param_overrides[target_text] = value
            mutation_metadata.append({"target": target_text, "kind": "global_param"})
            continue
        if _is_raw_directive_target(target_text):
            index = _raw_directive_index(target_text)
            _apply_raw_directive(projected, index, value, target_text)
            mutation_metadata.append({"target": target_text, "kind": "raw_directive", "index": index})
            continue
        _apply_to_scope(projected, target_text, value)
        mutation_metadata.append({"target": target_text, "kind": "structured"})
    return MutationProjection(
        projected,
        param_overrides,
        {"structured_mutations": mutation_metadata},
    )


def _apply_to_scope(scope: Circuit | SubCircuit, target: str, value: Any) -> None:
    element_name, param_name = _split_target(target)
    matches = _find_mutation_targets(scope, element_name)
    if not matches:
        raise MutationError(f"mutation target not found: {target}")
    if len(matches) > 1:
        candidates = ", ".join(_target_label(candidate_scope, element) for candidate_scope, element in matches)
        raise MutationError(f"ambiguous mutation target: {target} matched {candidates}")
    target_scope, element = matches[0]
    replacement = _mutated_element(element, param_name, value)
    _replace_element(target_scope, element, replacement)


def _is_raw_directive_target(target: str) -> bool:
    return _RAW_DIRECTIVE_TARGET_RE.match(target) is not None


def _raw_directive_index(target: str) -> int:
    match = _RAW_DIRECTIVE_TARGET_RE.match(target)
    if match is None:
        raise MutationError(f"invalid raw directive mutation target: {target}")
    return int(match.group(1))


def _apply_raw_directive(scope: Circuit | SubCircuit, index: int, value: Any, target: str) -> None:
    raw_directives = [(directive_index, directive) for directive_index, directive in enumerate(scope.directives) if directive.raw]
    if index >= len(raw_directives):
        raise MutationError(f"raw directive mutation target not found: {target}")
    directive_index, old = raw_directives[index]
    scope.directives[directive_index] = Directive(str(value), raw=True)


def _find_mutation_targets(scope: Circuit | SubCircuit, element_name: str) -> list[tuple[Circuit | SubCircuit, Element]]:
    matches: list[tuple[Circuit | SubCircuit, Element]] = []
    element = _find_element(scope, element_name)
    if element is not None:
        matches.append((scope, element))
    for subckt in getattr(scope, "subcircuits", ()):
        matches.extend(_find_mutation_targets(subckt, element_name))
    return matches


def _target_label(scope: Circuit | SubCircuit, element: Element) -> str:
    scope_name = getattr(scope, "name", None) or getattr(scope, "title", "circuit")
    return f"{scope_name}.{element.kind}{element.name}"


def _split_target(target: str) -> tuple[str, str]:
    if "." not in target:
        raise MutationError(f"structured mutation target requires element.param: {target}")
    element_name, param_name = target.split(".", 1)
    if not element_name or not param_name:
        raise MutationError(f"invalid mutation target: {target}")
    return element_name, param_name


def _find_element(scope: Circuit | SubCircuit, name: str) -> Element | None:
    direct = scope.get_element(name)
    if direct is not None:
        return direct
    if len(name) >= 2 and name[0].isalpha():
        kind = name[0]
        direct = scope.get_element(name, kind=kind)
        if direct is not None:
            return direct
        return scope.get_element(name[1:], kind=kind)
    return None


def _mutated_element(element: Element, param_name: str, value: Any) -> Element:
    key = param_name.lower()
    if key in {"value", element.kind.lower()}:
        if element.value is None and element.kind not in {"R", "C", "L", "V", "I", "B", "E", "G", "K", "T", "W"}:
            raise MutationError(f"element {element.name} has no mutable value")
        return replace(element, value=_mutated_value(element.value, value))
    params = OrderedDict(element.params)
    params[_canonical_param_name(params, param_name)] = value
    return replace(element, params=params)


def _canonical_param_name(params: OrderedDict[str, Any], param_name: str) -> str:
    matches = [key for key in params if key.lower() == param_name.lower()]
    if len(matches) > 1:
        candidates = ", ".join(matches)
        raise MutationError(f"ambiguous parameter target: {param_name} matched {candidates}")
    if matches:
        return matches[0]
    return param_name


def _mutated_value(current: Any, value: Any) -> Any:
    if isinstance(current, SourceValue):
        values = (value, *current.values[1:]) if current.values else (value,)
        return SourceValue(current.form, values, OrderedDict(current.params))
    return value


def _replace_element(scope: Circuit | SubCircuit, old: Element, new: Element) -> None:
    for index, element in enumerate(scope.elements):
        if element is old:
            scope.elements[index] = new
            _rebuild_element_indexes(scope)
            return
    raise MutationError(f"element not found in scope: {old.name}")


def _rebuild_element_indexes(scope: Circuit | SubCircuit) -> None:
    scope._rebuild_element_indexes()
