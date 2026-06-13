"""SPICE vector request and identity helpers."""

from __future__ import annotations

from dataclasses import dataclass

from monata.sim import _vector_identity


VECTOR_KINDS = {
    "abscissa",
    "node_voltage",
    "differential_voltage",
    "branch_current",
    "node_current",
    "element_parameter",
    "internal_parameter",
    "noise_spectrum",
    "noise_total",
    "ac_component",
    "sensitivity",
    "pole",
    "zero",
    "distortion",
    "transfer_function",
    "fourier_component",
    "expression",
    "unknown",
}


@dataclass(frozen=True)
class VectorName:
    """Canonical identity for a simulator vector name."""

    display_name: str
    normalized_name: str
    raw_vector_name: str
    vector_kind: str
    quantity: str | None = None


def voltage_vector(node: str, reference: str | None = None) -> str:
    """Return a SPICE voltage vector name for a node or differential node pair."""

    node_name = _validate_vector_part(node, "voltage node")
    if reference is None:
        return f"v({node_name})"
    reference_name = _validate_vector_part(reference, "voltage reference node")
    return f"v({node_name},{reference_name})"


def branch_current_vector(element: str) -> str:
    """Return a SPICE branch current vector name for an element name."""

    element_name = _validate_vector_part(element, "branch current element")
    return f"i({element_name})"


def node_current_vector(element: str, current: str) -> str:
    """Return a SPICE device-current vector such as ``@m1[id]``."""

    return device_parameter_vector(element, current)


def device_parameter_vector(element: str, parameter: str) -> str:
    """Return a SPICE device parameter vector such as ``@m1[gm]``."""

    element_name = _validate_device_vector_part(element, "device parameter element")
    parameter_name = _validate_device_vector_part(parameter, "device parameter name")
    return f"@{element_name}[{parameter_name}]"


def internal_parameter_vector(name: str) -> str:
    """Return a SPICE internal parameter vector such as ``@temp``."""

    parameter_name = _validate_device_vector_part(name, "internal parameter name")
    return f"@{parameter_name}"


def expression_vector(expression: str) -> str:
    """Return a validated single-token SPICE expression vector."""

    return _validate_expression_vector(expression)


def normalize_vector_name(name: str, *, display_name: str | None = None) -> VectorName:
    """Classify and normalize a simulator vector while preserving the raw name."""

    raw = str(name)
    text = raw.strip()
    lower = text.lower()
    kind = _vector_kind_for_name(text)
    quantity = _quantity_for_name(text)
    voltage_inner = _vector_identity.simple_vector_inner(text, "v")
    current_inner = _vector_identity.simple_vector_inner(text, "i")
    if voltage_inner is not None:
        inner = voltage_inner
        display = display_name or inner
        normalized = _vector_identity.safe_lookup_name(inner)
    elif current_inner is not None:
        inner = current_inner
        display = display_name or f"i({inner})"
        normalized = f"i_{_vector_identity.safe_lookup_name(inner)}"
    elif lower.endswith("#branch"):
        inner = text[:-7]
        display = display_name or text
        normalized = f"i_{_vector_identity.safe_lookup_name(inner)}"
    elif lower.startswith("@"):
        display = display_name or text
        normalized = _vector_identity.safe_lookup_name(text[1:])
    elif _vector_identity.looks_like_expression_vector(text):
        display = display_name or text
        normalized = _vector_identity.safe_lookup_name(text)
    else:
        display = display_name or text
        normalized = _vector_identity.safe_lookup_name(text)
    return VectorName(
        display_name=display,
        normalized_name=normalized,
        raw_vector_name=text,
        vector_kind=kind,
        quantity=quantity,
    )


def _quantity_for_name(name: str) -> str | None:
    lower = name.lower()
    if lower in {"onoise_spectrum", "inoise_spectrum"}:
        return "noise"
    if _vector_identity.simple_vector_inner(name, "i") is not None or lower.endswith("#branch"):
        return "current"
    if _vector_identity.simple_vector_inner(name, "v") is not None:
        return "voltage"
    return None


def _vector_kind_for_name(name: str) -> str:
    lower = name.lower()
    if lower in {"onoise_total", "inoise_total"}:
        return "noise_total"
    if lower in {"onoise_spectrum", "inoise_spectrum"}:
        return "noise_spectrum"
    if _vector_identity.simple_vector_inner(name, "i") is not None or lower.endswith("#branch"):
        return "branch_current"
    if lower.startswith("@"):
        if _vector_identity.device_parameter_from_vector(name) is not None:
            return "element_parameter"
        return "internal_parameter"
    voltage_inner = _vector_identity.simple_vector_inner(name, "v")
    if voltage_inner is not None and "," in voltage_inner:
        return "differential_voltage"
    if voltage_inner is not None:
        return "node_voltage"
    if _vector_identity.looks_like_expression_vector(name):
        return "expression"
    return "unknown"


def _validate_vector_part(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    if any(char in text for char in "\r\n;"):
        raise ValueError(f"invalid {label}: control characters or command separators are not allowed")
    if any(char.isspace() for char in text):
        raise ValueError(f"invalid {label}: whitespace is not allowed")
    if any(char in text for char in "(),"):
        raise ValueError(f"invalid {label}: vector delimiters are not allowed")
    return text


def _validate_device_vector_part(value: str, label: str) -> str:
    text = _validate_vector_part(value, label)
    if any(char in text for char in "[]"):
        raise ValueError(f"invalid {label}: device parameter delimiters are not allowed")
    return text


def _validate_expression_vector(expression: str) -> str:
    text = str(expression).strip()
    if not text:
        raise ValueError("expression vector is required")
    if any(char in text for char in "\r\n;"):
        raise ValueError("invalid expression vector: control characters or command separators are not allowed")
    if any(char.isspace() for char in text):
        raise ValueError("invalid expression vector: whitespace is not allowed")
    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
        raise ValueError("invalid expression vector: delimiters are not balanced")
    return text


__all__ = [
    "VECTOR_KINDS",
    "VectorName",
    "branch_current_vector",
    "device_parameter_vector",
    "expression_vector",
    "internal_parameter_vector",
    "node_current_vector",
    "normalize_vector_name",
    "voltage_vector",
]
