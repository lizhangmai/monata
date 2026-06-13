"""Internal vector-name identity helpers shared by simulation modules."""

from __future__ import annotations

import keyword
import re


def safe_lookup_name(name: str) -> str:
    value = re.sub(r"\W+", "_", str(name).strip().lower()).strip("_")
    if not value:
        return "waveform"
    if value[0].isdigit():
        value = f"_{value}"
    return value


def is_safe_attribute_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("_")


def simple_vector_inner(name: str, function: str) -> str | None:
    text = str(name).strip()
    prefix = f"{function.lower()}("
    if not text.lower().startswith(prefix) or not text.endswith(")"):
        return None
    inner = text[len(prefix) : -1]
    if not inner or any(char in inner for char in "()"):
        return None
    return inner


def looks_like_expression_vector(name: str) -> bool:
    text = str(name).strip()
    if not text:
        return False
    if text.startswith("@"):
        return False
    return "(" in text or any(operator in text for operator in ("*", "/"))


def device_parameter_from_vector(name: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"@([^\[\]]+)\[([^\[\]]+)\]", str(name).strip())
    if match is None:
        return None
    element = match.group(1).strip()
    parameter = match.group(2).strip()
    if not element or not parameter:
        return None
    return element, parameter


def pole_zero_vector_kind(name: str) -> str:
    lower = str(name).strip().lower()
    if "zero" in lower or lower.startswith("zer"):
        return "zero"
    return "pole"


def p3_vector_metadata(analysis_name: str | None, name: str, vector_kind: str) -> dict[str, str]:
    if analysis_name == "sens":
        return _sensitivity_vector_metadata(name)
    if analysis_name == "pz":
        return {"pole_zero_kind": vector_kind}
    if analysis_name == "disto":
        return {"distortion_vector": name}
    return {}


def _sensitivity_vector_metadata(name: str) -> dict[str, str]:
    inner = _voltage_inner(name)
    if inner is None:
        return {"sensitivity_vector": name}
    metadata = {"sensitivity_vector": name}
    if ":" in inner:
        element, parameter = inner.split(":", 1)
    elif "_" in inner:
        element, parameter = inner.split("_", 1)
    else:
        element, parameter = inner, ""
    if element:
        metadata["sensitivity_element"] = element
    if parameter:
        metadata["sensitivity_parameter"] = parameter
    return metadata


def _voltage_inner(name: str) -> str | None:
    text = str(name).strip()
    if not text.lower().startswith("v(") or not text.endswith(")"):
        return None
    inner = text[2:-1].strip()
    return inner or None
