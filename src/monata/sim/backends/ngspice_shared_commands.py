"""Command formatting and output parsing helpers for libngspice sessions."""

from __future__ import annotations


def command_part(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    if any(char in text for char in "\r\n;"):
        raise ValueError(f"{label} cannot contain control characters")
    return text


def command_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return command_part(value, "command value")


def join_command_outputs(outputs: list[str]) -> str:
    return "\n".join(output for output in outputs if output)


def parse_key_value_output(output: str) -> dict[str, object]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    result: dict[str, object] = {}
    if lines:
        result["description"] = lines[0]
    for line in lines:
        parsed = parse_key_value_line(line)
        if parsed is not None:
            key, value = parsed
            result[key] = coerce_ngspice_value(value)
    return result


def parse_key_value_line(line: str) -> tuple[str, str] | None:
    for separator in (" = ", ": ", "="):
        if separator not in line:
            continue
        key, value = line.split(separator, 1)
        key = key.strip()
        if key:
            return key, value.strip()
    return None


def coerce_ngspice_value(value: str) -> object:
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
