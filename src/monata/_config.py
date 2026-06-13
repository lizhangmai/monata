"""Internal TOML persistence helpers for Monata workspace metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from monata._paths import toml_string


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def reject_unknown_fields(
    data: Mapping[str, Any], allowed: frozenset[str], subject: str
) -> None:
    unknown = sorted(key for key in data if key not in allowed)
    if unknown:
        raise ValueError(f"{subject} has unknown fields: {', '.join(unknown)}")


def cell_config(name: str, *, description: str = "") -> dict[str, Any]:
    return {"cell": {"name": name, "description": description}, "views": {}}


def category_config(name: str, *, description: str = "") -> dict[str, Any]:
    return {"category": {"name": name, "description": description}}


def write_category_config(path: Path, config: Mapping[str, Any]) -> None:
    category = config["category"]
    lines = [f'[category]\nname = "{toml_string(category["name"])}"\n']
    if "description" in category:
        lines.append(f'description = "{toml_string(category["description"])}"\n')
    path.write_text("".join(lines))


def write_cell_config(path: Path, config: Mapping[str, Any]) -> None:
    cell = config["cell"]
    lines = [f'[cell]\nname = "{toml_string(cell["name"])}"\n']
    if "description" in cell:
        lines.append(f'description = "{toml_string(cell["description"])}"\n')
    lines.append("\n[views]\n")
    for view_type, view_config in config.get("views", {}).items():
        parts = ", ".join(f"{key} = {toml_value(value)}" for key, value in view_config.items())
        lines.append(f"{view_type} = {{ {parts} }}\n")
    path.write_text("".join(lines))


def write_library_config(
    path: Path,
    *,
    name: str,
    tech_model_paths: Iterable[Any],
    description: str = "",
    techlib_attachments: Iterable[str] = (),
    default_corner: str | None = None,
) -> None:
    model_paths = ", ".join(toml_value(model_path) for model_path in tech_model_paths)
    lines = [
        f'[library]\nname = "{toml_string(name)}"\n',
        f'description = "{toml_string(description)}"\n\n',
        f"[technology]\nmodel_paths = [{model_paths}]\n",
    ]
    attachments = list(techlib_attachments)
    if attachments or default_corner is not None:
        techlibs = ", ".join(toml_value(attachment) for attachment in attachments)
        lines.extend(["\n[attachments]\n", f"techlibs = [{techlibs}]\n"])
        if default_corner is not None:
            lines.append(f'default_corner = "{toml_string(default_corner)}"\n')
    path.write_text("".join(lines))


def write_project_config(
    path: Path,
    *,
    name: str,
    libraries: Iterable[Mapping[str, str]] = (),
) -> None:
    lines = [f'[project]\nname = "{toml_string(name)}"\n']
    for entry in libraries:
        lines.extend([
            "\n[[libraries]]\n",
            f'name = "{toml_string(entry["name"])}"\n',
            f'path = "{toml_string(entry["path"])}"\n',
        ])
    path.write_text("".join(lines))


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return f'"{toml_string(value)}"'
