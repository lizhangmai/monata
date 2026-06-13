"""Projection from parsed SPICE decks into Monata artifacts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import keyword
from pathlib import Path
from typing import Any

from monata._paths import toml_string, validate_path_segment
from monata.netlist import Circuit, SourceValue, SubCircuit, render_ngspice
from monata.netlist.ir import Element, NetlistError
from monata.parser.commands import (
    DOT_FLOW_COMMANDS,
    RAW_PRESERVED_DOT_COMMANDS,
    STRUCTURED_DOT_COMMANDS,
)
from monata.parser.deck import (
    ControlBlock,
    DotCommand,
    ElementStatement,
    ParsedStatement,
    SpiceDeck,
    parse_spice,
)
from monata.parser.errors import SpiceParseError, UnsupportedConstructError

IMPORTED_NETLIST_ENTRY = "netlist.cir"
IMPORT_METADATA_ENTRY = "import.toml"
ASSETS_DIRNAME = "assets"


@dataclass(frozen=True)
class ImportedAsset:
    """A reusable imported SPICE asset stored in a Monata library."""

    kind: str
    name: str
    path: Path
    library_name: str


def parse_spice_to_circuit(text: str, *, path: str | Path | None = None) -> Circuit:
    """Project supported parsed SPICE text into Monata's native Circuit IR."""

    deck = parse_spice(text, path=path)
    return _project_deck(deck)


def import_spice_deck(
    project,
    text_or_path: str | Path,
    *,
    library_name: str = "imported",
    cell_name: str | None = None,
) -> Any:
    """Import SPICE text or an explicit Path as a Cell with a generated netlist view."""

    text, source_path = _read_text_or_path(text_or_path)
    circuit = parse_spice_to_circuit(text, path=source_path)
    lib = _ensure_library(project, library_name)
    safe_cell_name = _safe_import_name(cell_name or _default_name(source_path, circuit.title), "imported cell name")
    cell = lib.create_cell(safe_cell_name, description=f"Imported from {source_path or 'SPICE text'}")
    _write_imported_netlist_view(cell, circuit)
    _write_import_metadata(cell, source_path=source_path, title=circuit.title)
    return cell


def import_spice_asset(
    project,
    text_or_path: str | Path,
    *,
    library_name: str = "imported",
    asset_name: str | None = None,
) -> ImportedAsset:
    """Import subcircuit/model SPICE text or an explicit Path as a reusable asset."""

    text, source_path = _read_text_or_path(text_or_path)
    kind = _asset_kind(text, path=source_path)
    lib = _ensure_library(project, library_name)
    safe_asset_name = _safe_import_name(asset_name or _default_name(source_path, kind), "imported asset name")
    asset_path = _imported_asset_path(lib, safe_asset_name)
    asset_path.write_text(text if text.endswith("\n") else f"{text}\n")
    return ImportedAsset(kind=kind, name=safe_asset_name, path=asset_path, library_name=library_name)


def spice_to_python(
    text_or_path: str | Path,
    *,
    function_name: str = "build",
) -> str:
    """Convert supported SPICE text or an explicit Path into Python IR builder code."""

    _validate_function_name(function_name)
    text, source_path = _read_text_or_path(text_or_path)
    circuit = parse_spice_to_circuit(text, path=source_path)
    lines = [
        "from collections import OrderedDict",
        "from monata.netlist import Circuit, Element, SourceValue, SubCircuit",
        "",
        "",
        f"def {function_name}():",
        f"    circuit = Circuit({circuit.title!r})",
    ]
    for name, value in circuit.params.items():
        lines.append(f"    circuit.param({name!r}, {str(value)!r})")
    for include in circuit.includes:
        lines.append(f"    circuit.include({include!r})")
    for directive in circuit.directives:
        lines.append(f"    {_directive_python('circuit', directive)}")
    for subckt in circuit.subcircuits:
        lines.extend(_subckt_python(subckt))
    for element in circuit.elements:
        lines.append(f"    {_element_python('circuit', element)}")
    for output in circuit.outputs:
        lines.append(f"    circuit.output({output!r})")
    lines.append("    return circuit")
    return "\n".join(lines) + "\n"


def _project_deck(deck: SpiceDeck) -> Circuit:
    circuit = Circuit(deck.title)
    scopes: list[Circuit | SubCircuit] = [circuit]
    for statement in deck.statements[1:]:
        if isinstance(statement, ControlBlock):
            for line in statement.lines:
                scopes[-1].raw_directive(line.text)
            continue
        if isinstance(statement, ParsedStatement):
            if statement.kind == "comment":
                scopes[-1].raw_directive(statement.text)
            continue
        if isinstance(statement, DotCommand):
            if statement.name not in DOT_FLOW_COMMANDS:
                _apply_dot(scopes[-1], statement, path=deck.path)
                continue
            if statement.name == "end":
                continue
            if statement.name == "subckt":
                if len(statement.args) < 2:
                    raise SpiceParseError(".subckt requires name and nodes", path=deck.path, line=statement.line)
                name, nodes, params = _subckt_signature(statement)
                subckt = SubCircuit(name, nodes)
                for key, value in params.items():
                    subckt.param(key, value)
                scopes.append(subckt)
                continue
            if len(scopes) == 1:
                raise SpiceParseError(".ends without .subckt", path=deck.path, line=statement.line)
            subckt = scopes.pop()
            if isinstance(subckt, SubCircuit):
                circuit.subckt(subckt)
            continue
        if isinstance(statement, ElementStatement):
            element = _project_element(statement, path=deck.path)
            if statement.comment is not None:
                element = element.clone(comment=statement.comment)
            scopes[-1].add(element)
    if len(scopes) != 1:
        open_scope = scopes[-1]
        name = getattr(open_scope, "name", "unknown")
        raise SpiceParseError(f"subcircuit missing .ends: {name}", path=deck.path)
    return circuit


def _apply_dot(scope: Circuit | SubCircuit, statement: DotCommand, *, path: str | None) -> None:
    name = statement.name
    args = statement.args
    params = statement.params
    if name in RAW_PRESERVED_DOT_COMMANDS:
        scope.raw_directive(statement.raw)
    elif name not in STRUCTURED_DOT_COMMANDS:
        raise UnsupportedConstructError(f"unsupported importer dot command: .{name}", path=path, line=statement.line)
    elif name == "include" and args:
        scope.include(_unquote(args[0]))
    elif name == "param":
        if params:
            for key, value in params.items():
                scope.param(key, value)
        elif len(args) >= 2:
            scope.param(args[0], args[1])
    elif name == "model" and len(args) >= 2:
        scope.model(args[0], args[1], **params)
    elif name == "lib" and args:
        scope.lib(_unquote(args[0]), args[1] if len(args) > 1 else None)
    elif name == "title":
        if isinstance(scope, Circuit):
            scope.title = " ".join(args)
        else:
            scope.raw_directive(statement.raw)
    elif name == "global":
        scope.global_(*args)
    elif name == "nodeset":
        scope.nodeset(**_node_params(params))
    elif name == "ic":
        scope.ic(**_node_params(params))
    elif name in {"option", "options"}:
        scope.options(*_directive_flags(args), **params)
    elif name == "save":
        scope.save(*args)
    elif name == "probe":
        scope.probe(*args)
    elif name == "print" and args:
        scope.print_(args[0], *args[1:])
    elif name in {"meas", "measure"} and len(args) >= 3:
        scope.measure(args[0], args[1], " ".join(args[2:]))
    else:
        scope.raw_directive(statement.raw)


def _project_element(statement: ElementStatement, *, path: str | None) -> Element:
    name = _element_name(statement)
    tokens = list(statement.tokens)
    positional = [token for token in tokens if "=" not in token]
    params = statement.params
    kind = statement.kind
    try:
        if kind in {"V", "I"}:
            source_tokens = tokens[2:]
            value = _source_value(source_tokens)
            projected_params = _source_element_params(source_tokens, params)
            return Element(kind, name, tuple(tokens[:2]), value=value, params=_params(projected_params))
        if kind in {"R", "C", "L"}:
            rest = positional[2:]
            value = None
            model = None
            if len(rest) == 1:
                value = rest[0]
            elif len(rest) == 2:
                value = rest[0]
                model = rest[1]
            elif rest:
                value = " ".join(rest)
            return Element(kind, name, tuple(positional[:2]), value=value, model=model, params=_params(params))
        if kind in {"D", "J", "M", "Q", "Z"}:
            nodes, model, tail = _semiconductor_element_parts(kind, positional)
            return Element(kind, name, nodes, model=model, params=_params(_semiconductor_params(kind, params, tail)))
        if kind in {"A", "N", "O", "P", "U", "X", "Y"}:
            if kind == "X":
                model = positional[-1] if positional else None
                nodes = positional[:-1]
            else:
                node_count = {
                    "O": 4,
                    "U": 3,
                    "Y": 4,
                }.get(kind)
                nodes = positional[:node_count] if node_count is not None else positional[:-1]
                model = positional[node_count] if node_count is not None and len(positional) > node_count else (
                    positional[-1] if positional else None
                )
            return Element(kind, name, tuple(nodes), model=model, params=_params(params))
        if kind == "S":
            return Element(
                kind,
                name,
                tuple(positional[:4]),
                model=positional[4] if len(positional) > 4 else None,
                params=_params(_switch_params(params, tuple(positional[5:]))),
            )
        if kind in {"E", "G"}:
            nodes, value, projected_params = _controlled_source_parts(positional, params)
            return Element(kind, name, nodes, value=value, params=_params(projected_params))
        if kind in {"F", "H"}:
            return Element(
                kind,
                name,
                tuple(positional[:2]),
                model=positional[2] if len(positional) > 2 else None,
                value=" ".join(positional[3:]) or None,
                params=_params(params),
            )
        if kind == "W":
            return Element(
                kind,
                name,
                tuple(positional[:2]),
                value=positional[2] if len(positional) > 2 else None,
                model=positional[3] if len(positional) > 3 else None,
                params=_params(_switch_params(params, tuple(positional[4:]))),
            )
        if kind == "B":
            nodes, value, projected_params = _behavioral_element_parts(tokens, positional, params)
            return Element(kind, name, nodes, value=value, params=_params(projected_params))
        if kind == "K":
            return Element(kind, name, tuple(positional[:2]), value=" ".join(positional[2:]) or None, params=_params(params))
        if kind == "T":
            return Element(kind, name, tuple(tokens[:4]), value=" ".join(tokens[4:]) or None)
    except NetlistError as exc:
        raise SpiceParseError(str(exc), path=path, line=statement.line) from exc
    raise UnsupportedConstructError(f"unsupported element kind: {kind}", path=path, line=statement.line)


def _semiconductor_element_parts(kind: str, positional: list[str]) -> tuple[tuple[str, ...], str | None, tuple[str, ...]]:
    if kind == "D":
        return tuple(positional[:2]), positional[2] if len(positional) > 2 else None, tuple(positional[3:])
    if kind in {"J", "Z"}:
        return tuple(positional[:3]), positional[3] if len(positional) > 3 else None, tuple(positional[4:])
    if kind == "M":
        return tuple(positional[:4]), positional[4] if len(positional) > 4 else None, tuple(positional[5:])
    if len(positional) >= 5 and positional[4].lower() != "off":
        return tuple(positional[:4]), positional[4], tuple(positional[5:])
    return tuple(positional[:3]), positional[3] if len(positional) > 3 else None, tuple(positional[4:])


def _semiconductor_params(kind: str, params: dict[str, str], tail: tuple[str, ...]) -> dict[str, str | bool]:
    result: dict[str, str | bool] = dict(params)
    positional_area_allowed = kind in {"D", "J", "Z"}
    for token in tail:
        if token.lower() == "off":
            result.setdefault("off", True)
        elif positional_area_allowed and "area" not in result:
            result["area"] = token
    return result


def _switch_params(params: dict[str, str], tail: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = dict(params)
    if tail:
        result.setdefault("initial_state", " ".join(tail))
    return result


def _behavioral_element_parts(
    tokens: list[str],
    positional: list[str],
    params: dict[str, str],
) -> tuple[tuple[str, ...], str | None, dict[str, Any]]:
    result: dict[str, Any] = dict(params)
    value = " ".join(positional[2:]) or None
    for key in list(result):
        if key.lower() in {"i", "v"} and value is None:
            value = f"{key}={result.pop(key)}"
            break
    return tuple(tokens[:2]), value, result


def _controlled_source_parts(
    positional: list[str],
    params: dict[str, str],
) -> tuple[tuple[str, ...], str | None, dict[str, Any]]:
    if len(positional) >= 3 and _is_nonlinear_controlled_source_head(positional[2]):
        return tuple(positional[:2]), " ".join(positional[2:]) or None, dict(params)
    if len(positional) >= 5:
        return tuple(positional[:4]), " ".join(positional[4:]) or None, dict(params)

    result: dict[str, Any] = dict(params)
    value = " ".join(positional[2:]) or None
    for key in list(result):
        if key.lower() in {"value", "vol", "cur"} and value is None:
            value = f"{key}={result.pop(key)}"
            break
    return tuple(positional[:2]), value, result


def _is_nonlinear_controlled_source_head(token: str) -> bool:
    upper = token.upper()
    return upper.startswith("POLY") or upper in {"LAPLACE", "TABLE"}


def _asset_kind(text: str, *, path: str | None) -> str:
    deck = parse_spice(_asset_deck_text(text), path=path)
    commands = [s for s in deck.statements if isinstance(s, DotCommand)]
    has_subckt = any(command.name == "subckt" for command in commands)
    has_model = any(command.name == "model" for command in commands)
    element_count = sum(isinstance(s, ElementStatement) for s in deck.statements)
    if has_subckt:
        return "subckt"
    if has_model and element_count == 0:
        return "model"
    raise UnsupportedConstructError("asset import expects subcircuit-only or model-only SPICE text", path=path)


def _subckt_python(subckt: SubCircuit) -> list[str]:
    var_name = f"subckt_{_python_identifier_suffix(subckt.name)}"
    lines = [f"    {var_name} = SubCircuit({subckt.name!r}, {tuple(subckt.nodes)!r})"]
    for name, value in subckt.params.items():
        lines.append(f"    {var_name}.param({name!r}, {str(value)!r})")
    for include in subckt.includes:
        lines.append(f"    {var_name}.include({include!r})")
    for directive in subckt.directives:
        lines.append(f"    {_directive_python(var_name, directive)}")
    for element in subckt.elements:
        lines.append(f"    {_element_python(var_name, element)}")
    lines.append(f"    circuit.subckt({var_name})")
    return lines


def _directive_python(target: str, directive) -> str:
    if directive.raw:
        return f"{target}.raw_directive({directive.name!r})"
    args = ", ".join(repr(str(arg)) for arg in directive.args)
    params = f"**OrderedDict({list((str(k), str(v)) for k, v in directive.params.items())!r})"
    pieces = [repr(directive.name)]
    if args:
        pieces.append(args)
    if directive.params:
        pieces.append(params)
    return f"{target}.directive({', '.join(pieces)})"


def _element_python(target: str, element: Element) -> str:
    params = f"OrderedDict({list((str(k), str(v)) for k, v in element.params.items())!r})"
    return (
        f"{target}.add(Element({element.kind!r}, {element.name!r}, {tuple(element.nodes)!r}, "
        f"value={_value_python(element.value)}, "
        f"model={element.model!r}, params={params}, comment={element.comment!r}))"
    )


def _value_python(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, SourceValue):
        if value.params:
            params = f"OrderedDict({list((str(k), str(v)) for k, v in value.params.items())!r})"
            return f"SourceValue({value.form!r}, {tuple(value.values)!r}, {params})"
        return f"SourceValue({value.form!r}, {tuple(value.values)!r})"
    return repr(str(value))


def _asset_deck_text(text: str) -> str:
    stripped = text.lstrip()
    if stripped.lower().startswith((".subckt", ".model")):
        return f"Imported asset\n{text}"
    return text


def _read_text_or_path(text_or_path: str | Path) -> tuple[str, str | None]:
    if isinstance(text_or_path, Path):
        return text_or_path.read_text(), str(text_or_path)
    return text_or_path, None


def _write_imported_netlist_view(cell, circuit: Circuit) -> Path:
    netlist_path = cell.path / IMPORTED_NETLIST_ENTRY
    netlist_path.write_text(render_ngspice(circuit))
    cell.create_view("netlist", entry=netlist_path.name)
    return netlist_path


def _write_import_metadata(cell, *, source_path: str | None, title: str) -> Path:
    metadata_path = cell.path / IMPORT_METADATA_ENTRY
    metadata_path.write_text(
        "[import]\n"
        f'source = "{toml_string(source_path or "<memory>")}"\n'
        f'title = "{toml_string(title)}"\n'
    )
    return metadata_path


def _imported_asset_path(lib, asset_name: str) -> Path:
    assets_dir = lib.path / ASSETS_DIRNAME
    assets_dir.mkdir(exist_ok=True)
    return assets_dir / f"{asset_name}.cir"


def _ensure_library(project, name: str):
    if name in project.list_libraries():
        return project.get_library(name)
    return project.create_library(name)


def _element_name(statement: ElementStatement) -> str:
    return statement.name


def _subckt_signature(statement: DotCommand) -> tuple[str, tuple[str, ...], OrderedDict[str, Any]]:
    name = statement.args[0]
    nodes: list[str] = []
    params: OrderedDict[str, Any] = OrderedDict(statement.params)
    in_params = False
    for token in statement.args[1:]:
        marker = token.lower().rstrip(":")
        if marker == "params":
            in_params = True
            continue
        if in_params or "=" in token:
            continue
        nodes.append(token)
    return name, tuple(nodes), params


def _params(params: dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(params)


def _directive_flags(args: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(arg for arg in args if "=" not in arg)


def _node_params(params: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in params.items():
        if key.lower().startswith("v(") and key.endswith(")"):
            result[key[2:-1]] = value
        else:
            result[key] = value
    return result


def _source_value(tokens: list[str]) -> SourceValue | str | None:
    if not tokens:
        return None
    if tokens[0].upper() == "DC" and len(tokens) >= 3:
        form, values = _source_form_values(tokens[2], tokens[3:])
        if form == "PWL":
            return _pwl_source_value(values, dc=tokens[1], fallback=tokens)
    form, values = _source_form_values(tokens[0], tokens[1:])
    if form is None:
        return " ".join(tokens)
    try:
        if form == "DC":
            return SourceValue("DC", (values[0],)) if len(values) == 1 else " ".join(tokens)
        if form == "AC":
            return SourceValue("AC", tuple(values)) if len(values) == 2 else " ".join(tokens)
        if form == "PWL":
            return _pwl_source_value(values, fallback=tokens)
        return SourceValue(form, tuple(values))
    except NetlistError:
        return " ".join(tokens)


def _source_element_params(source_tokens: list[str], params: dict[str, Any]) -> OrderedDict[str, Any]:
    expression = _source_expression_text(source_tokens).lower()
    result: OrderedDict[str, Any] = OrderedDict()
    for key, value in params.items():
        assignment = f"{key}={value}".lower()
        if assignment and assignment in expression:
            continue
        result[key] = value
    return result


def _source_expression_text(source_tokens: list[str]) -> str:
    if not source_tokens:
        return ""
    first = source_tokens[0].upper()
    if first == "DC" and len(source_tokens) >= 3 and source_tokens[2].upper().startswith("PWL"):
        return " ".join(source_tokens[:3])
    if "(" in source_tokens[0] and source_tokens[0].endswith(")"):
        return source_tokens[0]
    return " ".join(source_tokens)


def _pwl_source_value(
    values: list[str],
    *,
    dc: str | None = None,
    fallback: list[str],
) -> SourceValue | str:
    pwl_values: list[str] = []
    params: OrderedDict[str, str] = OrderedDict()
    if dc is not None:
        params["dc"] = dc
    for value in values:
        key, separator, item = value.partition("=")
        if separator and key.lower() in {"r", "td"}:
            params[key] = item
        else:
            pwl_values.append(value)
    try:
        return SourceValue("PWL", tuple(pwl_values), params)
    except NetlistError:
        return " ".join(fallback)


def _source_form_values(first: str, rest: list[str]) -> tuple[str | None, list[str]]:
    upper_first = first.upper()
    if upper_first == "DC" and len(rest) >= 3 and rest[1].upper() == "AC":
        return "AC", [rest[0], rest[2]]
    if upper_first in {"DC", "AC", "PULSE", "SIN", "EXP", "PWL", "SFFM", "AM", "TRRANDOM"}:
        values = ["0", *rest] if upper_first == "AC" and len(rest) == 1 else rest
        return upper_first, values
    if "(" in first and first.endswith(")"):
        form, raw_values = first.split("(", 1)
        upper_form = form.upper()
        if upper_form in {"PULSE", "SIN", "EXP", "PWL", "SFFM", "AM", "TRRANDOM"}:
            return upper_form, _tokenize_source_values(raw_values[:-1])
    return None, []


def _tokenize_source_values(text: str) -> list[str]:
    values: list[str] = []
    current: list[str] = []
    stack = 0
    for char in text:
        if char == "(":
            stack += 1
        elif char == ")" and stack:
            stack -= 1
        if char in {",", " ", "\t"} and not stack:
            if current:
                values.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        values.append("".join(current))
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _default_name(path: str | None, fallback: str) -> str:
    if path:
        return Path(path).stem
    return fallback or "imported"


def _safe_import_name(value: str, label: str) -> str:
    result = "".join(
        char if char.isascii() and (char.isalnum() or char in "._-") else "_"
        for char in str(value).strip()
    )
    result = result.strip("._-") or "imported"
    return validate_path_segment(result, label)


def _python_identifier_suffix(value: str) -> str:
    result = "".join(
        char if char.isascii() and (char.isalnum() or char == "_") else "_"
        for char in str(value).strip()
    )
    return result.strip("_") or "imported"


def _validate_function_name(function_name: str) -> None:
    text = str(function_name)
    if not text.isidentifier() or keyword.iskeyword(text):
        raise ValueError(f"invalid function_name: {function_name}")
