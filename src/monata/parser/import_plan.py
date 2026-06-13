"""SPICE import inspection records.

The plan is a public summary of how Monata will treat an input deck. It is not a
second netlist model; supported statements still project through the native IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

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
    UnsupportedStatement,
    parse_spice,
)
from monata.parser.errors import SpiceParseError, UnsupportedConstructError
from monata.parser.expression import SpiceExpression, parse_spice_expression
from monata.units import SpiceNumberDialect

if TYPE_CHECKING:
    from monata.netlist import Circuit


ImportAction = Literal["metadata", "project", "preserve", "flow", "unsupported"]
ImportKind = Literal["title", "element", "directive", "control", "comment", "unsupported"]
IssueSeverity = Literal["error", "warning"]
ControlCommandAction = Literal["analysis", "measurement", "state", "output", "side_effect", "unknown"]
SourceDependencyKind = Literal["include", "lib"]
SourceDependencyStatus = Literal["found", "missing", "unchecked"]
_CONTROL_ANALYSIS_COMMANDS = frozenset(
    {"ac", "dc", "disto", "distortion", "four", "fourier", "noise", "op", "pz", "sens", "tf", "tran"}
)
_CONTROL_MEASUREMENT_COMMANDS = frozenset({"meas", "measure"})
_CONTROL_STATE_COMMANDS = frozenset(
    {
        "alter",
        "altermod",
        "alterparam",
        "compose",
        "destroy",
        "let",
        "linearize",
        "reset",
        "resume",
        "run",
        "set",
        "stop",
        "unset",
    }
)
_CONTROL_OUTPUT_COMMANDS = frozenset(
    {"display", "echo", "listing", "plot", "print", "save", "show", "showmod"}
)
_CONTROL_SIDE_EFFECT_COMMANDS = frozenset({"exit", "hardcopy", "quit", "shell", "source", "write", "wrdata"})


@dataclass(frozen=True)
class SpiceImportIssue:
    """A diagnostic found while inspecting an import candidate."""

    severity: IssueSeverity
    message: str
    line: int | None = None
    source_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class SpiceImportStep:
    """One import-plan decision for a parsed SPICE statement."""

    action: ImportAction
    kind: ImportKind
    name: str
    raw: str
    line: int
    source_lines: tuple[int, ...]
    detail: str = ""


@dataclass(frozen=True)
class SpiceImportExpressionCheck:
    """One expression-bearing field inspected during import planning."""

    owner_kind: ImportKind
    owner_name: str
    field: str
    raw: str
    line: int
    source_lines: tuple[int, ...]
    expression: SpiceExpression | None = None
    message: str = ""

    @property
    def parsed(self) -> bool:
        return self.expression is not None


@dataclass(frozen=True)
class SpiceControlCommand:
    """One command found inside a preserved `.control` block."""

    action: ControlCommandAction
    name: str
    args: tuple[str, ...]
    raw: str
    line: int
    source_lines: tuple[int, ...]
    migratable: bool = False
    detail: str = ""


@dataclass(frozen=True)
class SpiceSourceDependency:
    """One external source file reference found while inspecting a SPICE deck."""

    kind: SourceDependencyKind
    target: str
    line: int
    source_lines: tuple[int, ...]
    resolved_path: Path | None = None
    section: str | None = None
    status: SourceDependencyStatus = "unchecked"
    search_paths: tuple[Path, ...] = ()
    detail: str = ""

    @property
    def exists(self) -> bool | None:
        if self.status == "found":
            return True
        if self.status == "missing":
            return False
        return None


@dataclass(frozen=True)
class SpiceImportPlan:
    """Inspectable SPICE import plan before native IR projection."""

    title: str
    path: str | None
    steps: tuple[SpiceImportStep, ...]
    issues: tuple[SpiceImportIssue, ...] = ()
    expression_checks: tuple[SpiceImportExpressionCheck, ...] = ()
    control_commands: tuple[SpiceControlCommand, ...] = ()
    source_dependencies: tuple[SpiceSourceDependency, ...] = ()
    _source_text: str = field(repr=False, compare=False, default="")

    @property
    def supported(self) -> bool:
        """Whether all statements are inside Monata's supported import contract."""

        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def projected_count(self) -> int:
        return self.count("project")

    @property
    def preserved_count(self) -> int:
        return self.count("preserve")

    @property
    def unsupported_count(self) -> int:
        return self.count("unsupported")

    @property
    def expression_count(self) -> int:
        return len(self.expression_checks)

    @property
    def parsed_expression_count(self) -> int:
        return sum(1 for check in self.expression_checks if check.parsed)

    @property
    def failed_expression_count(self) -> int:
        return sum(1 for check in self.expression_checks if not check.parsed)

    @property
    def control_command_count(self) -> int:
        return len(self.control_commands)

    @property
    def migratable_control_count(self) -> int:
        return sum(1 for command in self.control_commands if command.migratable)

    @property
    def source_dependency_count(self) -> int:
        return len(self.source_dependencies)

    @property
    def missing_source_dependency_count(self) -> int:
        return sum(1 for dependency in self.source_dependencies if dependency.status == "missing")

    def count(self, action: ImportAction) -> int:
        """Count import decisions by action."""

        return sum(1 for step in self.steps if step.action == action)

    def require_supported(self) -> None:
        """Raise on the first unsupported construct."""

        for issue in self.issues:
            if issue.severity == "error":
                raise UnsupportedConstructError(issue.message, path=self.path, line=issue.line)

    def to_circuit(self) -> Circuit:
        """Project the inspected deck into Monata's native Circuit IR."""

        self.require_supported()
        from monata.parser.importer import parse_spice_to_circuit

        return parse_spice_to_circuit(self._source_text, path=self.path)


def inspect_spice_import(
    text_or_path: str | Path,
    *,
    expression_dialect: SpiceNumberDialect = "standard",
    include_paths: Iterable[str | Path] = (),
) -> SpiceImportPlan:
    """Inspect how SPICE text or an explicit Path would import into Monata.

    String inputs are treated as inline SPICE text. Filesystem reads require a
    `Path`, matching the stricter import and conversion entry points.
    """

    text, source_path = _read_text_or_path(text_or_path)
    deck = parse_spice(text, path=source_path, strict=False)
    steps: list[SpiceImportStep] = []
    issues: list[SpiceImportIssue] = []
    expression_checks: list[SpiceImportExpressionCheck] = []
    control_commands: list[SpiceControlCommand] = []
    source_dependencies: list[SpiceSourceDependency] = []
    dependency_search_paths = _dependency_search_paths(source_path, include_paths)
    for statement in deck.statements:
        step = _step(statement)
        steps.append(step)
        if step.action == "unsupported":
            issues.append(
                SpiceImportIssue(
                    "error",
                    step.detail,
                    line=step.line,
                    source_lines=step.source_lines,
                )
            )
        for check in _expression_checks(statement, dialect=expression_dialect):
            expression_checks.append(check)
            if not check.parsed:
                issues.append(
                    SpiceImportIssue(
                        "warning",
                        check.message,
                        line=check.line,
                        source_lines=check.source_lines,
                    )
                )
        for command in _control_commands(statement):
            control_commands.append(command)
            if not command.migratable:
                issues.append(
                    SpiceImportIssue(
                        "warning",
                        command.detail,
                        line=command.line,
                        source_lines=command.source_lines,
                    )
                )
        for dependency in _source_dependencies(statement, search_paths=dependency_search_paths):
            source_dependencies.append(dependency)
            if dependency.status != "found":
                issues.append(
                    SpiceImportIssue(
                        "warning",
                        dependency.detail,
                        line=dependency.line,
                        source_lines=dependency.source_lines,
                    )
                )
    return SpiceImportPlan(
        title=deck.title,
        path=deck.path,
        steps=tuple(steps),
        issues=tuple(issues),
        expression_checks=tuple(expression_checks),
        control_commands=tuple(control_commands),
        source_dependencies=tuple(source_dependencies),
        _source_text=text,
    )


def _step(statement) -> SpiceImportStep:
    if isinstance(statement, ParsedStatement):
        if statement.kind == "comment":
            return SpiceImportStep(
                "preserve",
                "comment",
                "comment",
                statement.text,
                statement.line,
                statement.source_lines,
                detail="standalone comment",
            )
        return SpiceImportStep(
            "metadata",
            "title",
            statement.kind,
            statement.text,
            statement.line,
            statement.source_lines,
            detail="deck title",
        )
    if isinstance(statement, ElementStatement):
        return SpiceImportStep(
            "project",
            "element",
            statement.name,
            statement.raw,
            statement.line,
            statement.source_lines,
            detail=f"{statement.kind} element",
        )
    if isinstance(statement, DotCommand):
        return _dot_step(statement)
    if isinstance(statement, ControlBlock):
        return SpiceImportStep(
            "preserve",
            "control",
            "control",
            "\n".join(line.text for line in statement.lines),
            statement.line,
            statement.source_lines,
            detail="raw control block",
        )
    if isinstance(statement, UnsupportedStatement):
        return SpiceImportStep(
            "unsupported",
            "unsupported",
            "unsupported",
            statement.text,
            statement.line,
            statement.source_lines,
            detail=statement.message,
        )
    raise TypeError(f"unsupported parser statement type: {type(statement).__name__}")


def _dot_step(statement: DotCommand) -> SpiceImportStep:
    if statement.name == "title":
        return SpiceImportStep(
            "metadata",
            "title",
            statement.name,
            statement.raw,
            statement.line,
            statement.source_lines,
            detail="title directive",
        )
    if statement.name in STRUCTURED_DOT_COMMANDS:
        return SpiceImportStep(
            "project",
            "directive",
            statement.name,
            statement.raw,
            statement.line,
            statement.source_lines,
            detail=f".{statement.name} directive",
        )
    if statement.name in RAW_PRESERVED_DOT_COMMANDS:
        return SpiceImportStep(
            "preserve",
            "directive",
            statement.name,
            statement.raw,
            statement.line,
            statement.source_lines,
            detail=f".{statement.name} directive",
        )
    if statement.name in DOT_FLOW_COMMANDS:
        return SpiceImportStep(
            "flow",
            "directive",
            statement.name,
            statement.raw,
            statement.line,
            statement.source_lines,
            detail=f".{statement.name} scope directive",
        )
    return SpiceImportStep(
        "unsupported",
        "directive",
        statement.name,
        statement.raw,
        statement.line,
        statement.source_lines,
        detail=f"unsupported importer dot command: .{statement.name}",
    )


def _expression_checks(statement, *, dialect: SpiceNumberDialect) -> tuple[SpiceImportExpressionCheck, ...]:
    candidates = tuple(_expression_candidates(statement))
    return tuple(_check_expression(candidate, dialect=dialect) for candidate in candidates)


def _expression_candidates(statement) -> tuple[tuple[ImportKind, str, str, str, int, tuple[int, ...]], ...]:
    if isinstance(statement, ElementStatement):
        candidates: list[tuple[ImportKind, str, str, str, int, tuple[int, ...]]] = []
        for key, value in statement.params.items():
            candidates.append(("element", statement.name, f"param.{key}", value, statement.line, statement.source_lines))
        value = _element_expression_value(statement)
        if value is not None:
            candidates.append(("element", statement.name, "value", value, statement.line, statement.source_lines))
        return tuple(candidates)
    if not isinstance(statement, DotCommand):
        return ()
    candidates = [
        ("directive", statement.name, f"param.{key}", value, statement.line, statement.source_lines)
        for key, value in statement.params.items()
        if statement.name in {"csparam", "ic", "meas", "measure", "model", "nodeset", "options", "param", "subckt", "width"}
    ]
    if statement.name in {"if", "elseif"} and statement.args:
        candidates.append(("directive", statement.name, "condition", " ".join(statement.args), statement.line, statement.source_lines))
    if statement.name == "func" and len(statement.args) >= 2:
        candidates.append(("directive", statement.name, "body", statement.args[-1], statement.line, statement.source_lines))
    if statement.name == "temp":
        candidates.extend(
            ("directive", statement.name, f"arg.{index}", arg, statement.line, statement.source_lines)
            for index, arg in enumerate(statement.args)
        )
    return tuple(candidates)


def _element_expression_value(statement: ElementStatement) -> str | None:
    tokens = statement.tokens
    if statement.kind in {"R", "C", "L"} and len(tokens) >= 3 and "=" not in tokens[2]:
        return tokens[2]
    if statement.kind in {"E", "G"} and len(tokens) == 5 and "=" not in tokens[4]:
        return tokens[4]
    return None


def _check_expression(
    candidate: tuple[ImportKind, str, str, str, int, tuple[int, ...]],
    *,
    dialect: SpiceNumberDialect,
) -> SpiceImportExpressionCheck:
    owner_kind, owner_name, field, raw, line, source_lines = candidate
    try:
        expression = parse_spice_expression(raw, line=line, dialect=dialect)
    except SpiceParseError as exc:
        return SpiceImportExpressionCheck(
            owner_kind,
            owner_name,
            field,
            raw,
            line,
            source_lines,
            message=f"could not parse SPICE expression for {owner_name}.{field}: {exc.message}",
        )
    return SpiceImportExpressionCheck(
        owner_kind,
        owner_name,
        field,
        raw,
        line,
        source_lines,
        expression=expression,
    )


def _control_commands(statement) -> tuple[SpiceControlCommand, ...]:
    if not isinstance(statement, ControlBlock):
        return ()
    commands = []
    for line in statement.lines[1:-1]:
        command = _control_command(line.text, line=line.line, source_lines=line.source_lines)
        if command is not None:
            commands.append(command)
    return tuple(commands)


def _control_command(text: str, *, line: int, source_lines: tuple[int, ...]) -> SpiceControlCommand | None:
    tokens = text.split()
    if not tokens:
        return None
    name = tokens[0].lower().lstrip(".")
    args = tuple(tokens[1:])
    action, migratable, detail = _control_command_classification(name)
    return SpiceControlCommand(action, name, args, text, line, source_lines, migratable=migratable, detail=detail)


def _control_command_classification(name: str) -> tuple[ControlCommandAction, bool, str]:
    if name in _CONTROL_ANALYSIS_COMMANDS:
        return (
            "analysis",
            True,
            f".control {name} command can be migrated to a backend-neutral analysis task",
        )
    if name in _CONTROL_MEASUREMENT_COMMANDS:
        return (
            "measurement",
            True,
            f".control {name} command can be migrated to a simulator measure directive",
        )
    if name in _CONTROL_STATE_COMMANDS:
        return (
            "state",
            False,
            f".control {name} command mutates simulator state and is preserved as raw control text",
        )
    if name in _CONTROL_OUTPUT_COMMANDS:
        return (
            "output",
            False,
            f".control {name} command is preserved as raw simulator output text",
        )
    if name in _CONTROL_SIDE_EFFECT_COMMANDS:
        return (
            "side_effect",
            False,
            f".control {name} command has external side effects and is preserved as raw control text",
        )
    return (
        "unknown",
        False,
        f".control {name} command is not classified by Monata import inspection",
    )


def _source_dependencies(
    statement,
    *,
    search_paths: tuple[Path, ...],
) -> tuple[SpiceSourceDependency, ...]:
    if not isinstance(statement, DotCommand) or statement.name not in {"include", "lib"} or not statement.args:
        return ()
    kind: SourceDependencyKind = "include" if statement.name == "include" else "lib"
    target = _unquote(statement.args[0])
    section = statement.args[1] if kind == "lib" and len(statement.args) > 1 else None
    dependency = _resolve_source_dependency(
        kind,
        target,
        section=section,
        line=statement.line,
        source_lines=statement.source_lines,
        search_paths=search_paths,
    )
    return (dependency,)


def _resolve_source_dependency(
    kind: SourceDependencyKind,
    target: str,
    *,
    section: str | None,
    line: int,
    source_lines: tuple[int, ...],
    search_paths: tuple[Path, ...],
) -> SpiceSourceDependency:
    target_path = Path(target).expanduser()
    if target_path.is_absolute():
        exists = target_path.is_file()
        return SpiceSourceDependency(
            kind,
            target,
            line,
            source_lines,
            resolved_path=target_path if exists else target_path,
            section=section,
            status="found" if exists else "missing",
            search_paths=(target_path.parent,),
            detail=_dependency_detail(kind, target, "found" if exists else "missing", (target_path.parent,), section=section),
        )
    if not search_paths:
        return SpiceSourceDependency(
            kind,
            target,
            line,
            source_lines,
            section=section,
            status="unchecked",
            detail=_dependency_detail(kind, target, "unchecked", (), section=section),
        )
    for root in search_paths:
        candidate = root / target_path
        if candidate.is_file():
            return SpiceSourceDependency(
                kind,
                target,
                line,
                source_lines,
                resolved_path=candidate,
                section=section,
                status="found",
                search_paths=search_paths,
                detail=_dependency_detail(kind, target, "found", search_paths, section=section),
            )
    return SpiceSourceDependency(
        kind,
        target,
        line,
        source_lines,
        section=section,
        status="missing",
        search_paths=search_paths,
        detail=_dependency_detail(kind, target, "missing", search_paths, section=section),
    )


def _dependency_search_paths(source_path: str | None, include_paths: Iterable[str | Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    if source_path is not None:
        paths.append(Path(source_path).parent)
    paths.extend(Path(path) for path in include_paths)
    return tuple(_unique_paths(paths))


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.expanduser()
        try:
            key = normalized.resolve(strict=False)
        except OSError:
            key = normalized.absolute()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return tuple(result)


def _dependency_detail(
    kind: SourceDependencyKind,
    target: str,
    status: SourceDependencyStatus,
    search_paths: tuple[Path, ...],
    *,
    section: str | None,
) -> str:
    suffix = f" section {section!r}" if section else ""
    if status == "found":
        return f".{kind} dependency {target!r}{suffix} resolves on the import search path"
    if status == "missing":
        searched = ", ".join(str(path) for path in search_paths) or "<none>"
        return f".{kind} dependency {target!r}{suffix} was not found; searched: {searched}"
    return f".{kind} dependency {target!r}{suffix} was not checked because no source path or include_paths were provided"


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_text_or_path(text_or_path: str | Path) -> tuple[str, str | None]:
    if isinstance(text_or_path, Path):
        return text_or_path.read_text(), str(text_or_path)
    return text_or_path, None


__all__ = [
    "ControlCommandAction",
    "ImportAction",
    "ImportKind",
    "IssueSeverity",
    "SpiceControlCommand",
    "SpiceImportExpressionCheck",
    "SpiceImportIssue",
    "SpiceImportPlan",
    "SpiceImportStep",
    "SourceDependencyKind",
    "SourceDependencyStatus",
    "SpiceSourceDependency",
    "inspect_spice_import",
]
