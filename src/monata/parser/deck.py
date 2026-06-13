"""Import-only SPICE deck parser records.

This module intentionally stops before Monata IR projection. The parsed records
carry enough structure and source provenance for later import stages without
becoming a second public netlist model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from monata.parser.commands import SUPPORTED_DOT_COMMANDS
from monata.parser.errors import SpiceParseError, UnsupportedConstructError


_ELEMENT_KINDS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class LogicalLine:
    """A logical SPICE line after continuation/comment handling."""

    text: str
    line: int
    source_lines: tuple[int, ...]
    comment: str | None = None
    is_comment: bool = False


@dataclass(frozen=True)
class DotCommand:
    """A parsed dot command."""

    name: str
    args: tuple[str, ...]
    params: dict[str, str]
    raw: str
    line: int
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class ElementStatement:
    """A parsed element or subcircuit instance statement."""

    kind: str
    name: str
    tokens: tuple[str, ...]
    params: dict[str, str]
    raw: str
    line: int
    source_lines: tuple[int, ...]
    comment: str | None = None


@dataclass(frozen=True)
class ParsedStatement:
    """Fallback statement for comments, title, and preserved raw records."""

    kind: Literal["title", "raw", "comment"]
    text: str
    line: int
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class UnsupportedStatement:
    """Statement retained by tolerant import inspection when projection is unsupported."""

    text: str
    message: str
    line: int
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class ControlBlock:
    """A `.control` / `.endc` block with internal provenance."""

    lines: tuple[LogicalLine, ...]
    line: int
    source_lines: tuple[int, ...]


Statement = DotCommand | ElementStatement | ParsedStatement | UnsupportedStatement | ControlBlock


@dataclass(frozen=True)
class SpiceDeck:
    """Parsed deck record used as the P5 import frontend."""

    title: str
    statements: tuple[Statement, ...]
    path: str | None = None
    logical_lines: tuple[LogicalLine, ...] = field(default_factory=tuple)


def parse_spice(text: str, *, path: str | Path | None = None, strict: bool = True) -> SpiceDeck:
    """Parse SPICE text into import-only deck records."""

    source_path = str(path) if path is not None else None
    logical_lines = _logical_lines(text, path=source_path)
    title_index = _title_index(logical_lines)
    if title_index is None:
        raise SpiceParseError("SPICE deck is empty", path=source_path)

    title_line = logical_lines[title_index]
    title = _title_text(title_line.text)
    statements: list[Statement] = [ParsedStatement("title", title, title_line.line, title_line.source_lines)]
    index = 0
    while index < len(logical_lines):
        line = logical_lines[index]
        if index == title_index:
            index += 1
            continue
        if line.is_comment:
            statements.append(ParsedStatement("comment", line.text, line.line, line.source_lines))
            index += 1
            continue
        if _dot_command_name(line.text) == "control":
            block, index = _control_block(logical_lines, index, path=source_path)
            statements.append(block)
            continue
        try:
            statements.append(_parse_statement(line, path=source_path))
        except UnsupportedConstructError as exc:
            if strict:
                raise
            statements.append(
                UnsupportedStatement(
                    text=line.text,
                    message=exc.message,
                    line=line.line,
                    source_lines=line.source_lines,
                )
            )
        index += 1
    return SpiceDeck(_effective_title(title, statements), tuple(statements), path=source_path, logical_lines=tuple(logical_lines))


def _logical_lines(text: str, *, path: str | None) -> list[LogicalLine]:
    result: list[LogicalLine] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(("*", ";", "$")):
            result.append(LogicalLine(_comment_text(stripped), lineno, (lineno,), is_comment=True))
            continue
        if stripped.startswith("+"):
            if not result or result[-1].is_comment:
                raise SpiceParseError("continuation line has no previous statement", path=path, line=lineno)
            continuation, comment = _strip_inline_comment(stripped[1:].strip())
            previous = result[-1]
            text_value = f"{previous.text} {continuation}".strip()
            joined_comment = _join_comment(previous.comment, comment)
            result[-1] = LogicalLine(
                text=text_value,
                line=previous.line,
                source_lines=(*previous.source_lines, lineno),
                comment=joined_comment,
            )
            continue
        line_text, comment = _strip_inline_comment(stripped)
        if not line_text:
            continue
        result.append(LogicalLine(line_text, lineno, (lineno,), comment=comment))
    return result


def _title_index(lines: list[LogicalLine]) -> int | None:
    for index, line in enumerate(lines):
        if not line.is_comment:
            return index
    return None


def _comment_text(stripped: str) -> str:
    if stripped.startswith("*"):
        return stripped
    comment = stripped[1:].strip()
    return f"* {comment}" if comment else "*"


def _title_text(text: str) -> str:
    if _dot_command_name(text) == "title":
        parts = text.split(maxsplit=1)
        return parts[1] if len(parts) == 2 else ""
    return text


def _effective_title(initial_title: str, statements: list[Statement]) -> str:
    title = initial_title
    for statement in statements[1:]:
        if isinstance(statement, DotCommand) and statement.name == "title":
            title = " ".join(statement.args)
    return title


def _dot_command_name(text: str) -> str | None:
    if not text.startswith("."):
        return None
    return text.split(maxsplit=1)[0].lower().lstrip(".")


def _control_block(lines: list[LogicalLine], start: int, *, path: str | None) -> tuple[ControlBlock, int]:
    block_lines = [lines[start]]
    index = start + 1
    while index < len(lines):
        block_lines.append(lines[index])
        if _dot_command_name(lines[index].text) == "endc":
            source_lines = tuple(source for line in block_lines for source in line.source_lines)
            return ControlBlock(tuple(block_lines), lines[start].line, source_lines), index + 1
        index += 1
    raise SpiceParseError(".control block missing .endc", path=path, line=lines[start].line)


def _parse_statement(line: LogicalLine, *, path: str | None) -> Statement:
    if line.text.startswith("."):
        return _parse_dot_command(line, path=path)
    tokens = _tokenize(line.text, path=path, line=line.line)
    if not tokens:
        raise SpiceParseError("empty statement", path=path, line=line.line)
    name = tokens[0]
    kind = name[0].upper()
    if kind not in _ELEMENT_KINDS or not kind.isalpha():
        raise UnsupportedConstructError(f"unsupported statement: {line.text}", path=path, line=line.line)
    return ElementStatement(
        kind=kind,
        name=name,
        tokens=tuple(tokens[1:]),
        params=_split_params(tokens[1:]),
        raw=line.text,
        line=line.line,
        source_lines=line.source_lines,
        comment=line.comment,
    )


def _parse_dot_command(line: LogicalLine, *, path: str | None) -> DotCommand:
    tokens = _tokenize(line.text, path=path, line=line.line)
    if not tokens:
        raise SpiceParseError("empty dot command", path=path, line=line.line)
    name = tokens[0].lower().lstrip(".")
    if name not in SUPPORTED_DOT_COMMANDS:
        raise UnsupportedConstructError(f"unsupported dot command: .{name}", path=path, line=line.line)
    return DotCommand(
        name=name,
        args=tuple(tokens[1:]),
        params=_split_params(tokens[1:]),
        raw=line.text,
        line=line.line,
        source_lines=line.source_lines,
    )


def _tokenize(text: str, *, path: str | None, line: int) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    stack: list[str] = []
    quote: str | None = None
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {")", "]", "}"}
    for char in text:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            current.append(char)
            quote = char
            continue
        if char in pairs:
            stack.append(pairs[char])
            current.append(char)
            continue
        if char in closers:
            if not stack or stack[-1] != char:
                raise SpiceParseError(f"unbalanced delimiter {char}", path=path, line=line)
            stack.pop()
            current.append(char)
            continue
        if char == "," and not stack and "=" in current:
            current.append(char)
            continue
        if char in {",", " ", "\t"} and not stack:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)
    if quote is not None:
        raise SpiceParseError("unterminated quote", path=path, line=line)
    if stack:
        raise SpiceParseError(f"unclosed delimiter {stack[-1]}", path=path, line=line)
    if current:
        tokens.append("".join(current))
    return tokens


def _split_params(tokens: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for token in tokens:
        token = _ungroup_param_token(token)
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key:
            params[key.lower()] = value
    return params


def _ungroup_param_token(token: str) -> str:
    if len(token) >= 2 and token[0] == "(" and token[-1] == ")":
        return token[1:-1]
    return token


def _strip_inline_comment(text: str) -> tuple[str, str | None]:
    stack: list[str] = []
    quote: str | None = None
    pairs = {"(": ")", "[": "]", "{": "}"}
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in pairs:
            stack.append(pairs[char])
            continue
        if stack and char == stack[-1]:
            stack.pop()
            continue
        if char in {";", "$"} and not stack:
            return text[:index].rstrip(), text[index + 1 :].strip() or None
    return text.rstrip(), None


def _join_comment(first: str | None, second: str | None) -> str | None:
    if first and second:
        return f"{first} {second}"
    return first or second
