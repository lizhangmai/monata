"""SPICE expression records and parser.

This parser is an import frontend. It preserves expression structure for
inspection, validation, and later projection without becoming a second netlist
IR or an evaluator.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import NoReturn

from monata.parser.errors import SpiceParseError
from monata.units import SpiceNumberDialect, UnitError, parse_spice_number


class SpiceExpression:
    """Base class for parsed SPICE expression records."""


@dataclass(frozen=True)
class SpiceNumber(SpiceExpression):
    text: str
    value: float


@dataclass(frozen=True)
class SpiceIdentifier(SpiceExpression):
    name: str


@dataclass(frozen=True)
class SpiceBranch(SpiceExpression):
    source: str


@dataclass(frozen=True)
class SpiceInternalParameter(SpiceExpression):
    element: str
    parameter: str


@dataclass(frozen=True)
class SpiceUnary(SpiceExpression):
    operator: str
    operand: SpiceExpression


@dataclass(frozen=True)
class SpiceBinary(SpiceExpression):
    operator: str
    left: SpiceExpression
    right: SpiceExpression


@dataclass(frozen=True)
class SpiceTernary(SpiceExpression):
    condition: SpiceExpression
    when_true: SpiceExpression
    when_false: SpiceExpression


@dataclass(frozen=True)
class SpiceCall(SpiceExpression):
    name: str
    args: tuple[SpiceExpression, ...] = ()


@dataclass(frozen=True)
class SpiceVector(SpiceExpression):
    items: tuple[SpiceExpression, ...] = ()


@dataclass(frozen=True)
class SpiceGroup(SpiceExpression):
    kind: str
    expression: SpiceExpression


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


_BINARY_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<>": 3,
    "<=": 3,
    ">=": 3,
    "<": 3,
    ">": 3,
    "+": 4,
    "-": 4,
    "*": 5,
    "/": 5,
    "%": 5,
    "\\": 5,
    "**": 6,
    "^": 6,
}
_RIGHT_ASSOCIATIVE = {"**", "^"}
_UNARY_OPERATORS = {"-", "!"}
_MULTI_CHAR_TOKENS = ("#branch", "**", "&&", "||", "==", "!=", "<>", "<=", ">=")
_SINGLE_CHAR_TOKENS = set("+-*/%\\^<>(){}[],:?!@")
_GROUP_DELIMITERS = {
    "(": (")", "parenthesis"),
    "{": ("}", "brace"),
    "'": ("'", "quote"),
    '"': ('"', "quote"),
}


def parse_spice_expression(
    text: str,
    *,
    path: str | None = None,
    line: int | None = None,
    dialect: SpiceNumberDialect = "standard",
) -> SpiceExpression:
    """Parse a SPICE math/vector expression into lightweight records."""

    parser = _ExpressionParser(
        _tokenize(text, path=path, line=line, dialect=dialect),
        path=path,
        line=line,
        dialect=dialect,
    )
    expression = parser.parse()
    parser.expect("<eof>")
    return expression


def render_spice_expression(expression: SpiceExpression) -> str:
    """Render a parsed expression in stable SPICE syntax."""

    return _render(expression, parent_precedence=0)


def walk_spice_expression(expression: SpiceExpression) -> Iterator[SpiceExpression]:
    """Yield an expression tree in pre-order."""

    yield expression
    for child in _children(expression):
        yield from walk_spice_expression(child)


class _ExpressionParser:
    def __init__(
        self,
        tokens: tuple[_Token, ...],
        *,
        path: str | None,
        line: int | None,
        dialect: SpiceNumberDialect,
    ) -> None:
        self._tokens = tokens
        self._index = 0
        self._path = path
        self._line = line
        self._dialect: SpiceNumberDialect = dialect

    def parse(self) -> SpiceExpression:
        if self.peek().kind == "<eof>":
            self.error("empty SPICE expression")
        return self.expression()

    def expression(self, min_precedence: int = 0) -> SpiceExpression:
        left = self.prefix()
        while True:
            token = self.peek()
            if token.kind == "?":
                precedence = 0
                if precedence < min_precedence:
                    break
                self.advance()
                when_true = self.expression()
                self.expect(":")
                when_false = self.expression(precedence)
                left = SpiceTernary(left, when_true, when_false)
                continue
            precedence = _BINARY_PRECEDENCE.get(token.kind)
            if precedence is None or precedence < min_precedence:
                break
            operator = token.kind
            self.advance()
            right_min = precedence if operator in _RIGHT_ASSOCIATIVE else precedence + 1
            right = self.expression(right_min)
            left = SpiceBinary(operator, left, right)
        return left

    def prefix(self) -> SpiceExpression:
        token = self.peek()
        if token.kind in _UNARY_OPERATORS:
            self.advance()
            return SpiceUnary(token.kind, self.expression(7))
        if token.kind == "number":
            self.advance()
            try:
                value = parse_spice_number(token.value, dialect=self._dialect)
            except UnitError as exc:
                self.error(str(exc), token=token)
            return SpiceNumber(token.value, value)
        if token.kind == "identifier":
            self.advance()
            if self.match("#branch"):
                return SpiceBranch(token.value)
            if self.match("("):
                return SpiceCall(token.value, self.call_args())
            return SpiceIdentifier(token.value)
        if token.kind == "@":
            return self.internal_parameter()
        if token.kind == "[":
            return self.vector()
        if token.kind in _GROUP_DELIMITERS:
            return self.group()
        self.error(f"unexpected token in SPICE expression: {token.value}", token=token)

    def call_args(self) -> tuple[SpiceExpression, ...]:
        args: list[SpiceExpression] = []
        if self.match(")"):
            return ()
        while True:
            args.append(self.expression())
            if self.match(","):
                continue
            self.expect(")")
            return tuple(args)

    def vector(self) -> SpiceVector:
        self.expect("[")
        items: list[SpiceExpression] = []
        while not self.match("]"):
            if self.peek().kind in {"<eof>", ")", "}", "'", '"'}:
                self.error("unterminated SPICE vector expression")
            items.append(self.expression())
            self.match(",")
        return SpiceVector(tuple(items))

    def group(self) -> SpiceGroup:
        opener = self.advance()
        closer, kind = _GROUP_DELIMITERS[opener.kind]
        expression = self.expression()
        self.expect(closer)
        return SpiceGroup(kind, expression)

    def internal_parameter(self) -> SpiceInternalParameter:
        self.expect("@")
        element = self.expect("identifier")
        self.expect("[")
        parameter = self.expect("identifier")
        self.expect("]")
        return SpiceInternalParameter(element.value, parameter.value)

    def peek(self) -> _Token:
        return self._tokens[self._index]

    def advance(self) -> _Token:
        token = self.peek()
        self._index += 1
        return token

    def match(self, kind: str) -> bool:
        if self.peek().kind != kind:
            return False
        self.advance()
        return True

    def expect(self, kind: str) -> _Token:
        token = self.peek()
        if token.kind != kind:
            expected = "end of expression" if kind == "<eof>" else kind
            self.error(f"expected {expected}, got {token.value}", token=token)
        return self.advance()

    def error(self, message: str, *, token: _Token | None = None) -> NoReturn:
        detail = message
        if token is not None and token.kind != "<eof>":
            detail = f"{message} at column {token.position + 1}"
        raise SpiceParseError(detail, path=self._path, line=self._line)


def _tokenize(
    text: str,
    *,
    path: str | None,
    line: int | None,
    dialect: SpiceNumberDialect,
) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        matched = next((candidate for candidate in _MULTI_CHAR_TOKENS if text.startswith(candidate, index)), None)
        if matched is not None:
            tokens.append(_Token(matched, matched, index))
            index += len(matched)
            continue
        if char in _SINGLE_CHAR_TOKENS or char in {"'", '"'}:
            tokens.append(_Token(char, char, index))
            index += 1
            continue
        if char.isdigit() or (char == "." and index + 1 < len(text) and text[index + 1].isdigit()):
            token, index = _number_token(text, index, path=path, line=line, dialect=dialect)
            tokens.append(token)
            continue
        if _is_identifier_start(char):
            start = index
            index += 1
            while index < len(text) and _is_identifier_part(text[index]):
                index += 1
            value = text[start:index]
            tokens.append(_Token("identifier", value, start))
            continue
        raise SpiceParseError(f"unexpected SPICE expression character {char!r} at column {index + 1}", path=path, line=line)
    tokens.append(_Token("<eof>", "<eof>", len(text)))
    return tuple(tokens)


def _number_token(
    text: str,
    start: int,
    *,
    path: str | None,
    line: int | None,
    dialect: SpiceNumberDialect,
) -> tuple[_Token, int]:
    index = start
    saw_digit = False
    while index < len(text) and text[index].isdigit():
        index += 1
        saw_digit = True
    if index < len(text) and text[index] == ".":
        index += 1
        while index < len(text) and text[index].isdigit():
            index += 1
            saw_digit = True
    if not saw_digit:
        raise SpiceParseError(f"invalid SPICE number at column {start + 1}", path=path, line=line)
    if index < len(text) and text[index].lower() == "e":
        exponent_index = index
        index += 1
        if index < len(text) and text[index] in {"+", "-"}:
            index += 1
        exponent_start = index
        while index < len(text) and text[index].isdigit():
            index += 1
        if index == exponent_start:
            index = exponent_index
    while index < len(text) and text[index].isalpha():
        index += 1
    if dialect == "ltspice" and index < len(text) and text[index].isdigit():
        while index < len(text) and text[index].isalnum():
            index += 1
    return _Token("number", text[start:index], start), index


def _render(expression: SpiceExpression, *, parent_precedence: int) -> str:
    precedence = _precedence(expression)
    if isinstance(expression, SpiceNumber):
        text = expression.text
    elif isinstance(expression, SpiceIdentifier):
        text = expression.name
    elif isinstance(expression, SpiceBranch):
        text = f"{expression.source}#branch"
    elif isinstance(expression, SpiceInternalParameter):
        text = f"@{expression.element}[{expression.parameter}]"
    elif isinstance(expression, SpiceUnary):
        text = f"{expression.operator}{_render(expression.operand, parent_precedence=7)}"
    elif isinstance(expression, SpiceBinary):
        left = _render(expression.left, parent_precedence=precedence)
        right_parent = precedence - 1 if expression.operator in _RIGHT_ASSOCIATIVE else precedence
        right = _render(expression.right, parent_precedence=right_parent)
        text = f"{left} {expression.operator} {right}"
    elif isinstance(expression, SpiceTernary):
        condition = _render(expression.condition, parent_precedence=1)
        when_true = _render(expression.when_true, parent_precedence=0)
        when_false = _render(expression.when_false, parent_precedence=0)
        text = f"{condition} ? {when_true} : {when_false}"
    elif isinstance(expression, SpiceCall):
        args = ", ".join(render_spice_expression(arg) for arg in expression.args)
        text = f"{expression.name}({args})"
    elif isinstance(expression, SpiceVector):
        text = "[" + " ".join(render_spice_expression(item) for item in expression.items) + "]"
    elif isinstance(expression, SpiceGroup):
        inner = render_spice_expression(expression.expression)
        if expression.kind == "brace":
            text = "{" + inner + "}"
        elif expression.kind == "quote":
            text = "'" + inner + "'"
        else:
            text = "(" + inner + ")"
    else:
        raise TypeError(f"unsupported SPICE expression node: {type(expression).__name__}")
    if precedence and precedence < parent_precedence:
        return f"({text})"
    return text


def _children(expression: SpiceExpression) -> tuple[SpiceExpression, ...]:
    if isinstance(expression, SpiceUnary):
        return (expression.operand,)
    if isinstance(expression, SpiceBinary):
        return (expression.left, expression.right)
    if isinstance(expression, SpiceTernary):
        return (expression.condition, expression.when_true, expression.when_false)
    if isinstance(expression, SpiceCall):
        return expression.args
    if isinstance(expression, SpiceVector):
        return expression.items
    if isinstance(expression, SpiceGroup):
        return (expression.expression,)
    return ()


def _precedence(expression: SpiceExpression) -> int:
    if isinstance(expression, SpiceTernary):
        return 1
    if isinstance(expression, SpiceBinary):
        return _BINARY_PRECEDENCE[expression.operator]
    if isinstance(expression, SpiceUnary):
        return 7
    return 8


def _is_identifier_start(char: str) -> bool:
    return char.isalpha() or char == "_"


def _is_identifier_part(char: str) -> bool:
    return char.isalnum() or char in {"_", ".", "$"}


__all__ = [
    "SpiceBinary",
    "SpiceBranch",
    "SpiceCall",
    "SpiceExpression",
    "SpiceGroup",
    "SpiceIdentifier",
    "SpiceInternalParameter",
    "SpiceNumber",
    "SpiceTernary",
    "SpiceUnary",
    "SpiceVector",
    "parse_spice_expression",
    "render_spice_expression",
    "walk_spice_expression",
]
