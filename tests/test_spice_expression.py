import pytest

from monata.parser import (
    SpiceBinary,
    SpiceBranch,
    SpiceCall,
    SpiceGroup,
    SpiceIdentifier,
    SpiceInternalParameter,
    SpiceNumber,
    SpiceParseError,
    SpiceTernary,
    SpiceUnary,
    SpiceVector,
    parse_spice_expression,
    render_spice_expression,
    walk_spice_expression,
)


def test_parse_spice_expression_preserves_precedence_and_numbers():
    expression = parse_spice_expression("1kHz + 2 * gain")

    assert isinstance(expression, SpiceBinary)
    assert expression.operator == "+"
    assert isinstance(expression.left, SpiceNumber)
    assert expression.left.value == pytest.approx(1000.0)
    assert isinstance(expression.right, SpiceBinary)
    assert expression.right.operator == "*"
    assert render_spice_expression(expression) == "1kHz + 2 * gain"


def test_parse_spice_expression_keeps_ltspice_rkm_as_explicit_dialect():
    with pytest.raises(SpiceParseError, match="expected end of expression"):
        parse_spice_expression("2k3 + 4R7")

    expression = parse_spice_expression("2k3 + 4R7", dialect="ltspice")

    assert isinstance(expression, SpiceBinary)
    assert isinstance(expression.left, SpiceNumber)
    assert isinstance(expression.right, SpiceNumber)
    assert expression.left.value == pytest.approx(2300.0)
    assert expression.right.value == pytest.approx(4.7)
    assert render_spice_expression(expression) == "2k3 + 4R7"


def test_parse_spice_expression_handles_calls_references_and_groups():
    expression = parse_spice_expression("{limit(v(in), -1, 1) + @m1[id] + vdd#branch}")

    assert isinstance(expression, SpiceGroup)
    assert expression.kind == "brace"
    nodes = list(walk_spice_expression(expression))
    assert any(isinstance(node, SpiceCall) and node.name == "limit" for node in nodes)
    assert any(isinstance(node, SpiceCall) and node.name == "v" for node in nodes)
    assert any(isinstance(node, SpiceInternalParameter) and node.element == "m1" for node in nodes)
    assert any(isinstance(node, SpiceBranch) and node.source == "vdd" for node in nodes)
    assert render_spice_expression(expression) == "{limit(v(in), -1, 1) + @m1[id] + vdd#branch}"


def test_parse_spice_expression_handles_vectors_ternary_and_unary_nodes():
    expression = parse_spice_expression("[0.1 {-0.2} flag ? 1 : 0]")

    assert isinstance(expression, SpiceVector)
    assert len(expression.items) == 3
    assert isinstance(expression.items[1], SpiceGroup)
    assert isinstance(expression.items[1].expression, SpiceUnary)
    assert isinstance(expression.items[2], SpiceTernary)
    assert isinstance(expression.items[2].condition, SpiceIdentifier)
    assert render_spice_expression(expression) == "[0.1 {-0.2} flag ? 1 : 0]"


def test_parse_spice_expression_reports_source_location():
    with pytest.raises(SpiceParseError) as exc_info:
        parse_spice_expression("limit(v(out),", path="expr.cir", line=12)

    assert "expr.cir:12" in str(exc_info.value)
    assert "expected" in str(exc_info.value)
