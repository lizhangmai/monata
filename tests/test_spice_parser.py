import pytest

from monata.parser import SpiceParseError, UnsupportedConstructError
from monata.parser.deck import (
    ControlBlock,
    DotCommand,
    ElementStatement,
    ParsedStatement,
    UnsupportedStatement,
    parse_spice,
)


def test_parse_spice_handles_title_comments_and_continuations():
    deck = parse_spice(
        """
RC deck
* ignored comment
R1 in out 1k
C1 out 0 1u
+ ic=0 ; continuation comment
.tran 1n 10n
.end
""",
        path="rc.cir",
    )

    assert deck.title == "RC deck"
    capacitor = next(s for s in deck.statements if isinstance(s, ElementStatement) and s.name == "C1")
    assert capacitor.tokens == ("out", "0", "1u", "ic=0")
    assert capacitor.params == {"ic": "0"}
    assert capacitor.comment == "continuation comment"
    assert capacitor.source_lines == (5, 6)
    capacitor_line = next(line for line in deck.logical_lines if line.text.startswith("C1 "))
    assert capacitor_line.comment == "continuation comment"


def test_parse_spice_preserves_standalone_comments_as_statements():
    deck = parse_spice(
        """
* leading note
commented deck
; top-level note
R1 in out 1k
.subckt rc in out
$ subcircuit note
C1 in out 1p
.ends rc
.end
"""
    )

    comments = [s for s in deck.statements if isinstance(s, ParsedStatement) and s.kind == "comment"]

    assert deck.title == "commented deck"
    assert [comment.text for comment in comments] == [
        "* leading note",
        "* top-level note",
        "* subcircuit note",
    ]
    assert comments[0].line == 2
    assert any(line.is_comment and line.text == "* subcircuit note" for line in deck.logical_lines)


def test_parse_spice_groups_control_block_and_dot_commands():
    deck = parse_spice(
        """
.title controlled
.param gain={sqrt(4)}
.control
  set noaskquit
  run
.endc
.measure tran tphl trig v(in) val=0.5 rise=1 targ v(out) val=0.5 fall=1
.meas ac gain find v(out) at=1k
.step param rload list 1k 2k
.probe v(out) i(vdd)
.end
"""
    )

    assert deck.title == "controlled"
    param = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "param")
    assert param.params == {"gain": "{sqrt(4)}"}
    control = next(s for s in deck.statements if isinstance(s, ControlBlock))
    assert [line.text for line in control.lines] == [".control", "set noaskquit", "run", ".endc"]
    measure = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "measure")
    assert measure.args[:3] == ("tran", "tphl", "trig")
    meas = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "meas")
    assert meas.args[:3] == ("ac", "gain", "find")
    step = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "step")
    assert step.args == ("param", "rload", "list", "1k", "2k")
    probe = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "probe")
    assert probe.args == ("v(out)", "i(vdd)")


def test_parse_spice_accepts_library_section_end_marker():
    deck = parse_spice(
        """
library section
.lib tt
.model n nmos level=1
.endl tt
.end
"""
    )

    endl = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "endl")
    assert endl.args == ("tt",)


def test_parse_spice_matches_control_block_delimiters_by_dot_command_name():
    with pytest.raises(UnsupportedConstructError) as unsupported:
        parse_spice("bad\n.controlled\n.endc\n.end", path="bad.cir")
    assert "bad.cir:2" in str(unsupported.value)
    assert "unsupported dot command: .controlled" in str(unsupported.value)

    with pytest.raises(SpiceParseError) as missing_endc:
        parse_spice("bad\n.control\n.endcap\n.end\n", path="bad.cir")
    assert "bad.cir:2" in str(missing_endc.value)
    assert ".control block missing .endc" in str(missing_endc.value)


def test_parse_spice_matches_title_directive_by_dot_command_name():
    assert parse_spice(".title named deck\nR1 in out 1k\n.end").title == "named deck"
    assert parse_spice(".titlecase deck\nR1 in out 1k\n.end").title == ".titlecase deck"


def test_parse_spice_accepts_later_title_directive_as_effective_title():
    deck = parse_spice(
        """
placeholder title
.title final imported deck
R1 in out 1k
.end
"""
    )

    title = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "title")
    assert deck.title == "final imported deck"
    assert title.args == ("final", "imported", "deck")


def test_parse_spice_preserves_expression_arrays_branch_refs_and_nodeset():
    deck = parse_spice(
        """
expr deck
B1 out 0 v={limit(v(in), -1, 1)}
A1 [in out] [ctrl] model in_offset=[0.1 {-0.2}]
.save v(out) v1#branch @m1[id]
.nodeset v(out)=0 v(in)=1
.end
"""
    )

    bsource = next(s for s in deck.statements if isinstance(s, ElementStatement) and s.name == "B1")
    assert "v={limit(v(in), -1, 1)}" in bsource.tokens
    xspice = next(s for s in deck.statements if isinstance(s, ElementStatement) and s.name == "A1")
    assert "[in out]" in xspice.tokens
    assert xspice.params["in_offset"] == "[0.1 {-0.2}]"
    save = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "save")
    assert save.args == ("v(out)", "v1#branch", "@m1[id]")
    nodeset = next(s for s in deck.statements if isinstance(s, DotCommand) and s.name == "nodeset")
    assert nodeset.params == {"v(out)": "0", "v(in)": "1"}


def test_parse_spice_reports_structured_errors_with_location():
    with pytest.raises(SpiceParseError) as exc_info:
        parse_spice("+ orphan", path="bad.cir")
    assert "bad.cir:1" in str(exc_info.value)

    with pytest.raises(UnsupportedConstructError) as unsupported:
        parse_spice("bad\n.foo bar\n.end", path="bad.cir")
    assert "unsupported dot command: .foo" in str(unsupported.value)


def test_parse_spice_tolerant_mode_retains_unsupported_statements():
    deck = parse_spice("bad\n.foo bar\nR1 in out 1k\n.end", path="bad.cir", strict=False)

    unsupported = next(s for s in deck.statements if isinstance(s, UnsupportedStatement))
    resistor = next(s for s in deck.statements if isinstance(s, ElementStatement) and s.name == "R1")

    assert unsupported.text == ".foo bar"
    assert unsupported.message == "unsupported dot command: .foo"
    assert unsupported.line == 2
    assert resistor.tokens == ("in", "out", "1k")
