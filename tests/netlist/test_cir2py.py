import pytest

from monata.netlist import render_ngspice
from monata.parser import parse_spice_to_circuit, spice_to_python


def test_spice_to_python_recreates_imported_circuit():
    deck = """
python conversion
.param rload=1k
R1 in out {rload}
C1 out 0 1u
.tran 1n 10n
.end
"""
    namespace = {}
    exec(spice_to_python(deck), namespace)

    converted = namespace["build"]()
    original = parse_spice_to_circuit(deck)
    assert render_ngspice(converted) == render_ngspice(original)


def test_spice_to_python_reads_explicit_path_objects(tmp_path):
    deck_path = tmp_path / "path_deck.cir"
    deck_path.write_text(
        """
path conversion
R1 in out 1k
.end
"""
    )
    namespace = {}
    exec(spice_to_python(deck_path), namespace)

    converted = namespace["build"]()
    assert "R1 in out 1k" in render_ngspice(converted)


def test_spice_to_python_recreates_structured_source_values():
    deck = """
source conversion
V1 in 0 PULSE(0 1 1n 1n 1n 5n 10n)
.tran 1n 10n
.end
"""
    namespace = {}
    exec(spice_to_python(deck), namespace)

    converted = namespace["build"]()
    original = parse_spice_to_circuit(deck)
    assert render_ngspice(converted) == render_ngspice(original)


def test_spice_to_python_recreates_directives_with_non_identifier_params():
    deck = """
model conversion
.model dmod d is=1e-15 name=alias
.options name=deck
D1 in out dmod area=2
.end
"""
    namespace = {}
    exec(spice_to_python(deck), namespace)

    converted = namespace["build"]()
    original = parse_spice_to_circuit(deck)
    assert render_ngspice(converted) == render_ngspice(original)


def test_spice_to_python_recreates_element_params_named_like_ir_fields():
    deck = """
element field params
R1 in out 1k model=rmod value=tag kind=meta name=alias params=pmap
M1 d g s b nmos model=override value=vtag kind=ktag name=ntag params=mpmap
.end
"""
    namespace = {}
    exec(spice_to_python(deck), namespace)

    converted = namespace["build"]()
    original = parse_spice_to_circuit(deck)
    assert render_ngspice(converted) == render_ngspice(original)


def test_spice_to_python_recreates_raw_directives_and_control_blocks():
    deck = """
raw directive conversion
R1 in out 1k
.control
run
.endc
.ac dec 10 1 1e6
.end
"""
    namespace = {}
    exec(spice_to_python(deck), namespace)

    converted = namespace["build"]()
    original = parse_spice_to_circuit(deck)
    assert render_ngspice(converted) == render_ngspice(original)


def test_spice_to_python_recreates_element_inline_comments():
    deck = """
comment conversion
R1 in out 1k ; load branch
C1 out 0 1p $ hold cap
.end
"""
    namespace = {}
    exec(spice_to_python(deck), namespace)

    converted = namespace["build"]()
    original = parse_spice_to_circuit(deck)
    assert render_ngspice(converted) == render_ngspice(original)


def test_spice_to_python_recreates_standalone_comments():
    deck = """
standalone comment conversion
* top-level note
R1 in out 1k
.subckt rc in out
; subcircuit note
C1 in out 1p
.ends rc
.end
"""
    namespace = {}
    exec(spice_to_python(deck), namespace)

    converted = namespace["build"]()
    original = parse_spice_to_circuit(deck)
    assert render_ngspice(converted) == render_ngspice(original)


def test_spice_to_python_uses_identifier_safe_subckt_variables():
    deck = """
subckt conversion
.subckt gain-cell.v1 in out
R1 in out 1k
.ends gain-cell.v1
X1 in out gain-cell.v1
.end
"""
    namespace = {}
    code = spice_to_python(deck)

    assert "subckt_gain_cell_v1" in code
    exec(code, namespace)

    converted = namespace["build"]()
    original = parse_spice_to_circuit(deck)
    assert render_ngspice(converted) == render_ngspice(original)


@pytest.mark.parametrize("function_name", ["1build", "bad-name", "bad\nname", "class"])
def test_spice_to_python_rejects_invalid_function_names(function_name):
    deck = """
python conversion
R1 in out 1k
.end
"""

    with pytest.raises(ValueError, match="function_name"):
        spice_to_python(deck, function_name=function_name)
