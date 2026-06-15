import subprocess

import pytest

from monata.netlist import render_ngspice
from monata.parser import UnsupportedConstructError, parse_spice_to_circuit


REPRESENTATIVE_DECKS = [
    pytest.param(
        """
rc deck
R1 in out 1k
C1 out 0 1u
.tran 1n 10n
.end
""",
        id="rc",
    ),
    pytest.param(
        """
model deck
.model dmod d is=1e-15
D1 in out dmod area=2
V1 in 0 DC 1
.op
.end
""",
        id="model-device",
    ),
    pytest.param(
        """
controlled deck
E1 out 0 in 0 2
B1 lim 0 v={limit(v(in), -1, 1)}
.save v(out) v1#branch @m1[id]
.nodeset v(out)=0
.end
""",
        id="controlled-expression",
    ),
    pytest.param(
        """
subckt deck
.subckt gain in out
E1 out 0 in 0 2
.ends gain
X1 in out gain
.ac dec 10 1 1e6
.end
""",
        id="subckt-analysis",
    ),
]


@pytest.mark.parametrize("deck_text", REPRESENTATIVE_DECKS)
def test_parse_render_parse_roundtrip_preserves_normalized_text(deck_text):
    first = parse_spice_to_circuit(deck_text)
    rendered = render_ngspice(first)
    second = parse_spice_to_circuit(rendered)

    assert render_ngspice(second) == rendered


def test_unsupported_construct_fails_structurally():
    with pytest.raises(UnsupportedConstructError) as exc_info:
        parse_spice_to_circuit("bad\n.foo unsupported\n.end")
    assert "unsupported dot command" in str(exc_info.value)


def test_ngspice_sanity_when_available(tmp_path, require_ngspice):
    circuit = parse_spice_to_circuit(
        """
sanity
R1 in out 1k
C1 out 0 1u
V1 in 0 DC 1
.op
.end
"""
    )
    deck_path = tmp_path / "sanity.cir"
    deck_path.write_text(render_ngspice(circuit))
    result = subprocess.run(
        ["ngspice", "-b", str(deck_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
