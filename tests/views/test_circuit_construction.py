from __future__ import annotations

from typing import Any

import pytest

from monata.circuits import (
    TransistorParams,
    add_inverter,
    add_nand2,
    add_nmos,
    add_nor2,
    add_pmos,
    add_transmission_gate,
    build_source_subcircuit_instances,
    source_subcircuit_ports,
    transistor_params,
)
from monata.netlist import SubCircuit
from monata.parser import parse_source_subcircuit


def _params(**overrides: Any) -> TransistorParams:
    values: dict[str, Any] = {
        "techlib": "LIB",
        "nmos_cell": "nfet",
        "pmos_cell": "pfet",
        "w_n": "1.2u",
        "l_n": "65n",
        "w_p": "2.4u",
        "l_p": "65n",
        "power_node": "vdd",
        "ground_node": "gnd",
    }
    values.update(overrides)
    return transistor_params(**values)


def test_netlist_does_not_export_circuit_construction_helpers():
    import monata.netlist as netlist

    assert "TransistorParams" not in netlist.__all__
    assert "add_nmos" not in netlist.__all__
    with pytest.raises(AttributeError):
        getattr(netlist, "TransistorParams")
    with pytest.raises(AttributeError):
        getattr(netlist, "add_nmos")


def test_monata_circuits_does_not_export_project_multiplier_recipe():
    import monata.circuits as circuits

    private_recipe_names = ("build_unsigned_multiplier4", "PartialProductNamer", "InternalNodeNamer")

    for name in private_recipe_names:
        assert name not in circuits.__all__
        with pytest.raises(AttributeError):
            getattr(circuits, name)


def test_source_subcircuit_parser_is_owned_by_parser_module():
    import monata.circuits as circuits
    import monata.parser as parser

    assert "parse_source_subcircuit" in parser.__all__
    assert "SourceSubcircuit" in parser.__all__
    assert "parse_source_subcircuit" not in circuits.__all__
    assert "SourceSubcircuit" not in circuits.__all__
    with pytest.raises(AttributeError):
        getattr(circuits, "parse_source_subcircuit")


def test_transistor_params_defaults_and_overrides():
    params = _params(w_n="900n", l_p="70n")

    assert params == TransistorParams(
        techlib="LIB",
        nmos_cell="nfet",
        pmos_cell="pfet",
        w_n="900n",
        l_n="65n",
        w_p="2.4u",
        l_p="70n",
        power_node="vdd",
        ground_node="gnd",
    )
    assert transistor_params(params, w_p="3u").w_p == "3u"
    with pytest.raises(TypeError, match="unknown transistor parameter"):
        transistor_params(params, bad="value")
    with pytest.raises(TypeError, match="missing"):
        transistor_params()


def test_cmos_helpers_emit_pdk_instances():
    scope = SubCircuit("device_scope", nodes=("vdd", "gnd"))
    params = _params(view="schematic")

    nmos = add_nmos(scope, "mn", "out", "in", "gnd", "gnd", params)
    pmos = add_pmos(scope, "mp", "out", "in", "vdd", "vdd", params)

    assert nmos.lib == "LIB"
    assert nmos.cell == "nfet"
    assert nmos.view == "schematic"
    assert nmos.pins == {"d": "out", "g": "in", "s": "gnd", "b": "gnd"}
    assert nmos.params == {"w": "1.2u", "l": "65n"}
    assert pmos.cell == "pfet"
    assert pmos.params == {"w": "2.4u", "l": "65n"}
    assert scope.pdk_instances == [nmos, pmos]


def test_cmos_gate_helpers_preserve_device_sequences():
    scope = SubCircuit("gate_scope", nodes=("vdd", "gnd"))
    params = _params()

    add_inverter(scope, "inv", "a", "z", params)
    add_transmission_gate(scope, "tg", "a", "z", "en", "enb", params)
    add_nand2(scope, "nand", "a", "b", "zn", params)
    add_nor2(scope, "nor", "a", "b", "zn", params)

    assert [instance.name for instance in scope.pdk_instances] == [
        "inv_p",
        "inv_n",
        "tg_n",
        "tg_p",
        "nand_p1",
        "nand_p2",
        "nand_n1",
        "nand_n2",
        "nor_p1",
        "nor_p2",
        "nor_n1",
        "nor_n2",
    ]
    assert scope.pdk_instances[6].pins == {"d": "zn", "g": "a", "s": "nand_mid", "b": "gnd"}
    assert scope.pdk_instances[9].pins == {"d": "nor_mid", "g": "b", "s": "vdd", "b": "vdd"}


def test_source_subcircuit_parser_accepts_explicit_path(tmp_path):
    source = tmp_path / "adder.scs"
    source.write_text(
        """
.subckt sample in out vdd vss
I0 (in mid vdd vss) inv
I1 (mid out vdd vss) inv
.ends sample
""".strip()
    )

    netlist = parse_source_subcircuit(source)

    assert netlist.path == source
    assert netlist.name == "sample"
    assert netlist.ports == ("in", "out", "vdd", "vss")
    assert netlist.instances[0].name == "I0"
    assert netlist.instances[0].nodes == ("in", "mid", "vdd", "vss")
    assert netlist.instances[0].kind == "inv"
    assert source_subcircuit_ports(source_root=tmp_path, filename="adder.scs") == netlist.ports


def test_source_subcircuit_builder_validates_and_emits_instances(tmp_path):
    source = tmp_path / "cell.scs"
    source.write_text(
        """
subckt chain a z vdd vss
I0 (a n1 vdd vss) inv
I1 (n1 z vdd vss) inv
ends chain
""".strip()
    )
    scope = SubCircuit("target", nodes=("a", "z", "vdd", "vss"))

    build_source_subcircuit_instances(
        scope,
        source_root=tmp_path,
        filename="cell.scs",
        expected_name="chain",
        expected_ports=("a", "z", "vdd", "vss"),
        expected_count=2,
        allowed_kinds=("inv",),
    )

    assert [(element.name, element.nodes, element.model) for element in scope.elements] == [
        ("I0", ("a", "n1", "vdd", "vss"), "inv"),
        ("I1", ("n1", "z", "vdd", "vss"), "inv"),
    ]


def test_source_subcircuit_builder_rejects_contract_mismatches(tmp_path):
    source = tmp_path / "cell.scs"
    source.write_text(
        """
.subckt chain a z vdd vss
I0 (a z vdd vss) forbidden
.ends chain
""".strip()
    )

    with pytest.raises(ValueError, match="references unsupported subcircuits"):
        build_source_subcircuit_instances(
            SubCircuit("target", nodes=("a", "z", "vdd", "vss")),
            source_file=source,
            expected_name="chain",
            expected_ports=("a", "z", "vdd", "vss"),
            expected_count=1,
            allowed_kinds=("inv",),
        )
