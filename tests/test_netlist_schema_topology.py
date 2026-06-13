from collections import OrderedDict

import pytest

from monata.netlist import (
    Circuit,
    DeviceSchemaError,
    Element,
    SubCircuit,
    Topology,
    TopologyError,
    element_spec,
    normalize_element_params,
    render_ngspice,
)
from monata.parser import parse_spice_to_circuit


def test_device_schema_exposes_pins_optional_nodes_and_parameter_aliases():
    mos = element_spec("m")
    bjt = element_spec("Q")
    diode = element_spec("D")
    jfet = element_spec("J")
    resistor = element_spec("R")
    capacitor = element_spec("C")
    inductor = element_spec("L")
    mesfet = element_spec("Z")
    mutual = element_spec("K")
    line = element_spec("T")
    behavioral = element_spec("B")
    vcvs = element_spec("E")
    vccs = element_spec("G")
    cccs = element_spec("F")
    voltage_switch = element_spec("S")
    current_switch = element_spec("W")

    assert [pin.name for pin in mos.pins] == ["drain", "gate", "source", "bulk"]
    assert bjt.min_nodes == 3
    assert bjt.max_nodes == 4
    assert diode.model_name == "model"
    assert jfet.model_name == "model"
    assert mesfet.model_name == "model"
    assert normalize_element_params(
        "M",
        {"w": "1u", "L": "45n", "m": 2, "ad": "1p", "as": "2p", "nrd": 1, "nrs": 2, "ic": "0,0,0", "nfin": 4},
    ) == OrderedDict(
        [
            ("width", "1u"),
            ("length", "45n"),
            ("multiplier", 2),
            ("area_drain", "1p"),
            ("area_source", "2p"),
            ("drain_squares", 1),
            ("source_squares", 2),
            ("initial_condition", "0,0,0"),
            ("fins", 4),
        ]
    )
    assert normalize_element_params("D", {"area": 2, "m": 3, "pj": "1u", "off": True, "ic": 0.7, "temp": 27}) == OrderedDict(
        [
            ("area", 2),
            ("multiplier", 3),
            ("junction_perimeter", "1u"),
            ("off", True),
            ("initial_condition", 0.7),
            ("temperature", 27),
        ]
    )
    assert normalize_element_params("Q", {"areac": 2, "areab": 3, "m": 4, "dtemp": 5}) == OrderedDict(
        [("area_collector", 2), ("area_base", 3), ("multiplier", 4), ("device_temperature", 5)]
    )
    assert normalize_element_params("J", {"area": 2, "m": 3, "ic": "1,0", "temp": 27}) == OrderedDict(
        [("area", 2), ("multiplier", 3), ("initial_condition", "1,0"), ("temperature", 27)]
    )
    assert normalize_element_params("Z", {"area": 2, "m": 3, "off": True, "ic": "1,0"}) == OrderedDict(
        [("area", 2), ("multiplier", 3), ("off", True), ("initial_condition", "1,0")]
    )
    assert resistor.model_name == "model"
    assert capacitor.model_name == "model"
    assert inductor.model_name == "model"
    assert normalize_element_params("R", {"l": "2u", "w": "1u", "ac": "900", "m": 2, "temp": 27, "dtemp": 5}) == OrderedDict(
        [
            ("length", "2u"),
            ("width", "1u"),
            ("ac_resistance", "900"),
            ("multiplier", 2),
            ("temperature", 27),
            ("device_temperature", 5),
        ]
    )
    assert normalize_element_params("C", {"ic": "0.2", "tc1": "1m"}) == OrderedDict(
        [("initial_condition", "0.2"), ("temperature_coefficient_1", "1m")]
    )
    assert normalize_element_params("L", {"nt": 10, "tc2": "2u"}) == OrderedDict(
        [("turns_ratio", 10), ("temperature_coefficient_2", "2u")]
    )
    assert [pin.name for pin in mutual.pins] == ["inductor1", "inductor2"]
    assert line.value_name == "line_parameters"
    assert normalize_element_params("T", {"z0": 50, "TD": "1n", "F": "1meg", "NL": "0.25", "IC": "0,0,1,0"}) == OrderedDict(
        [
            ("impedance", 50),
            ("time_delay", "1n"),
            ("frequency", "1meg"),
            ("normalized_length", "0.25"),
            ("initial_condition", "0,0,1,0"),
        ]
    )
    assert behavioral.value_name == "expression"
    assert normalize_element_params("B", {"i": "V(in)/1k", "v": "V(in)*2", "tc1": "1m", "temp": 27, "dtemp": 5}) == OrderedDict(
        [
            ("current_expression", "V(in)/1k"),
            ("voltage_expression", "V(in)*2"),
            ("temperature_coefficient_1", "1m"),
            ("temperature", 27),
            ("device_temperature", 5),
        ]
    )
    assert vcvs.min_nodes == 2
    assert vcvs.max_nodes == 4
    assert normalize_element_params("G", {"m": 4}) == OrderedDict([("multiplier", 4)])
    assert normalize_element_params("F", {"m": 3}) == OrderedDict([("multiplier", 3)])
    assert vccs.min_nodes == 2
    assert vccs.max_nodes == 4
    assert cccs.fixed_node_count == 2
    assert normalize_element_params("S", {"initial_state": "on"}) == OrderedDict([("initial_state", "on")])
    assert normalize_element_params("W", {"initial_state": "off"}) == OrderedDict([("initial_state", "off")])
    assert voltage_switch.fixed_node_count == 4
    assert current_switch.fixed_node_count == 2

    with pytest.raises(DeviceSchemaError, match="unknown element kind"):
        element_spec("?")


def test_render_missing_inspice_element_letters_p_u_n():
    circuit = Circuit("extended devices")
    circuit.coupled_multiconductor_line("bus", ("a1", "a2", "0", "b1", "b2", "0"), "cplmod", len="1m")
    circuit.distributed_rc_line("wire", "out", "in", "cap", "urcmod", l="10u", n=8)
    circuit.gss_device("num", ("d", "g", "s"), "gssmod", area=2)

    assert render_ngspice(circuit) == (
        "extended devices\n"
        "Pbus a1 a2 0 b1 b2 0 cplmod len=1m\n"
        "Uwire out in cap urcmod l=10u n=8\n"
        "Nnum d g s gssmod area=2\n"
        ".end\n"
    )


def test_spice_import_projects_p_u_n_element_letters():
    circuit = parse_spice_to_circuit(
        """
extended import
Pbus a1 a2 0 b1 b2 0 cplmod len=1m
Uwire out in cap urcmod l=10u n=8
Nnum d g s gssmod area=2
.end
"""
    )

    assert circuit.element("Pbus", kind="P").model == "cplmod"
    assert circuit.element("Pbus", kind="P").nodes == ("a1", "a2", "0", "b1", "b2", "0")
    assert circuit.element("Uwire", kind="U").nodes == ("out", "in", "cap")
    assert circuit.element("Nnum", kind="N").model == "gssmod"
    assert "Pbus a1 a2 0 b1 b2 0 cplmod len=1m" in render_ngspice(circuit)


def test_topology_reconnects_pins_and_projects_to_record_ir():
    topology = Topology("editable")
    resistor = topology.add_element("R", "load", ("in", "out"), value="1k")
    capacitor = topology.add_element("C", "hold", ("mid", "0"), value="1p")

    resistor_output = resistor.pin("n")
    resistor_output += capacitor.pin("p")
    probe = resistor.pin("p").add_current_probe("sense")
    esr = capacitor.pin("n").add_esr("ground_esr", "10m")

    assert probe.kind == "V"
    assert esr.kind == "R"
    assert not topology.dangling_pins()
    assert "mid" not in {node.name for node in topology.nodes}
    assert render_ngspice(topology.to_circuit()) == (
        "editable\n"
        "Rload load_p out 1k\n"
        "Chold out hold_n 1p\n"
        "Vsense in load_p 0\n"
        "Rground_esr 0 hold_n 10m\n"
        ".end\n"
    )


def test_topology_connects_dipoles_in_series_with_explicit_api():
    topology = Topology("series")
    first = topology.add_element("R", "r1", ("in", "join_seed"), value="1k")
    second = topology.add_element("R", "r2", ("floating", "out"), value="2k")
    third = topology.add_element("C", "c1", ("other", "0"), value="1p")

    assert first.series_with(second) is second
    assert second.series_with(third, node="join2") is third

    assert first.nodes == ("in", "join_seed")
    assert second.nodes == ("join_seed", "join2")
    assert third.nodes == ("join2", "0")
    assert render_ngspice(topology.to_circuit()) == (
        "series\n"
        "Rr1 in join_seed 1k\n"
        "Rr2 join_seed join2 2k\n"
        "Cc1 join2 0 1p\n"
        ".end\n"
    )


def test_topology_connects_dipoles_in_parallel_with_explicit_api():
    topology = Topology("parallel")
    reference = topology.add_element("R", "r1", ("in", "out"), value="1k")
    branch = topology.add_element("C", "c1", ("floating_p", "floating_n"), value="1p")

    assert reference.parallel_with(branch) is reference

    assert reference.nodes == ("in", "out")
    assert branch.nodes == ("in", "out")
    assert "floating_p" not in {node.name for node in topology.nodes}
    assert "floating_n" not in {node.name for node in topology.nodes}
    assert render_ngspice(topology.to_circuit()) == (
        "parallel\n"
        "Rr1 in out 1k\n"
        "Cc1 in out 1p\n"
        ".end\n"
    )


def test_topology_dipole_connection_api_rejects_invalid_elements():
    topology = Topology("invalid")
    resistor = topology.add_element("R", "r1", ("in", "out"), value="1k")
    fet = topology.add_element("M", "m1", ("d", "g", "s", "b"), model="nmos")

    with pytest.raises(TopologyError, match="two-pin"):
        resistor.series_with(fet)

    other_topology = Topology("foreign")
    foreign = other_topology.add_element("R", "r2", ("a", "b"), value="2k")

    with pytest.raises(TopologyError, match="same topology"):
        resistor.parallel_with(foreign)


def test_topology_can_start_from_existing_scope():
    circuit = Circuit("source")
    circuit.add(Element("R", "r1", ("in", "out"), value="1k"))

    topology = Topology.from_scope(circuit)
    topology.element("r1").pin("n").connect("load")

    assert render_ngspice(topology.to_circuit()) == "source\nRr1 in load 1k\n.end\n"


def test_circuit_can_apply_editable_topology_without_losing_scope_metadata():
    circuit = Circuit("editable scope")
    circuit.param("bias", "1u")
    circuit.options("acct")
    circuit.resistor("load", "in", "out", "1k")
    circuit.capacitor("hold", "sense", "0", "1p")

    topology = circuit.to_topology()
    topology.element("load").pin("n").connect("mid")
    topology.element("hold").pin("p").connect("mid")

    assert circuit.apply_topology(topology) is circuit
    assert circuit.node_names == ("in", "mid", "0")
    assert render_ngspice(circuit) == (
        "editable scope\n"
        ".param bias=1u\n"
        ".options acct\n"
        "\n"
        "Rload in mid 1k\n"
        "Chold mid 0 1p\n"
        ".end\n"
    )


def test_subcircuit_can_apply_editable_topology():
    subckt = SubCircuit("rc", ("in", "out"))
    subckt.param("rload", "1k")
    subckt.resistor("load", "in", "mid", "{rload}")
    subckt.capacitor("hold", "float", "out", "1p")

    topology = subckt.to_topology()
    topology.element("hold").pin("p").connect("mid")
    subckt.apply_topology(topology)

    assert render_ngspice(subckt) == (
        ".subckt rc in out rload=1k\n"
        "Rload in mid {rload}\n"
        "Chold mid out 1p\n"
        ".ends rc\n"
    )


def test_apply_topology_rejects_invalid_projection_without_mutating_scope():
    circuit = Circuit("invalid topology")
    circuit.resistor("load", "in", "out", "1k")
    topology = circuit.to_topology()
    topology.element("load").pin("n").disconnect()

    with pytest.raises(TopologyError, match="dangling pins"):
        circuit.apply_topology(topology)

    assert render_ngspice(circuit) == "invalid topology\nRload in out 1k\n.end\n"
