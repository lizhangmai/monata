from collections import OrderedDict

import pytest

from monata.netlist import Circuit, Element, ModelCard, SourceValue, SubCircuit, render_ngspice
from monata.netlist.ir import NetlistError
from support.netlist_cases import Inverter
pytestmark = pytest.mark.slow

def test_native_subcircuit_rejects_duplicate_external_nodes():
    with pytest.raises(NetlistError, match="duplicate external node"):
        SubCircuit("bad", ("in", "IN"))

    subcircuit = SubCircuit("ok", ("in", "out"))
    with pytest.raises(NetlistError, match="duplicate external node"):
        subcircuit.clone(nodes=("a", "a"))


def test_element_parameter_lookup_exposes_native_params_read_only():
    element = Element(
        "R",
        "load",
        ("in", "out"),
        value="1k",
        params=OrderedDict([("temp", 27), ("lambda", 0.02), ("name", "alias")]),
    )

    assert element.parameters == ("temp", "lambda", "name")
    assert element["temp"] == 27
    assert element.temp == 27
    assert element["lambda_"] == 0.02
    assert element.lambda_ == 0.02
    assert element.get("missing", "fallback") == "fallback"
    assert element.has_parameter("temp")
    assert element.has_parameter("lambda_")
    assert not element.has_parameter("missing")
    assert element.name == "load"
    assert element["name"] == "alias"
    with pytest.raises(AttributeError, match="missing"):
        _ = element.missing


def test_instance_pins_rejects_missing_unknown_or_unordered_subcircuit_pins():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="missing subcircuit pin"):
        circuit.instance_pins("missing", Inverter, {"vin": "a", "out": "y", "vdd": "vdd"})

    with pytest.raises(NetlistError, match="unknown subcircuit pin"):
        circuit.instance_pins("unknown", Inverter, {"vin": "a", "out": "y", "vdd": "vdd", "gnd": "0", "bad": "x"})

    with pytest.raises(NetlistError, match="pin_order is required"):
        circuit.instance_pins("raw", "gain", {"in": "a", "out": "y"})

    with pytest.raises(NetlistError, match="cannot contain newlines"):
        circuit.instance_pins("badpin", "gain", {"in\nbad": "a"}, pin_order=("in\nbad",))


def test_passive_semantic_helpers_reject_duplicate_raw_and_alias_params():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="parameter m"):
        circuit.resistor("load", "in", "out", "1k", multiplier=2, m=3)

    with pytest.raises(NetlistError, match="parameter ic"):
        circuit.capacitor("hold", "out", "0", "1p", initial_condition=0, ic=1)


def test_transmission_line_semantic_helper_rejects_unrepresented_length_forms():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="requires time_delay"):
        circuit.transmission_line("missing", "in", "0", "out", "0")

    with pytest.raises(NetlistError, match="frequency requires normalized_length"):
        circuit.transmission_line("freq", "in", "0", "out", "0", frequency="1meg")

    with pytest.raises(NetlistError, match="time_delay cannot be combined"):
        circuit.transmission_line("mixed", "in", "0", "out", "0", time_delay="1n", frequency="1meg")

    with pytest.raises(NetlistError, match="initial_condition expects four values"):
        circuit.transmission_line("ic", "in", "0", "out", "0", time_delay="1n", initial_condition=(0, 1))

    with pytest.raises(NetlistError, match="raw transmission line value"):
        circuit.transmission_line("raw", "in", "0", "out", "0", "z0=50 td=1n", time_delay="2n")


def test_coupled_inductor_semantic_helper_rejects_bad_references_and_coupling():
    circuit = Circuit()
    circuit.inductor("primary", "a", "0", "1n")

    with pytest.raises(NetlistError, match="unknown inductor Lmissing"):
        circuit.coupled_inductor("bad_ref", ("primary", "missing"), 0.9, validate_inductors=True)

    with pytest.raises(NetlistError, match="between -1 and 1"):
        circuit.coupled_inductor("bad_k", ("primary", "missing"), 1.5)

    with pytest.raises(NetlistError, match="finite scalar"):
        circuit.coupled_inductor("bool_k", ("primary", "missing"), True)


def test_semiconductor_semantic_helpers_reject_ambiguous_or_malformed_params():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="parameter w"):
        circuit.mos("drive", "d", "g", "s", "b", "nmos", width="1u", w="2u")

    with pytest.raises(NetlistError, match="device initial_condition expects 3 values"):
        circuit.mos("bad_ic", "d", "g", "s", "b", "nmos", initial_condition=(0, 1))

    with pytest.raises(NetlistError, match="device initial_condition expects 2 values"):
        circuit.jfet("bad_j", "d", "g", "s", "njf", initial_condition=(0, 1, 2))


def test_controlled_switch_and_behavioral_helpers_reject_ambiguous_or_malformed_params():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="parameter m"):
        circuit.vccs("gm", "out", "0", "in", "0", "1m", multiplier=4, m=5)

    with pytest.raises(NetlistError, match="switch initial_state"):
        circuit.switch("sw", "out", "0", "ctrl", "0", "swmod", initial_state="closed")

    with pytest.raises(NetlistError, match="cannot be combined"):
        circuit.behavioral("braw", ("out", "0"), "v=V(in)", voltage_expression="V(in)")

    with pytest.raises(NetlistError, match="requires expression"):
        circuit.behavioral("bempty", ("out", "0"))


def test_raw_block_rejects_empty_blocks_and_multiline_iterable_items():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="at least one non-empty line"):
        circuit.raw_block("\n  \n")
    with pytest.raises(NetlistError, match="directive name cannot contain newlines"):
        circuit.raw_block([".control\n.endc"])


def test_native_netlist_exposes_ground_and_node_name_introspection():
    circuit = Circuit("ground helper")

    assert circuit.gnd == "0"
    assert circuit.ground_node == "0"
    assert circuit.node_names == ()
    assert circuit.has_ground_node is False
    assert circuit.get_node("out") is None
    assert circuit.has_node("out") is False

    circuit.voltage("dd", "vdd", circuit.gnd, 1)
    circuit.resistor("load", "vdd", "out", "1k")
    circuit.capacitor("hold", "OUT", circuit.ground_node, "1p")

    assert circuit.node_names == ("vdd", "0", "out")
    assert circuit.nodes == ("vdd", "0", "out")
    assert circuit.has_ground_node is True
    assert circuit.get_node("VDD") == "vdd"
    assert circuit.node("OUT") == "out"
    assert circuit.has_node("0") is True
    with pytest.raises(NetlistError, match="node not found"):
        circuit.node("missing")
    assert render_ngspice(circuit) == (
        "ground helper\n"
        "Vdd vdd 0 1\n"
        "Rload vdd out 1k\n"
        "Chold OUT 0 1p\n"
        ".end\n"
    )


def test_native_netlist_exposes_topology_node_objects_for_introspection():
    circuit = Circuit("node objects")
    circuit.resistor("load", "in", "out", "1k")
    circuit.capacitor("hold", "out", "0", "1p")
    circuit.node("loose", create=True)

    nodes = circuit.node_objects
    out = circuit.get_node_object("OUT")
    loose = circuit.node_object("loose")
    created = circuit.node_object("bias", create=True)

    assert tuple(node.name for node in nodes) == ("in", "out", "0", "loose")
    assert out is not None
    assert out.name == "out"
    assert str(out) == "out"
    assert bool(out)
    assert len(out) == 2
    assert {pin.element.name for pin in out.pins} == {"load", "hold"}
    assert {pin.element.name for pin in out} == {"load", "hold"}
    assert out.pins[0] in out
    assert circuit.node_object("0").is_ground_node is True
    assert loose.name == "loose"
    assert loose.pins == ()
    assert bool(loose) is False
    assert created.name == "bias"
    assert created.pins == ()
    circuit.resistor("sense", out, created, "2k")
    assert circuit.node_names == ("in", "out", "0", "bias", "loose")
    assert circuit.element("sense", kind="R").nodes == ("out", "bias")
    assert circuit.get_node_object("missing") is None
    with pytest.raises(NetlistError, match="node not found"):
        circuit.node_object("missing")


def test_topology_node_objects_merge_nodes_and_pin_lists():
    circuit = Circuit("node merge")
    circuit.resistor("load", "in", "out", "1k")
    circuit.capacitor("hold", "out", "0", "1p")
    topology = circuit.to_topology()

    merged = topology.node("merged")
    in_node = topology.node("in")
    hold_pin = topology.element("hold").pins[0]

    merged += [in_node, hold_pin]

    assert str(merged) == "merged"
    assert len(merged) == 3
    assert "in" not in {node.name for node in topology.nodes}
    assert "out" not in {node.name for node in topology.nodes}
    assert topology.element("load").pins[0].node is merged
    assert topology.element("load").pins[1].node is merged
    assert topology.element("hold").pins[0].node is merged


def test_subcircuit_node_names_include_external_ports_before_internal_nodes():
    subcircuit = Inverter().ensure_built()

    assert subcircuit.gnd == "0"
    assert subcircuit.node_names == ("vin", "out", "vdd", "gnd")
    assert subcircuit.has_ground_node is True
    assert subcircuit.get_node("VDD") == "vdd"
    assert subcircuit.node("gnd") == "gnd"


def test_model_card_lookup_reports_missing_or_ambiguous_names():
    circuit = Circuit()
    circuit.model("dup", "D", is_=1e-14)
    circuit.model("DUP", "D", is_=2e-14)

    assert circuit.get_model_card("dup") is None
    with pytest.raises(NetlistError, match="ambiguous model card"):
        circuit.model_card_by_name("dup")
    with pytest.raises(NetlistError, match="model card not found"):
        circuit.model_card_by_name("missing")


def test_model_card_records_reject_invalid_declarations():
    with pytest.raises(NetlistError, match="model card requires name and type"):
        ModelCard.create("", "D")

    with pytest.raises(NetlistError, match="cannot contain newlines"):
        ModelCard.create("bad\nname", "D")

    circuit = Circuit()
    model = ModelCard.create("dmod", "D")

    with pytest.raises(NetlistError, match="cannot be combined"):
        circuit.model(model, "D")

    with pytest.raises(NetlistError, match=".model requires name and type"):
        circuit.model("dmod")


def test_stable_element_lookup_preserves_identity():
    circuit = Circuit()
    element = circuit.resistor("load", "out", "0", "1k")
    model = circuit.model_card("nch", "NMOS", level=1)

    assert circuit.element_names == ("Rload",)
    assert circuit.get_element("LOAD") is element
    assert circuit.get_element("Rload") is element
    assert circuit.get_element("RLOAD", kind="R") is element
    assert circuit.element("load") is element
    assert circuit["Rload"] is element
    assert circuit.load is element
    assert circuit.Rload is element
    assert circuit["nch"] == model
    assert circuit.nch == model
    assert circuit["OUT"] == "out"
    assert circuit.out == "out"

    with pytest.raises(NetlistError, match="item not found"):
        _ = circuit["missing"]
    with pytest.raises(AttributeError, match="missing"):
        _ = circuit.missing


def test_stable_element_lookup_disambiguates_by_kind():
    circuit = Circuit()
    voltage = circuit.vdc("bias", "in", "0", 1)
    current = circuit.idc("bias", "out", "0", "1u")

    assert circuit.get_element("bias") is None
    assert circuit.get_element("Vbias") is voltage
    assert circuit.get_element("Ibias") is current
    assert circuit.get_element("bias", kind="V") is voltage
    assert circuit.get_element("Vbias", kind="V") is voltage
    assert circuit.element("bias", kind="I") is current
    assert circuit.Vbias is voltage
    assert circuit.Ibias is current
    with pytest.raises(AttributeError, match="bias"):
        _ = circuit.bias


def test_duplicate_element_names_are_rejected():
    circuit = Circuit()
    circuit.resistor("r0", "a", "b", "1k")

    with pytest.raises(NetlistError, match="duplicate element name"):
        circuit.resistor("r0", "b", "0", "2k")


def test_invalid_element_shapes_are_rejected():
    with pytest.raises(NetlistError, match="expects 2 nodes"):
        Element("R", "bad", ("a",), value="1k")

    with pytest.raises(NetlistError, match="requires a model"):
        Element("D", "bad", ("a", "0"))

    with pytest.raises(NetlistError, match="enabled flag must be a bool"):
        Element("R", "bad", ("a", "0"), value="1k", enabled="no")  # type: ignore[arg-type]

    with pytest.raises(NetlistError, match="expects 2 or 4 nodes"):
        Element("E", "bad", ("out", "0", "in"), value=1)

    with pytest.raises(NetlistError, match="requires a model"):
        Element("S", "bad", ("out", "0", "ctrl", "0"))

    with pytest.raises(NetlistError, match="expects at least 2 nodes"):
        Element("K", "bad", ("L1",), value=0.9)

    with pytest.raises(NetlistError, match="expects 4 nodes"):
        Element("T", "bad", ("a", "0", "b"), value="z0=50")

    with pytest.raises(NetlistError, match="expects at least 2 nodes"):
        Element("B", "bad", ("out",), value="v=1")

    with pytest.raises(NetlistError, match="PULSE source value expects 7 values"):
        SourceValue("PULSE", (0, 1))

    with pytest.raises(NetlistError, match="EXP source value expects 6 values"):
        SourceValue("EXP", (0, 1))

    with pytest.raises(NetlistError, match="PWL source value requires at least one"):
        SourceValue("PWL", ())

    with pytest.raises(NetlistError, match="PWL source value expects time/value pairs"):
        SourceValue("PWL", (0, 1, "2n"))

    with pytest.raises(NetlistError, match="PWL source value expects a flat"):
        SourceValue("PWL", ((0, 1), ("1n", 0)))

    with pytest.raises(NetlistError, match="SFFM source value expects 5 values"):
        SourceValue("SFFM", (0, 1, "1k"))

    with pytest.raises(NetlistError, match="AM source value expects 5 values"):
        SourceValue("AM", ("0.5", 1, "20k", "5meg", "1m", 0))

    with pytest.raises(NetlistError, match="TRRANDOM source value expects distribution code"):
        SourceValue("TRRANDOM", (99, "10m", 0, 1, 0))

    with pytest.raises(NetlistError, match="TRRANDOM source value expects distribution code"):
        SourceValue("TRRANDOM", (True, "10m", 0, 1, 0))

    with pytest.raises(NetlistError, match="TRRANDOM source value expects distribution code"):
        SourceValue("TRRANDOM", (False, "10m", 0, 1, 0))

    with pytest.raises(NetlistError, match="TRRANDOM source value expects 5 values"):
        SourceValue("TRRANDOM", (1, "10m"))

    with pytest.raises(NetlistError, match="unsupported local parameter"):
        SourceValue("PWL", (0, 0), OrderedDict([("foo", "1n")]))

    with pytest.raises(NetlistError, match="does not support local parameters"):
        SourceValue("SIN", (0, 1, "1k", 0, 0), OrderedDict([("td", "1n")]))


def test_missing_mosfet_model_is_rejected():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="requires a model"):
        circuit.mos("m0", "d", "g", "s", "b", model="")


def test_empty_subcircuit_nodes_are_rejected():
    class Empty(SubCircuit):
        NAME = "empty"

    with pytest.raises(NetlistError, match="requires external nodes"):
        Empty()


def test_nodeset_and_ic_require_values():
    circuit = Circuit()

    with pytest.raises(NetlistError, match=".nodeset requires at least one parameter"):
        circuit.nodeset()

    with pytest.raises(NetlistError, match=".ic requires at least one parameter"):
        circuit.ic()


def test_structured_netlist_values_reject_newlines():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="element name cannot contain newlines"):
        circuit.resistor("bad\n.control", "a", "0", "1k")

    with pytest.raises(NetlistError, match="Rok node cannot contain newlines"):
        circuit.resistor("ok", "a\n0", "0", "1k")

    with pytest.raises(NetlistError, match="Rok comment cannot contain newlines"):
        circuit.add(Element("R", "ok", ("a", "0"), value="1k", comment="load\nbranch"))

    with pytest.raises(NetlistError, match="Rok raw suffix cannot contain newlines"):
        circuit.add(Element("R", "ok", ("a", "0"), value="1k", raw_suffix="foo\nbar"))

    with pytest.raises(NetlistError, match=".save argument cannot contain newlines"):
        circuit.save("v(out)\nquit")

    with pytest.raises(NetlistError, match="parameter reltol value cannot contain newlines"):
        circuit.options(reltol="1e-4\nquit")

    with pytest.raises(NetlistError, match="parameter name cannot contain newlines"):
        circuit.options("acct\nquit")
