from collections import OrderedDict

import pytest

from monata.netlist import Circuit, Directive, Element, SourceValue, SubCircuit, render_ngspice
from monata.netlist.ir import NetlistError
from support.netlist_cases import Inverter
pytestmark = pytest.mark.slow

def test_native_subcircuit_clone_is_independent_and_renameable():
    base = Inverter().ensure_built()
    clone = base.clone(name="inverter_fast", nodes=("vin", "out", "vdd", "vss"))
    clone.param("scale", 2)
    clone.include("/models/fast.mod")
    clone.resistor("bleed", "out", "vss", "100k")

    assert base.name == "inverter"
    assert base.nodes == ("vin", "out", "vdd", "gnd")
    assert base.external_nodes == base.nodes
    assert base.params["scale"] == 1
    assert clone.name == "inverter_fast"
    assert clone.nodes == ("vin", "out", "vdd", "vss")
    assert clone.external_nodes == clone.nodes
    assert clone.params["scale"] == 2
    assert base.get_element("bleed", kind="R") is None
    assert clone.get_element("bleed", kind="R") is not None
    assert "/models/fast.mod" not in base.includes
    assert "/models/fast.mod" in clone.includes


def test_native_circuit_clone_is_an_independent_template_copy():
    base = Circuit("base")
    base.include("/models/base.mod")
    subckt = base.subckt(Inverter)
    base.instance("x0", ("in", "out", "vdd", "0"), subckt)

    clone = base.clone(title="variant")
    clone.include("/models/fast.mod")
    clone.param("corner", "ff")
    clone.subcircuits[0].param("scale", 3)
    clone.voltage("dd", "vdd", "0", "1.2")

    assert base.title == "base"
    assert clone.title == "variant"
    assert base.includes == ["/models/base.mod"]
    assert clone.includes == ["/models/base.mod", "/models/fast.mod"]
    assert base.subcircuits[0] is not clone.subcircuits[0]
    assert base.subcircuits[0].params["scale"] == 1
    assert clone.subcircuits[0].params["scale"] == 3
    assert "corner" not in base.params
    assert clone.params["corner"] == "ff"
    assert base.get_element("dd", kind="V") is None
    assert clone.get_element("dd", kind="V") is not None


def test_native_circuit_copy_to_populates_existing_target_independently():
    base = Circuit("base")
    base.include("/models/base.mod")
    base.param("corner", "tt")
    base.raw_directive("* copied raw")
    base.subckt(Inverter)
    base.resistor("load", "out", "0", "1k")
    base.outputs.append(".plot tran v(out)")

    target = Circuit("target")
    target.voltage("old", "vdd", "0", 1)

    assert base.copy_to(target) is target
    assert target.title == "target"
    assert target.includes == ["/models/base.mod"]
    assert target.params == OrderedDict([("corner", "tt")])
    assert target.raw_spice == "* copied raw"
    assert target.element_names == ("Rload",)
    assert target.get_element("old", kind="V") is None
    assert target.subcircuit_names == ("inverter",)
    assert target.subcircuits[0] is not base.subcircuits[0]
    assert target.outputs == [".plot tran v(out)"]

    target.replace_element("load", target.element("load").clone(value="2k"))
    target.subcircuits[0].param("scale", 9)
    target.outputs.append(".print tran v(out)")

    assert base.element("load").value == "1k"
    assert base.subcircuits[0].params["scale"] == 1
    assert base.outputs == [".plot tran v(out)"]


def test_directive_clone_preserves_original_and_merges_params():
    directive = Directive("options", params=OrderedDict([("reltol", "1e-4"), ("acct", True)]))

    clone = directive.clone(reltol="1e-5", savecurrents=True)

    assert directive.params == OrderedDict([("reltol", "1e-4"), ("acct", True)])
    assert clone.params == OrderedDict([
        ("reltol", "1e-5"),
        ("acct", True),
        ("savecurrents", True),
    ])
    assert clone.to_spice() == ".options reltol=1e-5 acct savecurrents"


def test_directive_clone_can_replace_args_and_preserve_raw_lines():
    measure = Directive("measure", ("tran", "delay", "param='td'"))
    renamed = measure.clone(args=("ac", "gain", "param='20*log10(v(out))'"))
    raw = Directive(".control", raw=True)

    assert measure.to_spice() == ".measure tran delay param='td'"
    assert renamed.to_spice() == ".measure ac gain param='20*log10(v(out))'"
    assert raw.clone(name=".endc").to_spice() == ".endc"
    with pytest.raises(NetlistError, match=".measure requires analysis"):
        measure.clone(args=("tran",))


def test_element_clone_preserves_original_and_merges_params_for_replacement():
    circuit = Circuit("clone replace")
    base = circuit.resistor("load", "out", "0", "1k", temperature=27)
    target = Circuit("copy target")

    clone = base.clone(
        name="feedback",
        nodes=("out", "in"),
        value="2k",
        temp=30,
        tc1="1m",
        comment="fast branch",
    )

    assert base.name == "load"
    assert base.nodes == ("out", "0")
    assert base.value == "1k"
    assert base.params == OrderedDict([("temp", 27)])
    assert clone.name == "feedback"
    assert clone.nodes == ("out", "in")
    assert clone.value == "2k"
    assert clone.params == OrderedDict([("temp", 30), ("tc1", "1m")])
    assert clone.to_spice() == "* fast branch\nRfeedback out in 2k temp=30 tc1=1m"
    copied = base.copy_to(target)
    copied.params["temp"] = 40
    assert copied is target.element("load", kind="R")
    assert copied is not base
    assert copied.nodes == ("out", "0")
    assert copied.params == OrderedDict([("temp", 40)])
    assert base.params == OrderedDict([("temp", 27)])
    assert render_ngspice(target) == "copy target\nRload out 0 1k temp=40\n.end\n"
    with pytest.raises(NetlistError, match="copy target"):
        base.copy_to(object())

    assert circuit.replace_element(base, clone) is clone
    assert render_ngspice(circuit) == (
        "clone replace\n"
        "* fast branch\n"
        "Rfeedback out in 2k temp=30 tc1=1m\n"
        ".end\n"
    )


def test_element_clone_can_clear_model_and_comment():
    element = Element(
        "R",
        "load",
        ("out", "0"),
        value="1k",
        model="rmod",
        params=OrderedDict([("temp", 27)]),
        comment="old branch",
    )

    clone = element.clone(model=None, comment=None, temp=30)

    assert clone.model is None
    assert clone.comment is None
    assert clone.params == OrderedDict([("temp", 30)])
    assert clone.to_spice() == "Rload out 0 1k temp=30"


def test_disabled_elements_stay_addressable_but_do_not_render():
    circuit = Circuit("enabled elements")
    active = circuit.resistor("load", "out", "0", "1k")
    spare = circuit.resistor("spare", "out", "0", "2k", temperature=50)

    disabled = spare.clone(enabled=False, comment="held out")

    assert circuit.replace_element(spare, disabled) is disabled
    assert disabled.to_spice() == ""
    assert circuit.element("spare", kind="R") is disabled
    assert circuit.elements == [active, disabled]
    assert circuit.element_names == ("Rload", "Rspare")
    assert render_ngspice(circuit) == (
        "enabled elements\n"
        "Rload out 0 1k\n"
        ".end\n"
    )


def test_disabled_subcircuit_elements_do_not_render():
    class OptionalBranch(SubCircuit):
        NAME = "optional_branch"
        NODES = ("in", "out")

        def build(self):
            self.resistor("load", "in", "out", "1k")
            self.add(Element("C", "spare", ("out", "0"), value="1p", enabled=False))

    subcircuit = OptionalBranch()

    assert subcircuit.to_spice() == (
        ".subckt optional_branch in out\n"
        "Rload in out 1k\n"
        ".ends optional_branch\n"
    )


def test_source_value_parameter_lookup_and_clone_preserve_source_metadata():
    source = SourceValue("PWL", (0, 0, "1n", 1), OrderedDict([("dc", 0), ("td", "2n")]))

    clone = source.clone(values=(0, 0, "2n", 1.2), td="3n", r="10n")

    assert source.parameters == ("dc", "td")
    assert source["td"] == "2n"
    assert source.td == "2n"
    assert source.get("missing", "fallback") == "fallback"
    assert source.has_parameter("td")
    assert not source.has_parameter("r")
    assert clone.values == (0, 0, "2n", 1.2)
    assert clone.params == OrderedDict([("dc", 0), ("td", "3n"), ("r", "10n")])
    assert clone.has_parameter("r")
    assert clone.to_spice() == "DC 0 PWL(0 0 2n 1.2 td=3n r=10n)"
    with pytest.raises(AttributeError, match="missing"):
        _ = source.missing
    with pytest.raises(NetlistError, match="PWL source value expects time/value pairs"):
        source.clone(values=(0, 0, "1n"))


def test_raw_spice_property_replaces_raw_directives_and_preserves_ordinary_directives():
    circuit = Circuit("raw spice property")
    circuit.options("acct")
    circuit.raw_directive(".old")

    circuit.raw_spice = """
    .control
    run
    .endc
    """

    assert circuit.raw_spice == ".control\nrun\n.endc"
    assert render_ngspice(circuit) == (
        "raw spice property\n"
        ".control\n"
        "run\n"
        ".endc\n"
        ".options acct\n"
        ".end\n"
    )

    circuit.raw_spice += "\n.save v(out)"

    assert circuit.raw_spice == ".control\nrun\n.endc\n.save v(out)"
    assert render_ngspice(circuit) == (
        "raw spice property\n"
        ".control\n"
        "run\n"
        ".endc\n"
        ".save v(out)\n"
        ".options acct\n"
        ".end\n"
    )

    circuit.raw_spice = ""

    assert circuit.raw_spice == ""
    assert render_ngspice(circuit) == (
        "raw spice property\n"
        ".options acct\n"
        ".end\n"
    )


def test_remove_element_updates_lookup_rendering_and_allows_name_reuse():
    circuit = Circuit("remove elements")
    resistor = circuit.resistor("load", "out", "0", "1k")
    capacitor = circuit.capacitor("hold", "out", "0", "1p")

    assert circuit.remove_element("Rload") is resistor
    assert circuit.get_element("load", kind="R") is None
    assert circuit.element_names == ("Chold",)
    assert circuit.node_names == ("out", "0")

    replacement = circuit.resistor("load", "out", "0", "2k")

    assert circuit.element("load", kind="R") is replacement
    assert render_ngspice(circuit) == (
        "remove elements\n"
        "Chold out 0 1p\n"
        "Rload out 0 2k\n"
        ".end\n"
    )

    assert circuit.remove_element(capacitor) is capacitor
    assert circuit.element_names == ("Rload",)


def test_remove_element_respects_kind_disambiguation():
    circuit = Circuit()
    voltage = circuit.vdc("bias", "in", "0", 1)
    current = circuit.idc("bias", "out", "0", "1u")

    with pytest.raises(NetlistError, match="element not found"):
        circuit.remove_element("bias")

    assert circuit.remove_element("bias", kind="I") is current
    assert circuit.element("bias") is voltage


def test_replace_element_preserves_order_and_rebuilds_lookup():
    circuit = Circuit("replace elements")
    circuit.capacitor("hold", "in", "0", "1p")
    old = circuit.resistor("load", "out", "0", "1k")
    circuit.inductor("wire", "out", "0", "1n")
    replacement = Element("R", "feedback", ("out", "in"), value="2k")

    assert circuit.replace_element("Rload", replacement) is replacement
    assert circuit.get_element("load", kind="R") is None
    assert circuit.element("feedback", kind="R") is replacement
    assert circuit.element_names == ("Chold", "Rfeedback", "Lwire")
    assert render_ngspice(circuit) == (
        "replace elements\n"
        "Chold in 0 1p\n"
        "Rfeedback out in 2k\n"
        "Lwire out 0 1n\n"
        ".end\n"
    )

    second = Element("R", "load", ("out", "0"), value="3k")
    assert circuit.replace_element(replacement, second) is second
    assert circuit.element("load", kind="R") is second
    assert old not in circuit.elements


def test_replace_element_rolls_back_on_duplicate_replacement_name():
    circuit = Circuit("replace rollback")
    target = circuit.resistor("target", "a", "0", "1k")
    keep = circuit.capacitor("keep", "b", "0", "1p")

    with pytest.raises(NetlistError, match="duplicate element name"):
        circuit.replace_element(target, Element("C", "keep", ("a", "0"), value="2p"))

    assert circuit.elements == [target, keep]
    assert circuit.element("target", kind="R") is target
    assert circuit.element("keep", kind="C") is keep
    assert render_ngspice(circuit) == (
        "replace rollback\n"
        "Rtarget a 0 1k\n"
        "Ckeep b 0 1p\n"
        ".end\n"
    )
