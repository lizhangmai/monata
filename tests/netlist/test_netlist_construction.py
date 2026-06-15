from collections import OrderedDict

import pytest

from monata.netlist import Circuit, Element, ModelCard, SourceValue, render_ngspice
from support.netlist_cases import Inverter
pytestmark = pytest.mark.slow

def test_instance_pins_orders_subcircuit_connections_by_declared_ports():
    circuit = Circuit("named pin instances")
    circuit.instance_pins(
        "inv0",
        Inverter,
        {"out": "y", "vin": "a", "gnd": "0", "vdd": "vdd"},
        m=2,
    )
    circuit.instance_pins(
        "gain0",
        "gain",
        {"out": "z", "in": "y"},
        pin_order=("in", "out"),
    )

    assert circuit.element("inv0", kind="X").nodes == ("a", "y", "vdd", "0")
    assert circuit.element("gain0", kind="X").nodes == ("y", "z")
    assert render_ngspice(circuit) == (
        "named pin instances\n"
        "Xinv0 a y vdd 0 inverter m=2\n"
        "Xgain0 y z gain\n"
        ".end\n"
    )


def test_long_element_aliases_are_not_public_scope_methods():
    for name in (
        "MOSFET",
        "BJT",
        "JFET",
        "MESFET",
        "VCVS",
        "VCCS",
        "CCCS",
        "CCVS",
        "VCS",
        "CCS",
        "TransmissionLine",
    ):
        assert not hasattr(Circuit, name)


def test_ac_line_helper_uses_rms_voltage_for_sinusoidal_peak():
    circuit = Circuit("ac line")
    line = circuit.ac_line("mains", "line", "0", rms_voltage=1, frequency=50, delay="1m", damping="2m")

    assert render_ngspice(circuit) == (
        "ac line\n"
        "Vmains line 0 SIN(0 1.4142135623730951 50 1m 2m)\n"
        ".end\n"
    )
    assert isinstance(line.value, SourceValue)
    assert line.value.form == "SIN"
    assert line.value.values[0] == 0
    assert line.value.values[1] == pytest.approx(2**0.5)
    assert line.value.values[2:] == (50, "1m", "2m")


def test_native_netlist_exposes_model_and_subcircuit_introspection():
    circuit = Circuit("introspection")
    diode = circuit.model_card("dmod", "D", is_=1e-14)
    circuit.model(ModelCard.create("nch", "NMOS", level=1, vto=0.4))
    subcircuit = circuit.subckt(Inverter)
    circuit.resistor("load", "out", "0", "1k")

    assert circuit.element_names == ("Rload",)
    assert circuit.model_names == ("dmod", "nch")
    assert circuit.model_cards == (
        diode,
        ModelCard.create("nch", "NMOS", level=1, vto=0.4),
    )
    assert circuit.models == circuit.model_cards
    assert circuit.get_model_card("DMOD") == diode
    assert circuit.model_card_by_name("nch").vto == 0.4
    assert circuit.get_model_card("missing") is None
    assert circuit.subcircuit_names == ("inverter",)
    assert circuit.get_subcircuit("INVERTER") is subcircuit
    assert circuit.subcircuit("inverter") is subcircuit


def test_native_netlist_subcircuit_method_can_register_definitions():
    class_based = Circuit("class-based")
    class_registered = class_based.subcircuit(Inverter)

    instance_based = Circuit("instance-based")
    instance = Inverter()
    instance_registered = instance_based.subcircuit(instance)

    assert isinstance(class_registered, Inverter)
    assert class_based.subcircuit("inverter") is class_registered
    assert instance_registered is instance
    assert instance_based.subcircuit("INVERTER") is instance


def test_element_params_may_use_ir_field_names_through_explicit_element_record():
    circuit = Circuit("element field params")
    resistor = circuit.add(
        Element(
            "R",
            "load",
            ("out", "0"),
            value="1k",
            params=OrderedDict(
                [
                    ("value", "tag"),
                    ("model", "rmod"),
                    ("name", "alias"),
                    ("params", "pmap"),
                ]
            ),
        )
    )
    mos = circuit.add(
        Element(
            "M",
            "drive",
            ("out", "in", "0", "0"),
            model="nmos",
            params=OrderedDict(
                [
                    ("model", "override"),
                    ("value", "vtag"),
                    ("name", "ntag"),
                    ("params", "mpmap"),
                ]
            ),
        )
    )

    assert resistor.value == "1k"
    assert resistor.model is None
    assert resistor.params["value"] == "tag"
    assert resistor.params["model"] == "rmod"
    assert resistor.params["name"] == "alias"
    assert resistor.params["params"] == "pmap"

    assert mos.model == "nmos"
    assert mos.value is None
    assert mos.params["model"] == "override"
    assert mos.params["value"] == "vtag"
    assert mos.params["name"] == "ntag"
    assert mos.params["params"] == "mpmap"

    assert render_ngspice(circuit) == (
        "element field params\n"
        "Rload out 0 1k value=tag model=rmod name=alias params=pmap\n"
        "Mdrive out in 0 0 nmos model=override value=vtag name=ntag params=mpmap\n"
        ".end\n"
    )
