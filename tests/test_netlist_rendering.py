from collections import OrderedDict

import pytest

from monata.netlist import Circuit, Directive, Element, ModelCard, SourceValue, SubCircuit, render_ngspice
from monata.netlist.ir import NetlistError
from monata.units import Hz, V, kOhm, ms, quantity
from support.netlist_cases import Inverter
pytestmark = pytest.mark.slow

def test_render_native_subcircuit():
    netlist = render_ngspice(Inverter())

    assert netlist == (
        ".subckt inverter vin out vdd gnd scale=1\n"
        '.include "/models/nmos.mod"\n'
        '.include "/models/pmos.mod"\n'
        "Mmn out vin gnd gnd nmos w=1u l=45n\n"
        "Mmp out vin vdd vdd pmos w=2u l=45n\n"
        ".ends inverter\n"
    )


def test_subcircuit_element_comments_render_as_spice_comment_lines():
    class Commented(SubCircuit):
        NAME = "commented"
        NODES = ("a", "b")

        def build(self):
            self.add(Element("R", "load", ("a", "b"), value="1k", comment="subckt load"))

    assert render_ngspice(Commented()) == (
        ".subckt commented a b\n"
        "* subckt load\n"
        "Rload a b 1k\n"
        ".ends commented\n"
    )


def test_subcircuit_to_spice_uses_native_renderer():
    subcircuit = Inverter()
    expected = render_ngspice(Inverter())

    assert subcircuit.to_spice() == expected
    assert subcircuit.external_nodes == ("vin", "out", "vdd", "gnd")
    assert str(subcircuit) == expected
    assert not hasattr(subcircuit, "str")


def test_render_native_circuit_with_all_milestone_primitives():
    circuit = Circuit("primitive smoke")
    circuit.include("/models/devices.mod")
    circuit.param("temp", 27)
    circuit.subckt(Inverter)
    circuit.voltage("dd", "vdd", "0", "1.0")
    circuit.current("bias", "vin", "0", "0")
    circuit.resistor("load", "out", "0", "1k")
    circuit.capacitor("out", "out", "0", "1f")
    circuit.inductor("wire", "vdd", "out", "1n")
    circuit.instance("inv0", ("vin", "out", "vdd", "0"), Inverter)

    netlist = render_ngspice(circuit)

    assert netlist == (
        "primitive smoke\n"
        '.include "/models/devices.mod"\n'
        ".param temp=27\n"
        "\n"
        ".subckt inverter vin out vdd gnd scale=1\n"
        '.include "/models/nmos.mod"\n'
        '.include "/models/pmos.mod"\n'
        "Mmn out vin gnd gnd nmos w=1u l=45n\n"
        "Mmp out vin vdd vdd pmos w=2u l=45n\n"
        ".ends inverter\n"
        "\n"
        "Vdd vdd 0 1.0\n"
        "Ibias vin 0 0\n"
        "Rload out 0 1k\n"
        "Cout out 0 1f\n"
        "Lwire vdd out 1n\n"
        "Xinv0 vin out vdd 0 inverter\n"
        ".end\n"
    )


def test_circuit_to_spice_uses_native_renderer():
    circuit = Circuit("direct export")
    circuit.resistor("load", "out", "0", "1k")
    expected = render_ngspice(circuit)

    assert circuit.to_spice() == expected
    assert str(circuit) == expected
    assert not hasattr(circuit, "str")
    assert not hasattr(circuit, "str_end")


def test_native_netlist_records_to_spice_use_ngspice_renderer():
    resistor = Element("R", "load", ("out", "0"), value="1k")
    options = Directive("options", params=OrderedDict([("acct", True), ("reltol", "1e-4")]))
    model = ModelCard.create("nch", "NMOS", level=1, vto=0.4)

    assert resistor.to_spice() == "Rload out 0 1k"
    assert options.to_spice() == ".options acct reltol=1e-4"
    assert model.to_spice() == ".model nch NMOS (level=1 vto=0.4)"
    assert str(resistor) == "Rload out 0 1k"
    assert str(options) == ".options acct reltol=1e-4"
    assert str(model) == ".model nch NMOS (level=1 vto=0.4)"


def test_directive_parameter_lookup_exposes_native_params_read_only():
    directive = Directive(
        "options",
        params=OrderedDict([
            ("reltol", "1e-4"),
            ("lambda", 0.02),
            ("name", "deck"),
        ]),
    )
    raw = Directive(".control", raw=True)

    assert directive.parameters == ("reltol", "lambda", "name")
    assert directive["reltol"] == "1e-4"
    assert directive.reltol == "1e-4"
    assert directive["lambda_"] == 0.02
    assert directive.lambda_ == 0.02
    assert directive.get("missing", "fallback") == "fallback"
    assert directive.has_parameter("reltol")
    assert directive.has_parameter("lambda_")
    assert not directive.has_parameter("missing")
    assert directive.name == "options"
    assert directive["name"] == "deck"
    assert raw.parameters == ()
    assert not raw.has_parameter("reltol")
    with pytest.raises(AttributeError, match="missing"):
        _ = directive.missing


def test_element_raw_suffix_renders_after_structured_params_and_can_be_cleared():
    circuit = Circuit("raw suffix")
    element = circuit.add(
        Element(
            "R",
            "sense",
            ("in", "out"),
            value="1k",
            params=OrderedDict([("temp", 27)]),
            raw_suffix=" noiseless custom_token ",
        )
    )

    assert element.raw_suffix == "noiseless custom_token"
    assert element.to_spice() == "Rsense in out 1k temp=27 noiseless custom_token"
    assert render_ngspice(circuit) == (
        "raw suffix\n"
        "Rsense in out 1k temp=27 noiseless custom_token\n"
        ".end\n"
    )

    clone = element.clone(raw_suffix=None, temp=30)

    assert clone.raw_suffix is None
    assert clone.to_spice() == "Rsense in out 1k temp=30"


def test_element_comments_render_as_spice_comment_lines():
    circuit = Circuit("commented elements")
    circuit.add(Element("R", "load", ("out", "0"), value="1k", comment="load branch"))
    circuit.add(Element("C", "hold", ("out", "0"), value="1p"))

    assert circuit.element("load", kind="R").to_spice() == "* load branch\nRload out 0 1k"
    assert render_ngspice(circuit) == (
        "commented elements\n"
        "* load branch\n"
        "Rload out 0 1k\n"
        "Chold out 0 1p\n"
        ".end\n"
    )


def test_source_value_to_spice_uses_ngspice_value_renderer():
    assert SourceValue("SIN", (0, 1, "1k", None, "")).to_spice() == "SIN(0 1 1k)"
    assert SourceValue("AC", (0, 1)).to_spice() == "DC 0 AC 1"
    assert (
        SourceValue("PWL", (0, 0, "1n", 1), OrderedDict([("dc", 0), ("r", "10n"), ("td", "2n")])).to_spice()
        == "DC 0 PWL(0 0 1n 1 r=10n td=2n)"
    )


def test_render_ngspice_formats_monata_quantities_as_spice_values():
    circuit = Circuit("quantity values")
    circuit.param("vdd", quantity(1200, "mV"))
    circuit.voltage("dd", "vdd", "0", V(1.2))
    circuit.resistor("load", "vdd", "out", 1 @ kOhm)
    circuit.vsin("sig", "in", "0", 0 @ V, quantity(500, "mV"), 1 @ Hz)
    circuit.vpulse("clk", "clk", "0", 0 @ V, V(1.2), 0 @ ms, 1 @ ms, 1 @ ms, 5 @ ms, 10 @ ms)

    assert render_ngspice(circuit) == (
        "quantity values\n"
        ".param vdd=1.2\n"
        "\n"
        "Vdd vdd 0 1.2\n"
        "Rload vdd out 1k\n"
        "Vsig in 0 SIN(0 500m 1 0 0)\n"
        "Vclk clk 0 PULSE(0 1.2 0 1m 1m 5m 10m)\n"
        ".end\n"
    )


def test_render_ngspice_omits_empty_source_value_tokens():
    circuit = Circuit("optional source values")
    circuit.voltage("sig", "in", "0", SourceValue("SIN", (0 @ V, 1 @ V, 1 @ Hz, None, "")))
    circuit.voltage("pwl", "out", "0", SourceValue("PWL", (0, 0, "1n", 1, None, ""), OrderedDict([("td", None)])))

    assert render_ngspice(circuit) == (
        "optional source values\n"
        "Vsig in 0 SIN(0 1 1)\n"
        "Vpwl out 0 PWL(0 0 1n 1)\n"
        ".end\n"
    )


def test_render_broad_shallow_element_surface():
    circuit = Circuit("broad surface")
    circuit.diode("d1", "a", "0", "dmod", area=2)
    circuit.bjt("q1", "c", "b", "e", "npn", area=1)
    circuit.jfet("j1", "d", "g", "s", "njf")
    circuit.mesfet("z1", "d", "g", "s", "nmf")
    circuit.switch("sw", "out", "0", "ctrl", "0", "swmod")
    circuit.current_switch("csw", "out", "0", "vsense", "cswmod")
    circuit.vcvs("e1", "out", "0", "in", "0", 10)
    circuit.cccs("f1", "out", "0", "vsense", 2)
    circuit.vccs("g1", "out", "0", "in", "0", "1m")
    circuit.ccvs("h1", "out", "0", "vsense", 50)
    circuit.behavioral("b1", ("out", "0"), "v=V(in)*2")
    circuit.coupled_inductor("k1", ("L1", "L2"), 0.9)
    circuit.transmission_line("t1", "a", "0", "b", "0", "z0=50 td=1n")
    circuit.lossy_line("o1", "a", "0", "b", "0", "ltra")
    circuit.txl_line("y1", "a", "0", "b", "0", "txl")
    circuit.code_model("cm1", ("in", "out"), "adc")
    circuit.arbitrary("A", "arb", ("n1", "n2", "n3"), model="custom")

    assert render_ngspice(circuit) == (
        "broad surface\n"
        "Dd1 a 0 dmod area=2\n"
        "Qq1 c b e npn area=1\n"
        "Jj1 d g s njf\n"
        "Zz1 d g s nmf\n"
        "Ssw out 0 ctrl 0 swmod\n"
        "Wcsw out 0 vsense cswmod\n"
        "Ee1 out 0 in 0 10\n"
        "Ff1 out 0 vsense 2\n"
        "Gg1 out 0 in 0 1m\n"
        "Hh1 out 0 vsense 50\n"
        "Bb1 out 0 v=V(in)*2\n"
        "Kk1 L1 L2 0.9\n"
        "Tt1 a 0 b 0 z0=50 td=1n\n"
        "Oo1 a 0 b 0 ltra\n"
        "Yy1 a 0 b 0 txl\n"
        "Acm1 in out adc\n"
        "Aarb n1 n2 n3 custom\n"
        ".end\n"
    )


def test_spice_letter_aliases_render_existing_element_surface():
    circuit = Circuit("letter aliases")
    circuit.V("dd", "vdd", "0", "1.2")
    circuit.I("bias", "in", "0", "1m")
    circuit.R("load", "vdd", "out", "1k")
    circuit.C("hold", "out", "0", "1p")
    circuit.L("primary", "vdd", "out", "1n")
    circuit.L("secondary", "sense", "0", "2n")
    circuit.D("clamp", "out", "0", "dmod")
    circuit.Q("gain", "c", "b", "e", "npn")
    circuit.J("drv", "d", "g", "s", "jmod")
    circuit.M("drive", "d", "g", "s", "b", "nmos", width="1u", length="45n")
    circuit.Z("mes", "d", "g", "s", "zmod")
    circuit.E("amp", "out", "0", "in", "0", 10)
    circuit.F("mirror", "out", "0", "vsense", 2)
    circuit.G("gm", "out", "0", "in", "0", "1m")
    circuit.H("sense", "out", "0", "vsense", 50)
    circuit.B("expr", ("out", "0"), voltage_expression="V(in)")
    circuit.K("link", ("primary", "secondary"), 0.9)
    circuit.S("sw", "out", "0", "ctrl", "0", "swmod")
    circuit.W("csw", "out", "0", "vsense", "cswmod")
    circuit.T("delay", "a", "0", "b", "0", time_delay="1n")
    circuit.O("loss", "a", "0", "b", "0", "ltra")
    circuit.P("cpl", ("a", "b", "c"), "cplmod")
    circuit.U("rc", "out", "in", "cap", "urcmod")
    circuit.Y("txl", "a", "0", "b", "0", "txlmod")
    circuit.A("adc", ("in", "out"), "adcmod")
    circuit.N("gss", ("n1",), "gssmod")
    circuit.X("cell", ("in", "out"), "gain")

    assert render_ngspice(circuit) == (
        "letter aliases\n"
        "Vdd vdd 0 1.2\n"
        "Ibias in 0 1m\n"
        "Rload vdd out 1k\n"
        "Chold out 0 1p\n"
        "Lprimary vdd out 1n\n"
        "Lsecondary sense 0 2n\n"
        "Dclamp out 0 dmod\n"
        "Qgain c b e npn\n"
        "Jdrv d g s jmod\n"
        "Mdrive d g s b nmos w=1u l=45n\n"
        "Zmes d g s zmod\n"
        "Eamp out 0 in 0 10\n"
        "Fmirror out 0 vsense 2\n"
        "Ggm out 0 in 0 1m\n"
        "Hsense out 0 vsense 50\n"
        "Bexpr out 0 v=V(in)\n"
        "Klink Lprimary Lsecondary 0.9\n"
        "Ssw out 0 ctrl 0 swmod\n"
        "Wcsw out 0 vsense cswmod\n"
        "Tdelay a 0 b 0 z0=50 td=1n\n"
        "Oloss a 0 b 0 ltra\n"
        "Pcpl a b c cplmod\n"
        "Urc out in cap urcmod\n"
        "Ytxl a 0 b 0 txlmod\n"
        "Aadc in out adcmod\n"
        "Ngss n1 gssmod\n"
        "Xcell in out gain\n"
        ".end\n"
    )


def test_render_passive_device_semantic_helpers():
    circuit = Circuit("passive semantics")
    circuit.resistor(
        "load",
        "in",
        "out",
        "1k",
        model="rmod",
        ac="900",
        multiplier=2,
        scale="1.5",
        temperature=27,
        device_temperature=5,
        noisy=False,
    )
    circuit.semiconductor_resistor("sheet", "out", "0", "100", "rpoly", length="2u", width="1u", multiplier=4)
    circuit.behavioral_resistor("sense", "ctrl", "0", "R={v(ctrl)+1}", tc1="1m", tc2="2u")
    circuit.capacitor("hold", "out", "0", "1p", model="cmod", multiplier=3, initial_condition="0.2")
    circuit.semiconductor_capacitor("gate", "g", "0", "1f", "cmos", length="45n", width="1u")
    circuit.behavioral_capacitor("var", "out", "0", "C={v(out)*1p}", tc1="10u")
    circuit.inductor("coil", "vdd", "out", "1n", model="lmod", turns_ratio=10, initial_condition="1m")
    circuit.behavioral_inductor("sat", "out", "0", "L={1n+v(out)*1p}", tc2="5u")

    assert render_ngspice(circuit) == (
        "passive semantics\n"
        "Rload in out 1k rmod ac=900 m=2 scale=1.5 temp=27 dtemp=5 noisy=False\n"
        "Rsheet out 0 100 rpoly l=2u w=1u m=4\n"
        "Rsense ctrl 0 R={v(ctrl)+1} tc1=1m tc2=2u\n"
        "Chold out 0 1p cmod m=3 ic=0.2\n"
        "Cgate g 0 1f cmos l=45n w=1u\n"
        "Cvar out 0 C={v(out)*1p} tc1=10u\n"
        "Lcoil vdd out 1n lmod nt=10 ic=1m\n"
        "Lsat out 0 L={1n+v(out)*1p} tc2=5u\n"
        ".end\n"
    )


def test_render_transmission_line_and_coupled_inductor_semantics():
    circuit = Circuit("line semantics")
    circuit.inductor("1", "a", "0", "1n")
    circuit.inductor("secondary", "b", "0", "2n")
    circuit.coupled_inductor("link", ("1", "secondary"), 0.9, validate_inductors=True)
    circuit.transmission_line("delay", "in", "0", "out", "0", time_delay="1n")
    circuit.transmission_line(
        "freq",
        "a",
        "0",
        "b",
        "0",
        impedance=75,
        frequency="1meg",
        normalized_length="0.25",
        initial_condition=(0, 0, 1, 0),
    )

    assert render_ngspice(circuit) == (
        "line semantics\n"
        "L1 a 0 1n\n"
        "Lsecondary b 0 2n\n"
        "Klink L1 Lsecondary 0.9\n"
        "Tdelay in 0 out 0 z0=50 td=1n\n"
        "Tfreq a 0 b 0 z0=75 f=1meg nl=0.25 ic=0,0,1,0\n"
        ".end\n"
    )


def test_render_semiconductor_device_semantic_helpers():
    circuit = Circuit("semiconductor semantics")
    circuit.mos(
        "drive",
        "d",
        "g",
        "s",
        "b",
        "nmos",
        width="1u",
        length="45n",
        multiplier=2,
        area_drain="1p",
        area_source="2p",
        perimeter_drain="3u",
        perimeter_source="4u",
        drain_squares=1,
        source_squares=2,
        off=True,
        initial_condition=(0.1, 0.2, 0.3),
        temperature=27,
        fins=4,
    )
    circuit.diode(
        "clamp",
        "out",
        "0",
        "dmod",
        area=2,
        multiplier=3,
        junction_perimeter="1u",
        off=True,
        initial_condition=0.7,
        temperature=27,
        device_temperature=5,
    )
    circuit.bjt(
        "gain",
        "c",
        "b",
        "e",
        "npn",
        substrate="sub",
        area=1,
        area_collector=2,
        area_base=3,
        multiplier=4,
        off=True,
        initial_condition=(0.7, 1.2),
        temperature=50,
        device_temperature=3,
    )
    circuit.jfet("jdrv", "d", "g", "s", "njf", area=2, multiplier=3, off=True, initial_condition=(1, 0), temperature=27)
    circuit.mesfet("zdrv", "d", "g", "s", "nmf", area=2, multiplier=3, off=True, initial_condition=(1, 0))

    assert render_ngspice(circuit) == (
        "semiconductor semantics\n"
        "Mdrive d g s b nmos w=1u l=45n m=2 ad=1p as=2p pd=3u ps=4u "
        "nrd=1 nrs=2 off ic=0.1,0.2,0.3 temp=27 nfin=4\n"
        "Dclamp out 0 dmod area=2 m=3 pj=1u off ic=0.7 temp=27 dtemp=5\n"
        "Qgain c b e sub npn area=1 areac=2 areab=3 m=4 off ic=0.7,1.2 temp=50 dtemp=3\n"
        "Jjdrv d g s njf area=2 m=3 off ic=1,0 temp=27\n"
        "Zzdrv d g s nmf area=2 m=3 off ic=1,0\n"
        ".end\n"
    )


def test_render_controlled_switch_and_behavioral_source_semantics():
    circuit = Circuit("controlled semantics")
    circuit.vccs("gm", "out", "0", "in", "0", "1m", multiplier=4)
    circuit.cccs("mirror", "out", "0", "vsense", 2, multiplier=3)
    circuit.switch("sw", "out", "0", "ctrl", "0", "swmod", initial_state=True)
    circuit.current_switch("csw", "out", "0", "vsense", "cswmod", initial_state="off")
    circuit.behavioral_voltage("bv", "out", "0", "V(in)*2", tc1="1m", temperature=27)
    circuit.behavioral_current("bi", "out", "0", "V(in)/1k", tc2="2m", device_temperature=5)
    circuit.behavioral("bboth", ("out", "0"), current_expression="V(in)", voltage_expression="V(ctrl)")

    assert render_ngspice(circuit) == (
        "controlled semantics\n"
        "Ggm out 0 in 0 1m m=4\n"
        "Fmirror out 0 vsense 2 m=3\n"
        "Ssw out 0 ctrl 0 swmod on\n"
        "Wcsw out 0 vsense cswmod off\n"
        "Bbv out 0 v=V(in)*2 tc1=1m temp=27\n"
        "Bbi out 0 i=V(in)/1k tc2=2m dtemp=5\n"
        "Bbboth out 0 i=V(in) v=V(ctrl)\n"
        ".end\n"
    )


def test_render_nonlinear_controlled_source_semantics():
    circuit = Circuit("nonlinear controlled sources")
    circuit.nonlinear_voltage_source("ev", "out", "0", "V(in)*2")
    circuit.nonlinear_current_source("gi", "out", "0", "{V(in)/1k}")
    circuit.table_voltage_source("etab", "lim", "0", "V(in)", [(0, 0), (1, 1)])
    circuit.table_current_source("gtab", "ilim", "0", "V(in)", [(-1, "-1m"), (1, "1m")])
    circuit.laplace_voltage_source("elap", "filt", "0", "V(in)", "10 / (s/6800 + 1)")
    circuit.laplace_current_source("glap", "ifilt", "0", "{V(in)}", "{1 / (s + 1)}")
    circuit.poly_voltage_source("epoly", "poly", "0", [("a", "0"), ("b", "0")], [0, "13.6", "0.2"])
    circuit.poly_current_source("gpoly", "ipoly", "0", [("a", "0")], ["0", "1m"])

    assert render_ngspice(circuit) == (
        "nonlinear controlled sources\n"
        "Eev out 0 value={V(in)*2}\n"
        "Ggi out 0 value={V(in)/1k}\n"
        "Eetab lim 0 TABLE {V(in)} = (0,0) (1,1)\n"
        "Ggtab ilim 0 TABLE {V(in)} = (-1,-1m) (1,1m)\n"
        "Eelap filt 0 LAPLACE {V(in)} {10 / (s/6800 + 1)}\n"
        "Gglap ifilt 0 LAPLACE {V(in)} {1 / (s + 1)}\n"
        "Eepoly poly 0 POLY(2) a 0 b 0 0 13.6 0.2\n"
        "Ggpoly ipoly 0 POLY(1) a 0 0 1m\n"
        ".end\n"
    )


def test_nonlinear_controlled_source_helpers_reject_malformed_table_points():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="at least one point"):
        circuit.table_voltage_source("empty", "out", "0", "V(in)", [])

    with pytest.raises(NetlistError, match="two-value pairs"):
        circuit.table_current_source("bad", "out", "0", "V(in)", [(0, 0, 0)])  # type: ignore[list-item]

    with pytest.raises(NetlistError, match="expects 2 or 4 nodes"):
        circuit.add(Element("E", "bad", ("out", "0", "extra"), value="value={V(in)}"))

    with pytest.raises(NetlistError, match="input expression"):
        circuit.laplace_voltage_source("empty_laplace", "out", "0", "", "1 / (s + 1)")

    with pytest.raises(NetlistError, match="transfer function"):
        circuit.laplace_current_source("empty_xfer", "out", "0", "V(in)", "")

    with pytest.raises(NetlistError, match="control node pair"):
        circuit.poly_voltage_source("empty_poly", "out", "0", [], [1])

    with pytest.raises(NetlistError, match="two-value node pairs"):
        circuit.poly_current_source("bad_poly", "out", "0", ["a"], [1])  # type: ignore[list-item]

    with pytest.raises(NetlistError, match="at least one coefficient"):
        circuit.poly_voltage_source("no_coeffs", "out", "0", [("a", "0")], [])


def test_render_source_helpers():
    circuit = Circuit("source helpers")
    circuit.vdc("dc", "in", "0", "1.2")
    circuit.idc("dc", "load", "0", "10u")
    circuit.vac("ac", "in", "0", "0", "1")
    circuit.vpulse("pulse", "clk", "0", 0, "1.8", "1n", "100p", "100p", "5n", "10n")
    circuit.vsin("sin", "in", "0", 0, 1, "1k", 0, 0)

    assert render_ngspice(circuit) == (
        "source helpers\n"
        "Vdc in 0 DC 1.2\n"
        "Idc load 0 DC 10u\n"
        "Vac in 0 DC 0 AC 1\n"
        "Vpulse clk 0 PULSE(0 1.8 1n 100p 100p 5n 10n)\n"
        "Vsin in 0 SIN(0 1 1k 0 0)\n"
        ".end\n"
    )


def test_render_extended_source_helpers():
    circuit = Circuit("extended source helpers")
    circuit.vexp("exp", "in", "0", 0, 1, "1n", "2n", "10n", "3n")
    circuit.iexp("iexp", "load", "0", 0, "10u", "1n", "2n", "10n", "3n")
    circuit.vpwl("pwl", "clk", "0", (0, 0), ("1n", "1.8"), ("2n", 0))
    circuit.ipwl("ipwl", "bias", "0", 0, 0, "1n", "1u")
    circuit.vsffm("sffm", "fm", "0", 0, "1m", "20k", 5, "1k")
    circuit.isffm("isffm", "ifm", "0", 0, "1m", "20k", 5, "1k")
    circuit.vam("am", "am", "0", "0.5", 1, "20k", "5meg", "1m")
    circuit.iam("iam", "iam", "0", "0.5", 1, "20k", "5meg", "1m")
    circuit.vtrrandom("rand", "noise", "0", "gaussian", "10m", 0, 1, 0)
    circuit.itrrandom("irand", "inoise", "0", "poisson", "1m", "10u", 4, 0)

    assert render_ngspice(circuit) == (
        "extended source helpers\n"
        "Vexp in 0 EXP(0 1 1n 2n 10n 3n)\n"
        "Iiexp load 0 EXP(0 10u 1n 2n 10n 3n)\n"
        "Vpwl clk 0 PWL(0 0 1n 1.8 2n 0)\n"
        "Iipwl bias 0 PWL(0 0 1n 1u)\n"
        "Vsffm fm 0 SFFM(0 1m 20k 5 1k)\n"
        "Iisffm ifm 0 SFFM(0 1m 20k 5 1k)\n"
        "Vam am 0 AM(0.5 1 20k 5meg 1m)\n"
        "Iiam iam 0 AM(0.5 1 20k 5meg 1m)\n"
        "Vrand noise 0 TRRANDOM(2 10m 0 1 0)\n"
        "Iirand inoise 0 TRRANDOM(4 1m 10u 4 0)\n"
        ".end\n"
    )


def test_render_pwl_source_options():
    circuit = Circuit("pwl source options")
    voltage = circuit.vpwl(
        "repeat",
        "out",
        "0",
        (0, 0),
        ("1n", 1),
        dc=0,
        repeat_time="10n",
        delay_time="2n",
        temp=27,
    )
    current = circuit.ipwl("delay", "bias", "0", 0, 0, "1n", "1u", delay_time="500p")

    assert render_ngspice(circuit) == (
        "pwl source options\n"
        "Vrepeat out 0 DC 0 PWL(0 0 1n 1 r=10n td=2n) temp=27\n"
        "Idelay bias 0 PWL(0 0 1n 1u td=500p)\n"
        ".end\n"
    )
    assert isinstance(voltage.value, SourceValue)
    assert voltage.value.params == OrderedDict([("dc", 0), ("r", "10n"), ("td", "2n")])
    assert isinstance(current.value, SourceValue)
    assert current.value.params == OrderedDict([("td", "500p")])


def test_extended_source_helpers_preserve_source_metadata():
    circuit = Circuit()
    pwl = circuit.vpwl("pwl", "out", "0", (0, 0), ("1n", 1))
    am = circuit.vam("am", "out", "0", "0.5", 1, "20k", "5meg", "1m")
    random = circuit.vtrrandom("rand", "out", "0", "gaussian", "10m", 0, 1, 0)

    assert isinstance(pwl.value, SourceValue)
    assert pwl.value.form == "PWL"
    assert pwl.value.values == (0, 0, "1n", 1)
    assert isinstance(am.value, SourceValue)
    assert am.value.form == "AM"
    assert am.value.values == ("0.5", 1, "20k", "5meg", "1m")
    assert isinstance(random.value, SourceValue)
    assert random.value.form == "TRRANDOM"
    assert random.value.values == (2, "10m", 0, 1, 0)


@pytest.mark.parametrize(
    ("distribution", "code"),
    [
        ("uniform", 1),
        ("gaussian", 2),
        ("exponential", 3),
        ("poisson", 4),
    ],
)
def test_trrandom_helpers_render_exact_ngspice_distribution_codes(distribution, code):
    circuit = Circuit("trrandom")
    voltage = circuit.vtrrandom("rand", "out", "0", distribution, "10m")
    current = circuit.itrrandom("irand", "out", "0", distribution, "10m")

    assert isinstance(voltage.value, SourceValue)
    assert isinstance(current.value, SourceValue)
    assert voltage.value.values == (code, "10m", 0, 1, 0)
    assert current.value.values == (code, "10m", 0, 1, 0)
    assert f"Vrand out 0 TRRANDOM({code} 10m 0 1 0)" in render_ngspice(circuit)
    assert f"Iirand out 0 TRRANDOM({code} 10m 0 1 0)" in render_ngspice(circuit)


def test_render_directives_and_preserve_include_param():
    circuit = Circuit("directives")
    circuit.include("/models/base.mod")
    circuit.param("scale", 1)
    circuit.parameter("bias", "vdd/2")
    circuit.model("dmod", "D", is_=1e-14)
    circuit.lib("/models/process.lib", "tt")
    circuit.global_("vdd", "vss")
    circuit.nodeset(out=0)
    circuit.ic(clk=0)
    circuit.options(reltol="1e-4")
    circuit.save("v(out)", "i(vdd)")
    circuit.probe("v(in)")
    circuit.print_("tran", "v(out)")
    circuit.measure("tran", "tphl", "TRIG v(in) VAL=0.5 RISE=1 TARG v(out) VAL=0.5 FALL=1")
    circuit.measure("tran", "tplh", "TRIG v(in) VAL=0.5 FALL=1", "TARG v(out) VAL=0.5 RISE=1")
    circuit.raw_directive(".control")
    circuit.raw_directive("quit")
    circuit.raw_directive(".endc")

    assert render_ngspice(circuit) == (
        "directives\n"
        '.include "/models/base.mod"\n'
        ".param scale=1\n"
        ".param bias=vdd/2\n"
        ".model dmod D (is=1e-14)\n"
        ".lib /models/process.lib tt\n"
        ".global vdd vss\n"
        ".nodeset v(out)=0\n"
        ".ic v(clk)=0\n"
        ".options reltol=1e-4\n"
        ".save v(out) i(vdd)\n"
        ".probe v(in)\n"
        ".print tran v(out)\n"
        ".measure tran tphl TRIG v(in) VAL=0.5 RISE=1 TARG v(out) VAL=0.5 FALL=1\n"
        ".measure tran tplh TRIG v(in) VAL=0.5 FALL=1 TARG v(out) VAL=0.5 RISE=1\n"
        ".control\n"
        "quit\n"
        ".endc\n"
        ".end\n"
    )


def test_native_include_and_lib_directives_are_deduplicated():
    circuit = Circuit("deduped model refs")

    circuit.include("/models/base.mod")
    circuit.include("/models/base.mod")
    lib = circuit.lib("/models/process.lib", "tt")
    duplicate = circuit.lib("/models/process.lib", "tt")

    assert duplicate is lib
    assert circuit.includes == ["/models/base.mod"]
    assert render_ngspice(circuit) == (
        "deduped model refs\n"
        '.include "/models/base.mod"\n'
        ".lib /models/process.lib tt\n"
        ".end\n"
    )


def test_raw_block_adds_ordered_raw_directives_from_text_and_iterables():
    circuit = Circuit("raw block")
    text_directives = circuit.raw_block(
        """
        .control
        run
        .endc
        """
    )
    iterable_directives = circuit.raw_block([".save v(out)", "write result.raw all"])

    assert tuple(directive.name for directive in text_directives) == (".control", "run", ".endc")
    assert tuple(directive.name for directive in iterable_directives) == (
        ".save v(out)",
        "write result.raw all",
    )
    assert all(directive.raw for directive in (*text_directives, *iterable_directives))
    assert render_ngspice(circuit) == (
        "raw block\n"
        ".control\n"
        "run\n"
        ".endc\n"
        ".save v(out)\n"
        "write result.raw all\n"
        ".end\n"
    )


def test_options_accept_spice_flags_without_boolean_suffixes():
    circuit = Circuit("option flags")
    circuit.options("acct", savecurrents=True, reltol="1e-4")

    assert render_ngspice(circuit) == (
        "option flags\n"
        ".options acct savecurrents reltol=1e-4\n"
        ".end\n"
    )


def test_model_card_records_are_reusable_and_render_as_model_directives():
    circuit = Circuit("model cards")
    nmos = ModelCard.create("nch", "NMOS", level=1, vto=0.4, lambda_=0.02)
    pmos = nmos.clone(name="pch", model_type="PMOS", vto=-0.4)
    switch = circuit.model_card("swmod", "SW", vt=0.5, vh=0.1)

    assert nmos.parameters == ("level", "vto", "lambda")
    assert nmos["lambda_"] == 0.02
    assert nmos.level == 1
    assert nmos.vto == 0.4
    assert nmos.lambda_ == 0.02
    assert nmos.get("missing", "fallback") == "fallback"
    assert nmos.has_parameter("level")
    assert nmos.has_parameter("lambda_")
    assert not nmos.has_parameter("missing")
    with pytest.raises(AttributeError, match="missing"):
        _ = nmos.missing
    assert pmos.model_type == "PMOS"
    assert pmos["vto"] == -0.4
    assert pmos.vto == -0.4
    assert switch["vt"] == 0.5
    assert switch.vt == 0.5

    circuit.model(nmos)
    pmos.apply(circuit)

    assert render_ngspice(circuit) == (
        "model cards\n"
        ".model swmod SW (vt=0.5 vh=0.1)\n"
        ".model nch NMOS (level=1 vto=0.4 lambda=0.02)\n"
        ".model pch PMOS (level=1 vto=-0.4 lambda=0.02)\n"
        ".end\n"
    )

    target = Circuit("copied model card")
    copied = nmos.copy_to(target)
    copied.params["vto"] = 0.7

    assert copied.to_spice() == ".model nch NMOS (level=1 vto=0.7 lambda=0.02)"
    assert nmos.vto == 0.4
    assert render_ngspice(target) == (
        "copied model card\n"
        ".model nch NMOS (level=1 vto=0.7 lambda=0.02)\n"
        ".end\n"
    )
    with pytest.raises(NetlistError, match="copy target"):
        nmos.copy_to(object())


def test_native_netlist_can_pre_register_loose_nodes_without_rendering():
    circuit = Circuit("loose nodes")

    assert circuit.get_node("probe", create=True) == "probe"
    assert circuit.get_node("PROBE", create=True) == "probe"
    assert circuit.node("bias", create=True) == "bias"
    assert circuit.node_names == ("probe", "bias")
    assert circuit["PROBE"] == "probe"
    assert render_ngspice(circuit) == "loose nodes\n.end\n"

    target = Circuit("copied")
    circuit.copy_to(target)

    assert target.node_names == ("probe", "bias")
    assert target.node("BIAS") == "bias"
    with pytest.raises(NetlistError, match="node name is required"):
        circuit.get_node("", create=True)
    with pytest.raises(NetlistError, match="cannot contain newlines"):
        circuit.node("bad\nnode", create=True)


def test_directive_params_may_use_name_key_without_colliding_with_api_names():
    circuit = Circuit("directive name params")
    circuit.model("dmod", "D", **{"name": "alias"})
    circuit.nodeset(**{"name": 0})
    circuit.options(**{"name": "deck"})

    assert render_ngspice(circuit) == (
        "directive name params\n"
        ".model dmod D (name=alias)\n"
        ".nodeset v(name)=0\n"
        ".options name=deck\n"
        ".end\n"
    )


def test_render_logical_model_reference_directive():
    circuit = Circuit("logical refs")
    circuit.model_ref(
        techlib="PTM_BULK",
        corner="ptm65",
        deck="ptm_bulk_65nm",
        section="ptm65",
        simulator="ngspice",
    )

    assert render_ngspice(circuit) == (
        "logical refs\n"
        ".monata_model_ref techlib=PTM_BULK corner=ptm65 "
        "deck=ptm_bulk_65nm section=ptm65 simulator=ngspice\n"
        ".end\n"
    )


def test_duplicate_rendered_element_names_are_rejected():
    circuit = Circuit()
    circuit.resistor("1", "a", "b", "1k")

    with pytest.raises(NetlistError, match="duplicate rendered element name"):
        circuit.resistor("R1", "b", "0", "2k")


def test_extended_source_helpers_reject_invalid_waveform_constraints():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="PWL source value requires at least one"):
        circuit.vpwl("pwl", "out", "0")

    with pytest.raises(NetlistError, match="PWL source value expects time/value pairs"):
        circuit.ipwl("pwl", "out", "0", (0, 1), ("2n",))

    with pytest.raises(NetlistError, match="parameter r value"):
        circuit.vpwl("bad_option", "out", "0", (0, 0), repeat_time="1n\n")

    with pytest.raises(NetlistError, match="unsupported TRRANDOM distribution"):
        circuit.vtrrandom("rand", "out", "0", "triangular", "10m")

    with pytest.raises(NetlistError, match="AM source value cannot contain newlines"):
        circuit.vam("am", "out", "0", "0.5\n", 1, "20k", "5meg", "1m")


def test_missing_element_lookup_and_invalid_directives_fail_explicitly():
    circuit = Circuit()

    with pytest.raises(NetlistError, match="element not found"):
        circuit.element("missing")

    with pytest.raises(NetlistError, match="subcircuit not found"):
        circuit.subcircuit("missing")

    with pytest.raises(NetlistError, match=".model requires name and type"):
        circuit.model("only_name", "")

    with pytest.raises(NetlistError, match=".lib requires a path"):
        circuit.lib("")

    with pytest.raises(NetlistError, match=r"\.measure requires at least one expression"):
        circuit.measure("tran", "empty")
