from collections import OrderedDict
from pathlib import Path
import tomllib

from monata.netlist import Circuit, SourceValue, SubCircuit, render_ngspice
from monata.parser import (
    ImportedAsset,
    import_spice_asset,
    import_spice_deck,
    parse_spice_to_circuit,
)
from monata.parser.commands import (
    DOT_COMMANDS_WITH_IMPORT_CONTRACT,
    DOT_FLOW_COMMANDS,
    RAW_PRESERVED_DOT_COMMANDS,
    STRUCTURED_DOT_COMMANDS,
    SUPPORTED_DOT_COMMANDS,
    UNCLASSIFIED_DOT_COMMANDS,
    UNKNOWN_DOT_COMMAND_CONTRACTS,
)
from monata.workspace import Project


def test_supported_dot_commands_have_explicit_import_projection_contract():
    assert DOT_COMMANDS_WITH_IMPORT_CONTRACT == SUPPORTED_DOT_COMMANDS
    assert UNCLASSIFIED_DOT_COMMANDS == frozenset()
    assert UNKNOWN_DOT_COMMAND_CONTRACTS == frozenset()
    assert DOT_FLOW_COMMANDS.isdisjoint(STRUCTURED_DOT_COMMANDS)
    assert DOT_FLOW_COMMANDS.isdisjoint(RAW_PRESERVED_DOT_COMMANDS)
    assert STRUCTURED_DOT_COMMANDS.isdisjoint(RAW_PRESERVED_DOT_COMMANDS)


def test_parse_spice_to_circuit_projects_elements_directives_and_subcircuits():
    circuit = parse_spice_to_circuit(
        """
amp deck
.param vdd=1.2
.model nmos nmos level=1
.subckt inv in out vdd vss
M1 out in vdd vdd pmos w=2u l=0.18u
M2 out in vss vss nmos w=1u l=0.18u
.ends inv
VDD vdd 0 DC 1.2
X1 in out vdd 0 inv m=2
.tran 1n 10n
.end
"""
    )

    assert isinstance(circuit, Circuit)
    assert circuit.params["vdd"] == "1.2"
    assert len(circuit.subcircuits) == 1
    subckt = circuit.subcircuits[0]
    assert isinstance(subckt, SubCircuit)
    assert subckt.name == "inv"
    assert subckt.nodes == ("in", "out", "vdd", "vss")
    assert subckt.element("M1", kind="M").params["w"] == "2u"
    assert circuit.element("X1", kind="X").model == "inv"
    assert circuit.element("VDD", kind="V").value == SourceValue("DC", ("1.2",))
    rendered = render_ngspice(circuit)
    assert ".subckt inv in out vdd vss" in rendered
    assert "X1 in out vdd 0 inv m=2" in rendered
    assert ".tran 1n 10n" in rendered


def test_parse_spice_to_circuit_projects_title_directive_to_circuit_title():
    circuit = parse_spice_to_circuit(
        """
placeholder title
.title final imported deck
R1 in out 1k
.end
"""
    )

    rendered = render_ngspice(circuit)
    assert circuit.title == "final imported deck"
    assert rendered.splitlines()[0] == "final imported deck"
    assert not any(line.startswith(".title") for line in rendered.splitlines())


def test_parse_spice_to_circuit_preserves_element_inline_comments():
    circuit = parse_spice_to_circuit(
        """
comment import
R1 in out 1k ; load branch
.subckt rc in out
C1 in out 1p $ hold cap
.ends rc
.end
"""
    )

    subckt = circuit.subcircuits[0]
    rendered = render_ngspice(circuit)

    assert circuit.element("R1", kind="R").comment == "load branch"
    assert subckt.element("C1", kind="C").comment == "hold cap"
    assert "* hold cap\nC1 in out 1p" in rendered
    assert "* load branch\nR1 in out 1k" in rendered


def test_parse_spice_to_circuit_preserves_standalone_comments():
    circuit = parse_spice_to_circuit(
        """
comment import
* top-level note
R1 in out 1k
.subckt rc in out
; subcircuit note
C1 in out 1p
.ends rc
.end
"""
    )

    subckt = circuit.subcircuits[0]
    rendered = render_ngspice(circuit)

    assert [directive.name for directive in circuit.directives if directive.raw] == ["* top-level note"]
    assert [directive.name for directive in subckt.directives if directive.raw] == ["* subcircuit note"]
    assert "* top-level note" in rendered
    assert "* subcircuit note\nC1 in out 1p" in rendered


def test_parse_spice_to_circuit_projects_passive_positional_models():
    circuit = parse_spice_to_circuit(
        """
passive model deck
R1 in out 1k rmod l=2u w=1u
C1 out 0 1p cmod ic=0.2
L1 vdd out 1n lmod nt=10
.end
"""
    )

    resistor = circuit.element("R1", kind="R")
    capacitor = circuit.element("C1", kind="C")
    inductor = circuit.element("L1", kind="L")

    assert resistor.value == "1k"
    assert resistor.model == "rmod"
    assert resistor.params["l"] == "2u"
    assert resistor.params["w"] == "1u"
    assert capacitor.value == "1p"
    assert capacitor.model == "cmod"
    assert capacitor.params["ic"] == "0.2"
    assert inductor.value == "1n"
    assert inductor.model == "lmod"
    assert inductor.params["nt"] == "10"

    assert render_ngspice(circuit) == (
        "passive model deck\n"
        "R1 in out 1k rmod l=2u w=1u\n"
        "C1 out 0 1p cmod ic=0.2\n"
        "L1 vdd out 1n lmod nt=10\n"
        ".end\n"
    )


def test_parse_spice_to_circuit_projects_line_and_coupling_elements():
    circuit = parse_spice_to_circuit(
        """
line and coupling deck
L1 a 0 1n
L2 b 0 2n
K12 L1 L2 0.9
Tdelay in 0 out 0 z0=50 td=1n
Tfreq a 0 b 0 z0=75 f=1meg nl=0.25
.end
"""
    )

    coupling = circuit.element("K12", kind="K")
    delay = circuit.element("Tdelay", kind="T")
    freq = circuit.element("Tfreq", kind="T")

    assert coupling.nodes == ("L1", "L2")
    assert coupling.value == "0.9"
    assert delay.nodes == ("in", "0", "out", "0")
    assert delay.value == "z0=50 td=1n"
    assert freq.value == "z0=75 f=1meg nl=0.25"
    assert render_ngspice(circuit) == (
        "line and coupling deck\n"
        "L1 a 0 1n\n"
        "L2 b 0 2n\n"
        "K12 L1 L2 0.9\n"
        "Tdelay in 0 out 0 z0=50 td=1n\n"
        "Tfreq a 0 b 0 z0=75 f=1meg nl=0.25\n"
        ".end\n"
    )


def test_parse_spice_to_circuit_projects_semiconductor_device_params_and_flags():
    circuit = parse_spice_to_circuit(
        """
semiconductor import
M1 d g s b nmos w=1u l=45n m=2 ad=1p as=2p nrd=1 nrs=2 off ic=0.1,0.2,0.3 temp=27 nfin=4
D1 out 0 dmod 2 off m=3 pj=1u ic=0.7 temp=27 dtemp=5
Q1 c b e sub npn area=1 areac=2 areab=3 m=4 off ic=0.7,1.2 temp=50 dtemp=3
Q2 c2 b2 e2 pnp off
J1 jd jg js njf 2 off m=3 ic=1,0 temp=27
Z1 zd zg zs nmf 2 off m=3 ic=1,0
.end
"""
    )

    mos = circuit.element("M1", kind="M")
    diode = circuit.element("D1", kind="D")
    bjt = circuit.element("Q1", kind="Q")
    bjt_without_substrate = circuit.element("Q2", kind="Q")
    jfet = circuit.element("J1", kind="J")
    mesfet = circuit.element("Z1", kind="Z")

    assert mos.nodes == ("d", "g", "s", "b")
    assert mos.model == "nmos"
    assert mos.params["off"] is True
    assert mos.params["ic"] == "0.1,0.2,0.3"
    assert mos.params["nfin"] == "4"
    assert diode.params["area"] == "2"
    assert diode.params["off"] is True
    assert bjt.nodes == ("c", "b", "e", "sub")
    assert bjt.model == "npn"
    assert bjt.params["areac"] == "2"
    assert bjt.params["off"] is True
    assert bjt.params["ic"] == "0.7,1.2"
    assert bjt_without_substrate.nodes == ("c2", "b2", "e2")
    assert bjt_without_substrate.model == "pnp"
    assert bjt_without_substrate.params["off"] is True
    assert jfet.params["area"] == "2"
    assert jfet.params["off"] is True
    assert mesfet.params["area"] == "2"
    assert mesfet.params["off"] is True
    rendered = render_ngspice(circuit)
    assert "M1 d g s b nmos w=1u l=45n m=2 ad=1p as=2p nrd=1 nrs=2 ic=0.1,0.2,0.3 temp=27 nfin=4 off" in rendered
    assert "Q2 c2 b2 e2 pnp off" in rendered


def test_parse_spice_to_circuit_projects_controlled_switch_and_behavioral_semantics():
    circuit = parse_spice_to_circuit(
        """
controlled import
Ggm out 0 in 0 1m m=4
Fmirror out 0 Vsense 2 m=3
Ssw out 0 ctrl 0 swmod on
Wcsw out 0 Vsense cswmod off
Bbv out 0 v=V(in)*2 tc1=1m temp=27
Bbi out 0 i=V(in)/1k tc2=2m dtemp=5
.end
"""
    )

    vccs = circuit.element("Ggm", kind="G")
    cccs = circuit.element("Fmirror", kind="F")
    voltage_switch = circuit.element("Ssw", kind="S")
    current_switch = circuit.element("Wcsw", kind="W")
    behavioral_voltage = circuit.element("Bbv", kind="B")
    behavioral_current = circuit.element("Bbi", kind="B")

    assert vccs.params["m"] == "4"
    assert cccs.params["m"] == "3"
    assert voltage_switch.params["initial_state"] == "on"
    assert current_switch.params["initial_state"] == "off"
    assert behavioral_voltage.value == "v=V(in)*2"
    assert behavioral_voltage.params["tc1"] == "1m"
    assert behavioral_voltage.params["temp"] == "27"
    assert behavioral_current.value == "i=V(in)/1k"
    assert behavioral_current.params["tc2"] == "2m"
    assert behavioral_current.params["dtemp"] == "5"
    assert render_ngspice(circuit) == (
        "controlled import\n"
        "Ggm out 0 in 0 1m m=4\n"
        "Fmirror out 0 Vsense 2 m=3\n"
        "Ssw out 0 ctrl 0 swmod on\n"
        "Wcsw out 0 Vsense cswmod off\n"
        "Bbv out 0 v=V(in)*2 tc1=1m temp=27\n"
        "Bbi out 0 i=V(in)/1k tc2=2m dtemp=5\n"
        ".end\n"
    )


def test_parse_spice_to_circuit_projects_nonlinear_controlled_sources():
    circuit = parse_spice_to_circuit(
        """
nonlinear controlled import
Eev out 0 value={V(in)*2}
Ggi out 0 value={V(in)/1k}
Eelap filt 0 LAPLACE {V(in)} {10 / (s/6800 + 1)}
Gglap ifilt 0 LAPLACE {V(in)} {1 / (s + 1)}
Epoly poly 0 POLY(2) a 0 b 0 0 13.6 0.2
Gpoly ipoly 0 POLY(1) a 0 0 1m
.end
"""
    )

    voltage = circuit.element("Eev", kind="E")
    current = circuit.element("Ggi", kind="G")
    laplace_voltage = circuit.element("Eelap", kind="E")
    laplace_current = circuit.element("Gglap", kind="G")
    poly_voltage = circuit.element("Epoly", kind="E")
    poly_current = circuit.element("Gpoly", kind="G")

    assert voltage.nodes == ("out", "0")
    assert voltage.value == "value={V(in)*2}"
    assert current.nodes == ("out", "0")
    assert current.value == "value={V(in)/1k}"
    assert laplace_voltage.nodes == ("filt", "0")
    assert laplace_voltage.value == "LAPLACE {V(in)} {10 / (s/6800 + 1)}"
    assert laplace_current.nodes == ("ifilt", "0")
    assert laplace_current.value == "LAPLACE {V(in)} {1 / (s + 1)}"
    assert poly_voltage.nodes == ("poly", "0")
    assert poly_voltage.value == "POLY(2) a 0 b 0 0 13.6 0.2"
    assert poly_current.nodes == ("ipoly", "0")
    assert poly_current.value == "POLY(1) a 0 0 1m"
    assert render_ngspice(circuit) == (
        "nonlinear controlled import\n"
        "Eev out 0 value={V(in)*2}\n"
        "Ggi out 0 value={V(in)/1k}\n"
        "Eelap filt 0 LAPLACE {V(in)} {10 / (s/6800 + 1)}\n"
        "Gglap ifilt 0 LAPLACE {V(in)} {1 / (s + 1)}\n"
        "Epoly poly 0 POLY(2) a 0 b 0 0 13.6 0.2\n"
        "Gpoly ipoly 0 POLY(1) a 0 0 1m\n"
        ".end\n"
    )


def test_parse_spice_to_circuit_projects_subckt_default_params_without_treating_them_as_pins():
    circuit = parse_spice_to_circuit(
        """
subckt params
.subckt inv in out vdd vss params: wp=2u wn=1u
M1 out in vdd vdd pmos w={wp} l=0.18u
M2 out in vss vss nmos w={wn} l=0.18u
.ends inv
X1 in out vdd 0 inv
.end
"""
    )

    subckt = circuit.subcircuits[0]
    rendered = render_ngspice(circuit)

    assert subckt.nodes == ("in", "out", "vdd", "vss")
    assert subckt.params == {"wp": "2u", "wn": "1u"}
    assert ".subckt inv in out vdd vss wp=2u wn=1u" in rendered
    assert ".param wp=2u" not in rendered
    assert ".param wn=1u" not in rendered
    assert "params:" not in rendered


def test_parse_spice_to_circuit_projects_bare_subckt_default_params():
    circuit = parse_spice_to_circuit(
        """
subckt bare params
.subckt rc in out rload=1k cload=1p
R1 in out {rload}
C1 out 0 {cload}
.ends rc
.end
"""
    )

    subckt = circuit.subcircuits[0]

    assert subckt.nodes == ("in", "out")
    assert subckt.params == {"rload": "1k", "cload": "1p"}
    assert ".subckt rc in out rload=1k cload=1p" in render_ngspice(circuit)


def test_parse_spice_to_circuit_projects_source_value_forms():
    circuit = parse_spice_to_circuit(
        """
source deck
V1 in 0 PULSE(0 1 1n 1n 1n 5n 10n)
I1 out 0 SIN 0 1m 1k 0 0
V2 ac 0 AC 1
.end
"""
    )

    assert circuit.element("V1", kind="V").value == SourceValue(
        "PULSE",
        ("0", "1", "1n", "1n", "1n", "5n", "10n"),
    )
    assert circuit.element("I1", kind="I").value == SourceValue("SIN", ("0", "1m", "1k", "0", "0"))
    assert circuit.element("V2", kind="V").value == SourceValue("AC", ("0", "1"))


def test_parse_spice_to_circuit_projects_pwl_source_options():
    circuit = parse_spice_to_circuit(
        """
pwl deck
V1 out 0 DC 0 PWL(0 0 1n 1 r=10n td=2n)
I1 bias 0 PWL(0 0 1n 1u td=500p)
.end
"""
    )

    assert circuit.element("V1", kind="V").value == SourceValue(
        "PWL",
        ("0", "0", "1n", "1"),
        OrderedDict([("dc", "0"), ("r", "10n"), ("td", "2n")]),
    )
    assert circuit.element("I1", kind="I").value == SourceValue(
        "PWL",
        ("0", "0", "1n", "1u"),
        OrderedDict([("td", "500p")]),
    )
    assert render_ngspice(circuit) == (
        "pwl deck\n"
        "V1 out 0 DC 0 PWL(0 0 1n 1 r=10n td=2n)\n"
        "I1 bias 0 PWL(0 0 1n 1u td=500p)\n"
        ".end\n"
    )


def test_parse_spice_to_circuit_projects_probe_directive():
    circuit = parse_spice_to_circuit(
        """
probe deck
R1 in out 1k
.probe v(out) i(vdd)
.end
"""
    )

    rendered = render_ngspice(circuit)
    assert ".probe v(out) i(vdd)" in rendered


def test_parse_spice_to_circuit_raw_preserves_analysis_dot_commands():
    circuit = parse_spice_to_circuit(
        """
analysis deck
R1 in out 1k
.ac dec 10 1 1e6
.op
.tran 1n 10n
.end
"""
    )

    rendered = render_ngspice(circuit)
    assert ".ac dec 10 1 1e6" in rendered
    assert ".op" in rendered
    assert ".tran 1n 10n" in rendered


def test_parse_spice_to_circuit_raw_preserves_extended_dot_frontend():
    circuit = parse_spice_to_circuit(
        """
extended dots
.func gain(x) {x*2}
.csparam doubled={gain(1)}
.temp 75
.if doubled > 1
R1 in out 1k
.else
R2 in out 2k
.endif
.width out=132
.pss 1k
.noise v(out) vin dec 10 1 1meg
.step param rload list 1k 2k
.end
"""
    )

    rendered = render_ngspice(circuit)

    assert ".func gain(x) {x*2}" in rendered
    assert ".csparam doubled={gain(1)}" in rendered
    assert ".temp 75" in rendered
    assert ".if doubled > 1" in rendered
    assert ".else" in rendered
    assert ".endif" in rendered
    assert ".width out=132" in rendered
    assert ".pss 1k" in rendered
    assert ".noise v(out) vin dec 10 1 1meg" in rendered
    assert ".step param rload list 1k 2k" in rendered


def test_parse_spice_to_circuit_projects_meas_alias_and_preserves_library_endl():
    circuit = parse_spice_to_circuit(
        """
measure aliases
.lib tt
.model n nmos level=1
.endl tt
.meas tran delay FIND v(out) AT=1n
R1 in out 1k
.end
"""
    )

    rendered = render_ngspice(circuit)
    assert ".lib tt" in rendered
    assert ".endl tt" in rendered
    assert ".measure tran delay FIND v(out) AT=1n" in rendered
    assert not any(line.startswith(".meas ") for line in rendered.splitlines())


def test_parse_spice_to_circuit_projects_directive_params_named_name():
    circuit = parse_spice_to_circuit(
        """
directive name params
.model dmod d name=alias
.nodeset v(name)=0
.options name=deck
D1 in out dmod
.end
"""
    )

    rendered = render_ngspice(circuit)
    assert ".model dmod d (name=alias)" in rendered
    assert ".nodeset v(name)=0" in rendered
    assert ".options name=deck" in rendered


def test_parse_spice_to_circuit_projects_option_alias_and_flags():
    circuit = parse_spice_to_circuit(
        """
option aliases
.option acct savecurrents reltol=1e-4
.options noacct method=gear
.end
"""
    )

    first, second = circuit.directives
    assert first.name == "options"
    assert first.params == OrderedDict([("acct", True), ("savecurrents", True), ("reltol", "1e-4")])
    assert second.params == OrderedDict([("noacct", True), ("method", "gear")])

    rendered = render_ngspice(circuit)
    assert ".options acct savecurrents reltol=1e-4" in rendered
    assert ".options noacct method=gear" in rendered
    assert ".option " not in rendered


def test_parse_spice_to_circuit_preserves_element_params_named_like_ir_fields():
    circuit = parse_spice_to_circuit(
        """
element field params
R1 in out 1k model=rmod value=tag kind=meta name=alias params=pmap
M1 d g s b nmos model=override value=vtag kind=ktag name=ntag params=mpmap
.end
"""
    )

    resistor = circuit.element("R1", kind="R")
    mos = circuit.element("M1", kind="M")

    assert resistor.value == "1k"
    assert resistor.model is None
    assert resistor.params["model"] == "rmod"
    assert resistor.params["value"] == "tag"
    assert resistor.params["kind"] == "meta"
    assert resistor.params["name"] == "alias"
    assert resistor.params["params"] == "pmap"

    assert mos.model == "nmos"
    assert mos.value is None
    assert mos.params["model"] == "override"
    assert mos.params["value"] == "vtag"
    assert mos.params["kind"] == "ktag"
    assert mos.params["name"] == "ntag"
    assert mos.params["params"] == "mpmap"

    rendered = render_ngspice(circuit)
    assert "R1 in out 1k model=rmod value=tag kind=meta name=alias params=pmap" in rendered
    assert "M1 d g s b nmos model=override value=vtag kind=ktag name=ntag params=mpmap" in rendered


def test_import_spice_deck_creates_cell_with_netlist_view(tmp_path):
    project = Project.create(tmp_path / "proj")
    cell = import_spice_deck(
        project,
        """
rc deck
R1 in out 1k
C1 out 0 1u
.tran 1n 10n
.end
""",
        library_name="imported",
        cell_name="rc_filter",
    )

    assert cell.name == "rc_filter"
    assert "netlist" in cell
    netlist_view = cell["netlist"]
    netlist_path = netlist_view.load()
    metadata = tomllib.loads((cell.path / "import.toml").read_text())

    assert netlist_view.generated is True
    assert netlist_path == cell.path / "netlist.cir"
    assert "R1 in out 1k" in netlist_path.read_text()
    assert metadata["import"] == {"source": "<memory>", "title": "rc deck"}
    assert "imported" in project.list_libraries()
    assert "rc_filter" in project.get_library("imported")


def test_import_spice_deck_reads_only_explicit_path_objects(tmp_path):
    project = Project.create(tmp_path / "proj")
    source = tmp_path / "from_file.cir"
    source.write_text(
        """
file deck
Rfile in out 1k
.end
"""
    )

    cell = import_spice_deck(project, source)
    metadata = tomllib.loads((cell.path / "import.toml").read_text())

    assert cell.name == "from_file"
    assert "Rfile in out 1k" in (cell.path / "netlist.cir").read_text()
    assert metadata["import"] == {"source": str(source), "title": "file deck"}


def test_import_spice_deck_treats_string_path_collision_as_inline_text(tmp_path):
    project = Project.create(tmp_path / "proj")
    source = tmp_path / "inline.cir"
    source.write_text(
        """
file deck
Rfile in out 1k
.end
"""
    )

    cell = import_spice_deck(project, str(source), cell_name="inline_text")
    metadata = tomllib.loads((cell.path / "import.toml").read_text())
    netlist_text = (cell.path / "netlist.cir").read_text()

    assert "Rfile in out 1k" not in netlist_text
    assert metadata["import"] == {"source": "<memory>", "title": str(source)}


def test_import_spice_deck_coerces_default_cell_name_to_safe_path_segment(tmp_path):
    project = Project.create(tmp_path / "proj")

    cell = import_spice_deck(
        project,
        """
μ amp deck!
R1 in out 1k
.end
""",
    )

    assert cell.name == "amp_deck"
    assert cell.path.name == "amp_deck"


def test_import_spice_asset_writes_subckt_and_model_assets(tmp_path):
    project = Project.create(tmp_path / "proj")
    subckt = import_spice_asset(
        project,
        ".subckt gain in out\nE1 out 0 in 0 2\n.ends gain\n",
        library_name="analog",
        asset_name="gain",
    )
    model = import_spice_asset(
        project,
        ".model nmos nmos level=1\n",
        library_name="analog",
        asset_name="models",
    )

    assert isinstance(subckt, ImportedAsset)
    assert subckt.kind == "subckt"
    assert subckt.path == project.path / "libraries" / "analog" / "assets" / "gain.cir"
    assert ".subckt gain" in subckt.path.read_text()
    assert model.kind == "model"
    assert model.path == project.path / "libraries" / "analog" / "assets" / "models.cir"
    assert "analog" in project.list_libraries()
    assert Path(project.get_library("analog").path / "assets").is_dir()


def test_import_spice_asset_coerces_asset_name_to_safe_path_segment(tmp_path):
    project = Project.create(tmp_path / "proj")

    asset = import_spice_asset(
        project,
        ".model nmos nmos level=1\n",
        asset_name="μ model deck!",
    )

    assert asset.name == "model_deck"
    assert asset.path.name == "model_deck.cir"
