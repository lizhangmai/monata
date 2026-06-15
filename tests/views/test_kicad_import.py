from pathlib import Path

import pytest

from monata.eda.kicad import (
    KiCadImportError,
    KiCadImportPolicy,
    import_kicad_netlist,
    inspect_kicad_netlist,
    kicad_netlist_to_circuit,
    kicad_netlist_to_python,
    parse_kicad_netlist,
)
from monata.netlist import render_ngspice
from monata.workspace import Project


KICAD_XML = """<?xml version="1.0" encoding="utf-8"?>
<export version="D">
  <design>
    <source>amp.kicad_sch</source>
  </design>
  <components>
    <comp ref="R1">
      <value>1k</value>
      <libsource lib="Device" part="R"/>
    </comp>
    <comp ref="C1">
      <value>1u</value>
      <libsource lib="Device" part="C"/>
    </comp>
    <comp ref="U1">
      <value>inv</value>
      <fields>
        <field name="spice_pins">1 2 3 4</field>
        <field name="spice_subckt">inv</field>
      </fields>
      <libsource lib="monata" part="inv"/>
    </comp>
  </components>
  <nets>
    <net code="1" name="in">
      <node ref="R1" pin="1"/>
      <node ref="U1" pin="1"/>
    </net>
    <net code="2" name="out">
      <node ref="R1" pin="2"/>
      <node ref="C1" pin="1"/>
      <node ref="U1" pin="2"/>
    </net>
    <net code="3" name="GND">
      <node ref="C1" pin="2"/>
      <node ref="U1" pin="4"/>
    </net>
    <net code="4" name="vdd">
      <node ref="U1" pin="3"/>
    </net>
  </nets>
</export>
"""


def test_parse_kicad_netlist_reads_components_and_nets():
    netlist = parse_kicad_netlist(KICAD_XML)

    assert netlist.title == "amp"
    assert netlist.source == "amp.kicad_sch"
    assert netlist.component("R1").value == "1k"
    assert netlist.component("U1").field_value("spice_subckt") == "inv"
    assert netlist.nets[0].nodes[0].ref == "R1"


def test_inspect_kicad_netlist_projects_supported_components():
    plan = inspect_kicad_netlist(KICAD_XML)

    assert plan.supported is True
    assert plan.projected_count == 3
    assert plan.unsupported_count == 0
    assert plan.steps[2].ref == "U1"
    assert plan.steps[2].kind == "X"
    assert plan.steps[2].nodes == ("in", "out", "vdd", "0")
    assert plan.steps[2].model == "inv"


def test_kicad_netlist_to_circuit_projects_to_native_ir():
    circuit = kicad_netlist_to_circuit(KICAD_XML)
    rendered = render_ngspice(circuit)

    assert circuit.title == "amp"
    assert "R1 in out 1k" in rendered
    assert "C1 out 0 1u" in rendered
    assert "XU1 in out vdd 0 inv" in rendered


def test_kicad_netlist_to_python_exports_executable_builder_source():
    source = kicad_netlist_to_python(KICAD_XML, function_name="build_amp")
    namespace = {}

    exec(source, namespace)
    circuit = namespace["build_amp"]()
    rendered = render_ngspice(circuit)

    assert source.startswith("from monata.netlist import Circuit, Element")
    assert "def build_amp():" in source
    assert "Element('R', 'R1', ('in', 'out'), value='1k')" in source
    assert "Element('X', 'U1', ('in', 'out', 'vdd', '0'), model='inv')" in source
    assert circuit.title == "amp"
    assert "R1 in out 1k" in rendered
    assert "C1 out 0 1u" in rendered
    assert "XU1 in out vdd 0 inv" in rendered


def test_kicad_netlist_to_python_validates_function_name():
    with pytest.raises(ValueError, match="invalid function_name"):
        kicad_netlist_to_python(KICAD_XML, function_name="not-valid")
    with pytest.raises(ValueError, match="invalid function_name"):
        kicad_netlist_to_python(KICAD_XML, function_name="class")


def test_kicad_import_reports_unsupported_components_without_throwing():
    plan = inspect_kicad_netlist(
        """<export>
  <components>
    <comp ref="TP1"><value>testpoint</value></comp>
    <comp ref="R1"><value>1k</value></comp>
  </components>
  <nets>
    <net code="1" name="sig"><node ref="TP1" pin="1"/><node ref="R1" pin="1"/></net>
    <net code="2" name="0"><node ref="R1" pin="2"/></net>
  </nets>
</export>
"""
    )

    assert plan.supported is False
    assert plan.unsupported_count == 1
    assert plan.issues[0].ref == "TP1"
    assert "missing supported SPICE kind" in plan.issues[0].message
    with pytest.raises(KiCadImportError, match="TP1"):
        plan.to_circuit()


def test_kicad_import_can_override_kind_and_ignore_components():
    xml = """<export>
  <components>
    <comp ref="TP1">
      <value>ignored</value>
      <fields><field name="monata_ignore">true</field></fields>
    </comp>
    <comp ref="RV1">
      <value>10k</value>
      <fields><field name="spice_kind">R</field><field name="spice_pins">2 1</field></fields>
    </comp>
  </components>
  <nets>
    <net code="1" name="a"><node ref="RV1" pin="1"/></net>
    <net code="2" name="b"><node ref="RV1" pin="2"/></net>
  </nets>
</export>
"""

    plan = inspect_kicad_netlist(xml)
    rendered = render_ngspice(plan.to_circuit())

    assert plan.supported is True
    assert plan.ignored_count == 1
    assert "RV1 b a 10k" in rendered


def test_kicad_import_reads_only_explicit_path_objects(tmp_path):
    netlist_path = tmp_path / "amp.xml"
    netlist_path.write_text(KICAD_XML)

    from_path = parse_kicad_netlist(netlist_path)

    assert from_path.path == str(netlist_path)
    assert from_path.title == "amp"
    with pytest.raises(KiCadImportError, match="invalid KiCad XML netlist"):
        parse_kicad_netlist(str(netlist_path))


def test_kicad_import_supports_policy_ref_prefix_overrides():
    xml = """<export>
  <components><comp ref="RV1"><value>10k</value></comp></components>
  <nets>
    <net code="1" name="a"><node ref="RV1" pin="1"/></net>
    <net code="2" name="b"><node ref="RV1" pin="2"/></net>
  </nets>
</export>
"""

    policy = KiCadImportPolicy(ref_kind_overrides={"RV": "R"})
    rendered = render_ngspice(kicad_netlist_to_circuit(xml, policy=policy))

    assert "RV1 a b 10k" in rendered


def test_kicad_import_path_accepts_path_subclasses(tmp_path):
    class CustomPath(Path):
        _flavour = type(tmp_path)._flavour

    netlist_path = CustomPath(tmp_path / "custom.xml")
    netlist_path.write_text(KICAD_XML)

    assert parse_kicad_netlist(netlist_path).title == "amp"


def test_import_kicad_netlist_creates_cell_with_generated_netlist_view(tmp_path):
    project = Project.create(tmp_path / "proj")

    cell = import_kicad_netlist(project, KICAD_XML, library_name="eda", cell_name="amp_from_kicad")

    assert cell.name == "amp_from_kicad"
    assert "eda" in project.list_libraries()
    assert "netlist" in cell
    netlist_view = cell["netlist"]
    netlist_path = netlist_view.load()
    metadata = (cell.path / "import.toml").read_text()

    assert netlist_view.generated is True
    assert netlist_path == cell.path / "netlist.cir"
    assert "R1 in out 1k" in netlist_path.read_text()
    assert "XU1 in out vdd 0 inv" in netlist_path.read_text()
    assert 'format = "kicad-xml-netlist"' in metadata
    assert 'source = "<memory>"' in metadata
    assert 'kicad_source = "amp.kicad_sch"' in metadata
    assert "components = 3" in metadata
    assert "projected = 3" in metadata


def test_import_kicad_netlist_reads_only_explicit_path_objects(tmp_path):
    project = Project.create(tmp_path / "proj")
    netlist_path = tmp_path / "layout.xml"
    netlist_path.write_text(KICAD_XML)

    from_path = import_kicad_netlist(project, netlist_path)

    assert from_path.name == "layout"
    assert f'source = "{netlist_path}"' in (from_path.path / "import.toml").read_text()
    with pytest.raises(KiCadImportError, match="invalid KiCad XML netlist"):
        import_kicad_netlist(project, str(netlist_path), cell_name="inline_path")


def test_import_kicad_netlist_reuses_existing_library(tmp_path):
    project = Project.create(tmp_path / "proj")
    project.create_library("eda")

    first = import_kicad_netlist(project, KICAD_XML, library_name="eda", cell_name="first")
    second = import_kicad_netlist(project, KICAD_XML, library_name="eda", cell_name="second")

    assert first.library.path == second.library.path
    assert project.get_library("eda").list_cells() == ["first", "second"]
