from pathlib import Path

import pytest

from monata.netlist import Circuit, SubCircuit, render_ngspice
from monata.spice_library import SpiceLibrary, SpiceLibraryAsset, SpiceLibraryError, SpiceLibraryReference


def test_spice_library_scans_models_subcircuits_and_pins(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "analog.lib").write_text(
        """
* demo library
.model nch NMOS level=1
.subckt opamp inp inn out vdd vss gain=10
R1 inp out 1k
.ends opamp
"""
    )

    library = SpiceLibrary(library_root)

    assert library["nch"].kind == "model"
    assert library["nch"].model_type == "NMOS"
    assert library["nch"].category == "mosfet"
    assert {"model", "mosfet", "nmos"}.issubset(library["nch"].tags)
    assert library["opamp"].kind == "subckt"
    assert library["opamp"].pins == ("inp", "inn", "out", "vdd", "vss")
    assert library["opamp"].category == "amplifier"
    assert library.include_path("opamp") == library_root / "analog.lib"
    assert library.cache_path.exists()


def test_spice_library_loads_cache_without_rescanning(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    source = library_root / "devices.cir"
    source.write_text(".model pch PMOS level=1\n")
    first = SpiceLibrary(library_root)
    source.unlink()

    cached = SpiceLibrary(library_root, scan=False)

    assert first.cache_path == cached.cache_path
    assert cached["pch"].path == Path(library_root / "devices.cir")


def test_spice_library_search_and_ambiguous_lookup(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "mixed.lib").write_text(
        """
.model amp NPN bf=100
.subckt amp in out
.ends amp
"""
    )

    library = SpiceLibrary(library_root)

    assert [item.kind for item in library.find("amp")] == ["model", "subckt"]
    assert [item.kind for item in library.search("amp")] == ["model", "subckt"]
    assert [item.kind for item in library.search("amp", kind="subckt")] == ["subckt"]
    assert library.get("amp", kind="model").model_type == "NPN"
    assert bool(library)
    with pytest.raises(SpiceLibraryError, match="ambiguous"):
        library["amp"]


def test_empty_spice_library_is_falsey_without_rescanning(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    library = SpiceLibrary(library_root, scan=False)

    assert not library
    assert library.search("anything") == ()


def test_spice_library_returns_indexed_model_and_subcircuit_source(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "mixed.lib").write_text(
        """
.model nch NMOS (
+ level=1
+ vto=0.4
+)
.subckt opamp inp inn out
R1 inp out 1k
.ends opamp
"""
    )

    library = SpiceLibrary(library_root)

    assert library.source("nch", kind="model") == ".model nch NMOS (\n+ level=1\n+ vto=0.4\n+)\n"
    assert library["opamp"].source_text() == ".subckt opamp inp inn out\nR1 inp out 1k\n.ends opamp\n"


def test_spice_library_exposes_source_assets_by_file_and_section(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    source = library_root / "mixed.lib"
    source.write_text(
        """
.model dfast D is=1e-15
.lib ff
* fast corner
.model nfet NMOS level=1
.subckt inv in out vdd vss
.ends inv
.endl ff
"""
    )
    library = SpiceLibrary(library_root)

    plain = library.asset(source)
    section = library.asset(source, section="FF")

    assert isinstance(plain, SpiceLibraryAsset)
    assert plain.section is None
    assert [item.name for item in plain.items] == ["dfast"]
    assert list(plain.models) == ["dfast"]
    assert plain.categories["diode"][0].name == "dfast"
    assert plain.source_text() == source.read_text()
    assert section.section == "ff"
    assert [item.name for item in section.items] == ["nfet", "inv"]
    assert list(section.models) == ["nfet"]
    assert list(section.subcircuits) == ["inv"]
    assert section.source_text() == (
        ".lib ff\n"
        "* fast corner\n"
        ".model nfet NMOS level=1\n"
        ".subckt inv in out vdd vss\n"
        ".ends inv\n"
        ".endl ff\n"
    )
    assert [(asset.path, asset.section) for asset in library.assets] == [(source, None), (source, "ff")]
    with pytest.raises(KeyError):
        library.asset(source, section="ss")


def test_spice_library_source_reports_missing_subcircuit_end(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "broken.lib").write_text(".subckt broken in out\nR1 in out 1k\n")
    library = SpiceLibrary(library_root)

    with pytest.raises(SpiceLibraryError, match="missing .ends"):
        library.source("broken")


def test_spice_library_filters_by_category_and_tags(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "mixed.lib").write_text(
        """
.lib ff
.model nfet NMOS level=1
.endl
.model d1 D is=1e-15
.subckt inv in out vdd vss
.ends inv
"""
    )

    library = SpiceLibrary(library_root)

    assert [item.name for item in library.by_category("mosfet")] == ["nfet"]
    assert [item.name for item in library.by_category("diode")] == ["d1"]
    assert [item.name for item in library.tagged("section:ff")] == ["nfet"]
    assert [item.name for item in library.categories["logic"]] == ["inv"]


def test_spice_library_manages_category_directories(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    library = SpiceLibrary(library_root, scan=False)

    created = library.add_category("vendor/mouser/opamps")

    assert created == library_root / "vendor" / "mouser" / "opamps"
    assert library.category_path("vendor/mouser/opamps") == created
    assert library.list_categories() == (
        "vendor",
        "vendor/mouser",
        "vendor/mouser/opamps",
    )


def test_spice_library_category_directories_validate_roots_and_names(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    library = SpiceLibrary(library_root, scan=False)

    with pytest.raises(SpiceLibraryError, match="invalid SPICE library category"):
        library.add_category("../escape")
    with pytest.raises(SpiceLibraryError, match="invalid SPICE library category"):
        library.category_path("")

    file_root = tmp_path / "one.lib"
    file_root.write_text(".model d1 D\n")
    file_library = SpiceLibrary(file_root)
    with pytest.raises(SpiceLibraryError, match="require a directory root"):
        file_library.list_categories()


def test_spice_library_applies_toml_catalog_metadata(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "analog.lib").write_text(
        """
.model nch NMOS level=1
.subckt opamp inp inn out vdd vss
.ends opamp
"""
    )
    catalog = tmp_path / "catalog.toml"
    catalog.write_text(
        """
[items.opamp]
kind = "subckt"
category = "Vendor/Mouser/Linear Amplifiers"
tags = ["vendor:mouser", "product:opamp"]

[items."model:nch"]
category = "Vendor/Infineon/MOSFETs"
tags = "power"
"""
    )

    library = SpiceLibrary(library_root, catalog=catalog)

    assert library["opamp"].category == "vendor/mouser/linear amplifiers"
    assert {"vendor:mouser", "product:opamp", "vendor_mouser_linear_amplifiers"}.issubset(library["opamp"].tags)
    assert library["nch"].category == "vendor/infineon/mosfets"
    assert {"power", "vendor_infineon_mosfets", "mosfet", "nmos"}.issubset(library["nch"].tags)
    assert [item.name for item in library.by_category("vendor/infineon/mosfets")] == ["nch"]


def test_spice_library_apply_catalog_mapping_and_cache_roundtrip(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "mixed.lib").write_text(
        """
.model d1 D is=1e-15
.subckt gain in out
.ends gain
"""
    )
    library = SpiceLibrary(library_root)

    returned = library.apply_catalog(
        {
            "items": {
                "subckt:gain": {"category": "vendor/local/amplifiers", "tags": ["favorite", "lab"]},
                "model:d1": "vendor/local/diodes",
            }
        }
    )
    library.save_cache()
    cached = SpiceLibrary(library_root, scan=False)

    assert returned is library
    assert cached["gain"].category == "vendor/local/amplifiers"
    assert {"favorite", "lab", "vendor_local_amplifiers"}.issubset(cached["gain"].tags)
    assert cached["d1"].category == "vendor/local/diodes"
    assert "vendor_local_diodes" in cached["d1"].tags


def test_spice_library_catalog_reports_unknown_ambiguous_or_invalid_items(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "mixed.lib").write_text(
        """
.model amp NPN bf=100
.subckt amp in out
.ends amp
"""
    )
    library = SpiceLibrary(library_root)

    with pytest.raises(SpiceLibraryError, match="ambiguous"):
        library.apply_catalog({"items": {"amp": {"category": "vendor/local/amplifiers"}}})
    with pytest.raises(SpiceLibraryError, match="catalog item not found"):
        library.apply_catalog({"items": {"model:missing": {"category": "vendor/local/models"}}})
    with pytest.raises(SpiceLibraryError, match="invalid SPICE library category"):
        library.apply_catalog({"items": {"model:amp": {"category": "../escape"}}})
    with pytest.raises(SpiceLibraryError, match="SPICE library catalog item model:amp has unknown fields: tag"):
        library.apply_catalog({"items": {"model:amp": {"category": "vendor/local/models", "tag": "typo"}}})
    with pytest.raises(SpiceLibraryError, match="unsupported SPICE library catalog format"):
        library.load_catalog(tmp_path / "catalog.yaml")


def test_spice_library_reference_attaches_include_and_lib_sections(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    source = library_root / "mixed.lib"
    source.write_text(
        """
.model d1 D is=1e-15
.lib ff
.model nfet NMOS level=1
.endl
"""
    )
    library = SpiceLibrary(library_root)
    circuit = Circuit("uses library")

    plain = library.attach(circuit, "d1", kind="model")
    section = library.attach(circuit, "nfet", kind="model")
    library.attach(circuit, "nfet", kind="model")

    assert isinstance(plain, SpiceLibraryReference)
    assert plain.directive_name == "include"
    assert section.directive_name == "lib"
    assert section.section == "ff"
    rendered = render_ngspice(circuit)
    assert rendered.count(f'.include "{source}"') == 1
    assert rendered.count(f".lib {source} ff") == 1


def test_scope_include_accepts_library_items_and_references(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    source = library_root / "mixed.lib"
    source.write_text(
        """
.model d1 D is=1e-15
.lib ff
.model nfet NMOS level=1
.endl
"""
    )
    library = SpiceLibrary(library_root)
    circuit = Circuit("direct include")

    circuit.include(library["d1"])
    circuit.include(library["nfet"])
    circuit.include(library.reference("nfet", kind="model"))

    rendered = render_ngspice(circuit)
    assert rendered.count(f'.include "{source}"') == 1
    assert rendered.count(f".lib {source} ff") == 1


def test_spice_library_can_follow_inner_include_references_from_file_roots(tmp_path):
    top = tmp_path / "top.lib"
    vendor = tmp_path / "vendor.inc"
    top.write_text(
        """
.include "vendor.inc"
.subckt wrapper in out
X1 in out vendor_cell
.ends wrapper
"""
    )
    vendor.write_text(
        """
.model dfast D is=1e-15
.subckt vendor_cell in out
D1 in out dfast
.ends vendor_cell
"""
    )

    library = SpiceLibrary(top, follow_references=True)

    assert library["dfast"].path == vendor
    assert library["vendor_cell"].path == vendor
    assert library["wrapper"].path == top


def test_spice_library_can_follow_external_lib_references_by_section(tmp_path):
    top = tmp_path / "top.lib"
    corners = tmp_path / "corners.lib"
    top.write_text('.lib "corners.lib" ff\n')
    corners.write_text(
        """
.lib tt
.model ntt NMOS level=1
.endl
.lib ff
.model nff NMOS level=1
.endl
"""
    )

    library = SpiceLibrary(top, follow_references=True)

    assert library["nff"].path == corners
    assert library["nff"].section == "ff"
    assert "section:ff" in library["nff"].tags
    assert "ntt" not in library


def test_spice_library_reference_can_attach_to_subcircuit_scope(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    source = library_root / "analog.lib"
    source.write_text(
        """
.lib tt
.model nch NMOS level=1
.endl
"""
    )
    library = SpiceLibrary(library_root)
    subckt = SubCircuit("gain", ("in", "out", "vdd", "vss"))
    subckt.mos("1", "out", "in", "vss", "vss", "nch")

    reference = library.reference("nch", kind="model").apply(subckt)
    circuit = Circuit("top")
    circuit.subckt(subckt)
    circuit.instance("x1", ("in", "out", "vdd", "0"), "gain")

    assert reference.directive_name == "lib"
    rendered = render_ngspice(circuit)
    assert rendered.count(f".lib {source} tt") == 1
    assert ".subckt gain in out vdd vss" in rendered


def test_spice_library_instantiates_indexed_subcircuits_by_named_pins(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    source = library_root / "analog.lib"
    source.write_text(
        """
.subckt opamp inp inn out vdd vss gain=10
E1 out vss inp inn 10
.ends opamp
"""
    )
    library = SpiceLibrary(library_root)
    circuit = Circuit("library instance")
    pins = {"out": "y", "inn": "b", "inp": "a", "vss": "0", "vdd": "vdd"}

    item = library["opamp"]
    element = library.instantiate(circuit, "u1", "opamp", pins, m=2)
    library.reference("opamp", kind="subckt").instantiate(circuit, "u2", pins)

    assert item.ordered_nodes(pins) == ("a", "b", "y", "vdd", "0")
    assert element.nodes == ("a", "b", "y", "vdd", "0")
    rendered = render_ngspice(circuit)
    assert rendered.count(f'.include "{source}"') == 1
    assert "Xu1 a b y vdd 0 opamp m=2" in rendered
    assert "Xu2 a b y vdd 0 opamp" in rendered


def test_spice_library_named_pin_instantiation_reports_library_errors(tmp_path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    (library_root / "mixed.lib").write_text(
        """
.model nch NMOS level=1
.subckt gain in out
.ends gain
"""
    )
    library = SpiceLibrary(library_root)
    gain = library["gain"]

    with pytest.raises(SpiceLibraryError, match="missing pin"):
        gain.ordered_nodes({"in": "a"})
    with pytest.raises(SpiceLibraryError, match="unknown pin"):
        gain.ordered_nodes({"in": "a", "out": "y", "vdd": "vdd"})
    with pytest.raises(SpiceLibraryError, match="only indexed subcircuits"):
        library["nch"].ordered_nodes({})
    with pytest.raises(SpiceLibraryError, match="only subcircuit"):
        library.reference("nch", kind="model").instantiate(Circuit(), "m1", {})
