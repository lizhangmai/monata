from __future__ import annotations

import os
from typing import Any, cast

import pytest

import monata.projection as projection_module
from monata.corner import OperatingCorner
from monata.library import Library
from monata.netlist import Circuit, render_ngspice
from monata.projection import project_pdk_instances, resolve_pdk_corner
from monata.techlib.parse import (
    parse_corner,
    parse_device,
    parse_model_deck,
    parse_techlib_attachments,
)
from monata.techlib.registry import Techlib, TechlibRegistry
from monata.techlib.schema import TechlibError


def _write_minimal_techlib(root, name="PTM_MG"):
    techlib_dir = root / name
    model_dir = techlib_dir / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ptm_mg_models.mod").write_text(".LIB ptm20hp\n.ENDL ptm20hp\n")
    (techlib_dir / "techlib.toml").write_text(
        f"""
[techlib]
name = "{name}"
description = "test techlib"
default_corner = "ptm20hp"

[[model_decks]]
name = "ptm_mg"
path = "models/ptm_mg_models.mod"

[[corners]]
name = "ptm20hp"
model_deck = "ptm_mg"
section = "ptm20hp"
nominal_vdd = 0.9
process_node = "20nm"
flavor = "hp"
device_defaults = {{ nfet = {{ l = "20n", nfin = 15 }} }}

[[devices]]
name = "nfet"
kind = "mosfet"
pins = ["d", "g", "s", "b"]

[devices.params.l]
default = "lg"
unit = "m"

[devices.params.nfin]
default = 1
type = "integer"

[[devices.views]]
name = "symbol"
primitive = "symbol"
pin_order = ["d", "g", "s", "b"]

[[devices.views]]
name = "ngspice"
primitive = "subckt"
subckt = "nfet"
pin_order = ["d", "g", "s", "b"]
params = ["l", "nfin"]
model_deck = "ptm_mg"
"""
    )
    return techlib_dir


def test_techlib_loads_device_corner_and_model_selection(tmp_path):
    techlib = Techlib.load(_write_minimal_techlib(tmp_path))

    assert techlib.name == "PTM_MG"
    assert techlib.device("nfet").view("ngspice").pin_order == ("d", "g", "s", "b")
    assert techlib.supported_corner_names() == ("ptm20hp",)
    assert techlib.default_operating_corner().name == "ptm20hp"
    assert techlib.corner("ptm20hp").process_node == "20nm"
    assert techlib.corner("ptm20hp").device_defaults == {
        "nfet": {"l": "20n", "nfin": 15}
    }
    selection = techlib.model_selection("ptm20hp")
    assert selection.lib_sections == [
        (str(tmp_path / "PTM_MG" / "models" / "ptm_mg_models.mod"), "ptm20hp")
    ]


def test_techlib_corner_parser_rejects_legacy_corner_field_names():
    with pytest.raises(TechlibError, match="unknown fields: node, voltage"):
        parse_corner({
            "name": "tt",
            "model_deck": "models",
            "node": "legacy-node",
            "voltage": {"vdd": 0.8},
        })


def test_techlib_model_deck_parser_rejects_unknown_fields():
    with pytest.raises(
        TechlibError, match="model deck ptm has unknown fields: unexpected"
    ):
        parse_model_deck({
            "name": "ptm",
            "path": "models/ptm.mod",
            "unexpected": True,
        })


def test_techlib_device_parser_rejects_unknown_fields():
    with pytest.raises(TechlibError, match="device nfet has unknown fields: unexpected"):
        parse_device({
            "name": "nfet",
            "kind": "mosfet",
            "pins": ["d", "g", "s", "b"],
            "unexpected": True,
            "views": [
                {
                    "name": "symbol",
                    "primitive": "symbol",
                    "pin_order": ["d", "g", "s", "b"],
                }
            ],
        })


def test_techlib_param_parser_rejects_unknown_fields():
    with pytest.raises(TechlibError, match="parameter l has unknown fields: unexpected"):
        parse_device({
            "name": "nfet",
            "kind": "mosfet",
            "pins": ["d", "g", "s", "b"],
            "params": {"l": {"default": "lg", "unexpected": True}},
            "views": [
                {
                    "name": "symbol",
                    "primitive": "symbol",
                    "pin_order": ["d", "g", "s", "b"],
                }
            ],
        })


def test_techlib_view_parser_rejects_unknown_fields():
    with pytest.raises(TechlibError, match="view ngspice has unknown fields: unexpected"):
        parse_device({
            "name": "nfet",
            "kind": "mosfet",
            "pins": ["d", "g", "s", "b"],
            "views": [
                {
                    "name": "ngspice",
                    "primitive": "subckt",
                    "subckt": "nfet",
                    "pin_order": ["d", "g", "s", "b"],
                    "unexpected": True,
                }
            ],
        })


def test_techlib_attachments_parser_rejects_unknown_fields():
    with pytest.raises(
        TechlibError, match="techlib attachments has unknown fields: unexpected"
    ):
        parse_techlib_attachments({"techlibs": [], "unexpected": True})

    with pytest.raises(
        TechlibError,
        match="techlib attachment PTM_MG has unknown fields: unexpected",
    ):
        parse_techlib_attachments({
            "techlibs": [{"name": "PTM_MG", "unexpected": True}]
        })


def test_techlib_from_dict_rejects_unknown_root_fields(tmp_path):
    with pytest.raises(TechlibError, match="techlib.toml has unknown fields: unexpected"):
        Techlib.from_dict(
            {
                "techlib": {"name": "PTM_MG"},
                "unexpected": True,
            },
            root=tmp_path,
        )


def test_techlib_from_dict_rejects_unknown_header_fields(tmp_path):
    with pytest.raises(
        TechlibError, match="techlib header has unknown fields: unexpected"
    ):
        Techlib.from_dict(
            {
                "techlib": {
                    "name": "PTM_MG",
                    "unexpected": True,
                }
            },
            root=tmp_path,
        )


def test_mos_view_can_map_models_per_corner(tmp_path):
    techlib_dir = tmp_path / "PTM_BULK"
    model_dir = techlib_dir / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ptm_65nm_bulk.mod").write_text(".LIB ptm65\n.ENDL ptm65\n")
    (model_dir / "ptm_45nm_hp_bulk.mod").write_text(".LIB ptm45hp\n.ENDL ptm45hp\n")
    (techlib_dir / "techlib.toml").write_text(
        """
[techlib]
name = "PTM_BULK"
default_corner = "ptm65"

[[model_decks]]
name = "ptm_bulk_65nm"
path = "models/ptm_65nm_bulk.mod"

[[model_decks]]
name = "ptm_bulk_45nm_hp"
path = "models/ptm_45nm_hp_bulk.mod"

[[corners]]
name = "ptm65"
model_deck = "ptm_bulk_65nm"
section = "ptm65"

[[corners]]
name = "ptm45hp"
model_deck = "ptm_bulk_45nm_hp"
section = "ptm45hp"

[[devices]]
name = "nmos"
kind = "mosfet"
pins = ["d", "g", "s", "b"]

[devices.params.w]
default = "1u"

[devices.params.l]
default = "65n"

[[devices.views]]
name = "ngspice"
primitive = "mos"
pin_order = ["d", "g", "s", "b"]
params = ["w", "l"]

[devices.views.corner_models]
ptm65 = "ptm65nm_nmos"
ptm45hp = "ptm45nm_hp_nmos"
"""
    )
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("bulk corner")
    source = circuit.pdk_instance(
        "mn",
        lib="PTM_BULK",
        cell="nmos",
        view="ngspice",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
        params={"l": "45n"},
    )

    projection = registry.validate_instance(
        source,
        attachments=["PTM_BULK"],
        corner="ptm45hp",
    ).project()
    projection.apply_to(circuit)

    assert projection.element.kind == "M"
    assert projection.element.model == "ptm45nm_hp_nmos"
    assert render_ngspice(circuit) == (
        "bulk corner\n"
        f".lib {tmp_path / 'PTM_BULK' / 'models' / 'ptm_45nm_hp_bulk.mod'} ptm45hp\n"
        "\n"
        "Mmn out in 0 0 ptm45nm_hp_nmos w=1u l=45n\n"
        ".end\n"
    )


def test_techlib_validates_required_corner_runtime_metadata(tmp_path):
    techlib = Techlib.load(_write_minimal_techlib(tmp_path))

    corner = techlib.validate_corner_metadata(
        "ptm20hp",
        require_nominal_vdd=True,
        require_model_deck=True,
        require_model_section=True,
        required_device_defaults={"nfet": ("l", "nfin")},
    )

    assert corner.name == "ptm20hp"


def test_techlib_reports_missing_corner_runtime_metadata_by_field(tmp_path):
    techlib_dir = _write_minimal_techlib(tmp_path)
    text = (techlib_dir / "techlib.toml").read_text()
    (techlib_dir / "techlib.toml").write_text(
        text.replace("nominal_vdd = 0.9\n", "").replace('section = "ptm20hp"\n', "")
    )
    techlib = Techlib.load(techlib_dir)

    with pytest.raises(
        TechlibError,
        match="nominal_vdd, section, device_defaults.nfet.missing_param",
    ):
        techlib.validate_corner_metadata(
            "ptm20hp",
            require_nominal_vdd=True,
            require_model_section=True,
            required_device_defaults={"nfet": ("l", "missing_param")},
        )


def test_techlib_rejects_view_pin_order_unknown_pin(tmp_path):
    techlib_dir = _write_minimal_techlib(tmp_path)
    text = (techlib_dir / "techlib.toml").read_text()
    (techlib_dir / "techlib.toml").write_text(
        text.replace(
            'pin_order = ["d", "g", "s", "b"]',
            'pin_order = ["d", "g", "s", "bulk"]',
            1,
        )
    )

    with pytest.raises(TechlibError, match="unknown pins"):
        Techlib.load(techlib_dir)


@pytest.mark.parametrize("path", ["/abs/model.mod", "../escape.mod", "models/../escape.mod"])
def test_techlib_rejects_model_deck_paths_outside_root(tmp_path, path):
    techlib_dir = _write_minimal_techlib(tmp_path)
    text = (techlib_dir / "techlib.toml").read_text()
    (techlib_dir / "techlib.toml").write_text(
        text.replace('path = "models/ptm_mg_models.mod"', f'path = "{path}"')
    )

    with pytest.raises(TechlibError, match="relative to the techlib root"):
        Techlib.load(techlib_dir)


def test_techlib_rejects_view_param_unknown_param(tmp_path):
    techlib_dir = _write_minimal_techlib(tmp_path)
    text = (techlib_dir / "techlib.toml").read_text()
    (techlib_dir / "techlib.toml").write_text(
        text.replace('params = ["l", "nfin"]', 'params = ["length_typo"]', 1)
    )

    with pytest.raises(TechlibError, match="unknown params"):
        Techlib.load(techlib_dir)


def test_techlib_rejects_corner_model_unknown_corner(tmp_path):
    techlib_dir = _write_minimal_techlib(tmp_path)
    text = (techlib_dir / "techlib.toml").read_text()
    (techlib_dir / "techlib.toml").write_text(
        text.replace(
            'params = ["l", "nfin"]\nmodel_deck = "ptm_mg"',
            'params = ["l", "nfin"]\nmodel_deck = "ptm_mg"\n'
            'corner_models = { missing = "nfet_missing" }',
        )
    )

    with pytest.raises(TechlibError, match="unknown corners"):
        Techlib.load(techlib_dir)


def test_registry_explicit_search_path_and_entry_point_discovery(tmp_path, monkeypatch):
    techlib_dir = _write_minimal_techlib(tmp_path)

    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    assert registry.list_techlibs() == ["PTM_MG"]

    class EntryPoint:
        def load(self):
            return lambda: [techlib_dir]

    class EntryPoints(list):
        def select(self, group):
            assert group == "monata.techlibs"
            return self

    monkeypatch.setattr(
        "monata.techlib.registry.metadata.entry_points",
        lambda: EntryPoints([EntryPoint()]),
    )
    discovered = TechlibRegistry(search_paths=[], auto_discover=True)
    assert discovered.list_techlibs() == ["PTM_MG"]


def test_registry_uses_monata_techlib_path_when_search_paths_are_default(tmp_path, monkeypatch):
    first = _write_minimal_techlib(tmp_path / "first", name="PTM_MG")
    second = _write_minimal_techlib(tmp_path / "second", name="PTM_BULK")
    monkeypatch.setenv(
        "MONATA_TECHLIB_PATH",
        os.pathsep.join([str(first.parent), str(second.parent)]),
    )

    registry = TechlibRegistry(auto_discover=False)

    assert registry.list_techlibs() == ["PTM_BULK", "PTM_MG"]


def test_registry_uses_monata_home_techlib_dir_when_present(tmp_path, monkeypatch):
    home = tmp_path / "monata-home"
    _write_minimal_techlib(home / "techlibs", name="PTM_MG")
    monkeypatch.setenv("MONATA_HOME", str(home))
    monkeypatch.delenv("MONATA_TECHLIB_PATH", raising=False)

    registry = TechlibRegistry(auto_discover=False)

    assert registry.list_techlibs() == ["PTM_MG"]


def test_registry_uses_default_monata_home_techlib_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_minimal_techlib(
        home / ".monata" / "techlibs",
        name="PTM_MG",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("MONATA_HOME", raising=False)
    monkeypatch.delenv("MONATA_TECHLIB_PATH", raising=False)

    registry = TechlibRegistry(auto_discover=False)

    assert registry.list_techlibs() == ["PTM_MG"]


def test_registry_explicit_empty_search_paths_skip_default_resource_paths(tmp_path, monkeypatch):
    _write_minimal_techlib(tmp_path, name="PTM_MG")
    monkeypatch.setenv("MONATA_TECHLIB_PATH", str(tmp_path))

    registry = TechlibRegistry(search_paths=[], auto_discover=False)

    assert registry.list_techlibs() == []


def test_registry_records_broken_optional_entry_points_without_failing(tmp_path, monkeypatch):
    techlib_dir = _write_minimal_techlib(tmp_path)

    class BrokenEntryPoint:
        name = "broken-techlib"

        def load(self):
            raise ModuleNotFoundError("missing optional techlib")

    class ValidEntryPoint:
        name = "valid-techlib"

        def load(self):
            return lambda: [techlib_dir]

    class EntryPoints(list):
        def select(self, group):
            assert group == "monata.techlibs"
            return self

    monkeypatch.setattr(
        "monata.techlib.registry.metadata.entry_points",
        lambda: EntryPoints([BrokenEntryPoint(), ValidEntryPoint()]),
    )

    registry = TechlibRegistry(search_paths=[], auto_discover=True)

    assert registry.list_techlibs() == ["PTM_MG"]
    assert len(registry.discovery_errors) == 1
    assert registry.discovery_errors[0].group == "monata.techlibs"
    assert registry.discovery_errors[0].entry_point == "broken-techlib"
    assert "missing optional techlib" in registry.discovery_errors[0].message


def test_registry_strict_discovery_raises_for_broken_entry_points(monkeypatch):
    class BrokenEntryPoint:
        name = "broken-techlib"

        def load(self):
            raise ModuleNotFoundError("missing optional techlib")

    class EntryPoints(list):
        def select(self, group):
            assert group == "monata.techlibs"
            return self

    monkeypatch.setattr(
        "monata.techlib.registry.metadata.entry_points",
        lambda: EntryPoints([BrokenEntryPoint()]),
    )

    with pytest.raises(TechlibError, match="broken-techlib"):
        TechlibRegistry(search_paths=[], auto_discover=True, strict_discovery=True)


def test_registry_rejects_duplicate_techlib_names_from_different_roots(tmp_path):
    first = _write_minimal_techlib(tmp_path / "first")
    second = _write_minimal_techlib(tmp_path / "second")
    registry = TechlibRegistry(search_paths=[first.parent], auto_discover=False)

    with pytest.raises(TechlibError, match="duplicate techlib name PTM_MG"):
        registry.add_search_path(second.parent)


def test_registry_allows_same_techlib_root_to_be_added_twice(tmp_path):
    techlib_dir = _write_minimal_techlib(tmp_path)
    registry = TechlibRegistry(search_paths=[techlib_dir], auto_discover=False)

    registry.add_search_path(techlib_dir)

    assert registry.list_techlibs() == ["PTM_MG"]


def test_library_attachments_are_additive_and_optional(tmp_path):
    lib_dir = tmp_path / "analog"
    lib_dir.mkdir()
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "analog"\ndescription = "test"\n\n'
        '[technology]\nmodel_paths = ["legacy.mod"]\n\n'
        '[attachments]\ntechlibs = ["PTM_MG"]\ndefault_corner = "ptm20hp"\n'
    )

    lib = Library(lib_dir)
    assert lib.tech_model_paths == ["legacy.mod"]
    assert lib.attached_techlibs == ["PTM_MG"]
    assert lib.techlib_attachments[0].default_corner == "ptm20hp"

    no_attachment_dir = tmp_path / "plain"
    no_attachment_dir.mkdir()
    (no_attachment_dir / "lib.toml").write_text(
        '[library]\nname = "plain"\ndescription = "test"\n\n'
        '[technology]\nmodel_paths = []\n'
    )
    assert Library(no_attachment_dir).techlib_attachments == []


def test_library_consumes_attachments_when_validating_pdk_instance(tmp_path):
    _write_minimal_techlib(tmp_path)
    lib_dir = tmp_path / "analog"
    lib_dir.mkdir()
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "analog"\ndescription = "test"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_MG"]\ndefault_corner = "ptm20hp"\n'
    )
    lib = Library(lib_dir)
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("library attached")
    source = circuit.pdk_instance(
        "mn",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
    )

    validated = lib.validate_pdk_instance(source, registry=registry)

    assert validated.corner is not None
    assert isinstance(validated.corner, OperatingCorner)
    assert validated.corner.name == "ptm20hp"
    assert validated.ordered_nets == ("out", "in", "0", "0")


def test_library_rejects_name_only_corner_resolution_with_multiple_techlibs(tmp_path):
    lib_dir = tmp_path / "analog"
    lib_dir.mkdir()
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "analog"\ndescription = "test"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_MG", "PTM_BULK"]\n'
    )

    with pytest.raises(ValueError, match="ambiguous"):
        Library(lib_dir).resolve_pdk_corner("ptm20hp")


def test_corner_device_defaults_fill_omitted_params_and_explicit_params_win(tmp_path):
    _write_minimal_techlib(tmp_path)
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("defaults")
    omitted = circuit.pdk_instance(
        "mn_default",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
    )
    explicit = circuit.pdk_instance(
        "mn_explicit",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out2", "g": "in", "s": "0", "b": "0"},
        params={"l": "18n", "nfin": 2},
    )

    default_projection = registry.validate_instance(
        omitted,
        attachments=["PTM_MG"],
        corner="ptm20hp",
    ).project()
    explicit_projection = registry.validate_instance(
        explicit,
        attachments=["PTM_MG"],
        corner=registry["PTM_MG"].corner("ptm20hp"),
    ).project()

    assert default_projection.element.params["l"] == "20n"
    assert default_projection.element.params["nfin"] == 15
    assert explicit_projection.element.params["l"] == "18n"
    assert explicit_projection.element.params["nfin"] == 2


def test_pdk_instance_preserves_identity_and_projects_to_existing_ir(tmp_path):
    _write_minimal_techlib(tmp_path)
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("ptm smoke")
    source = circuit.pdk_instance(
        "mn",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
        params={"nfin": 2},
    )

    validated = registry.validate_instance(
        source,
        attachments=["PTM_MG"],
        corner="ptm20hp",
    )
    projection = validated.project()
    projection.apply_to(circuit)

    assert source.lib == "PTM_MG"
    assert source.cell == "nfet"
    assert source.view == "ngspice"
    assert tuple(source.pins) == ("d", "g", "s", "b")
    assert validated.ordered_nets == ("out", "in", "0", "0")
    assert projection.source is source
    assert projection.element.kind == "X"
    assert projection.element.model == "nfet"
    assert projection.element.params["l"] == "20n"
    assert projection.element.params["nfin"] == 2
    assert render_ngspice(circuit) == (
        "ptm smoke\n"
        f".lib {tmp_path / 'PTM_MG' / 'models' / 'ptm_mg_models.mod'} ptm20hp\n"
        "\n"
        "Xmn out in 0 0 nfet l=20n nfin=2\n"
        ".end\n"
    )


def test_library_can_project_pdk_instances_as_logical_model_refs(tmp_path):
    _write_minimal_techlib(tmp_path)
    lib_dir = tmp_path / "analog"
    lib_dir.mkdir()
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "analog"\ndescription = "test"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_MG"]\ndefault_corner = "ptm20hp"\n'
    )
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("logical attached")
    circuit.pdk_instance(
        "mn",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
        params={"nfin": 2},
    )

    Library(lib_dir).project_pdk_instances(
        circuit,
        registry=registry,
        reference_mode="logical",
    )

    netlist = render_ngspice(circuit)
    assert ".lib " not in netlist
    assert str(tmp_path) not in netlist
    assert (
        ".monata_model_ref techlib=PTM_MG corner=ptm20hp "
        "deck=ptm_mg section=ptm20hp simulator=ngspice"
    ) in netlist
    assert "Xmn out in 0 0 nfet l=20n nfin=2" in netlist


def test_library_can_project_pdk_instances_without_model_directives(tmp_path):
    _write_minimal_techlib(tmp_path)
    lib_dir = tmp_path / "analog"
    lib_dir.mkdir()
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "analog"\ndescription = "test"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_MG"]\ndefault_corner = "ptm20hp"\n'
    )
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("resolver aware")
    circuit.pdk_instance(
        "mn",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
        params={"nfin": 2},
    )

    Library(lib_dir).project_pdk_instances(
        circuit,
        registry=registry,
        reference_mode="concrete",
        include_models=False,
    )

    netlist = render_ngspice(circuit)
    assert ".lib " not in netlist
    assert ".monata_model_ref" not in netlist
    assert "Xmn out in 0 0 nfet l=20n nfin=2" in netlist


def test_projection_service_uses_library_techlib_boundary(tmp_path):
    _write_minimal_techlib(tmp_path)
    lib_dir = tmp_path / "analog"
    lib_dir.mkdir()
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "analog"\ndescription = "test"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_MG"]\ndefault_corner = "ptm20hp"\n'
    )
    library = Library(lib_dir)
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("projection service")
    circuit.pdk_instance(
        "mn",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
        params={"nfin": 2},
    )

    resolved = resolve_pdk_corner(library, "ptm20hp", registry=registry)
    result = project_pdk_instances(
        library,
        circuit,
        registry=registry,
        reference_mode="logical",
    )

    assert resolved is not None
    assert resolved.name == "ptm20hp"
    assert result is circuit
    netlist = render_ngspice(circuit)
    assert ".monata_model_ref techlib=PTM_MG corner=ptm20hp" in netlist
    assert "Xmn out in 0 0 nfet l=20n nfin=2" in netlist


def test_project_pdk_instances_reuses_default_registry_for_batch(monkeypatch):
    created = []

    class Owner:
        techlib_attachments = ("PTM_MG",)

    class FakeRegistry:
        def __init__(self):
            self.instances = []
            created.append(self)

        def validate_instance(self, instance, **_kwargs):
            self.instances.append(instance)
            return FakeValidated(instance)

    class FakeValidated:
        def __init__(self, instance):
            self.instance = instance

        def project(self):
            return FakeProjection(self.instance)

    class FakeProjection:
        model_selection = None

        def __init__(self, instance):
            self.instance = instance

        def apply_to(self, _scope, *, include_models=True, reference_mode="concrete"):
            self.instance.params["projected"] = f"{include_models}:{reference_mode}"

    circuit = Circuit("batch projection")
    circuit.pdk_instance(
        "mn1",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out1", "g": "in", "s": "0", "b": "0"},
    )
    circuit.pdk_instance(
        "mn2",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out2", "g": "in", "s": "0", "b": "0"},
    )
    monkeypatch.setattr(projection_module, "_default_techlib_registry", FakeRegistry)

    project_pdk_instances(Owner(), circuit)

    assert len(created) == 1
    assert len(created[0].instances) == 2
    assert circuit.pdk_instances == []


def test_library_rejects_unknown_model_reference_mode(tmp_path):
    _write_minimal_techlib(tmp_path)
    lib_dir = tmp_path / "analog"
    lib_dir.mkdir()
    (lib_dir / "lib.toml").write_text(
        '[library]\nname = "analog"\ndescription = "test"\n\n'
        '[technology]\nmodel_paths = []\n\n'
        '[attachments]\ntechlibs = ["PTM_MG"]\ndefault_corner = "ptm20hp"\n'
    )
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("bad reference mode")
    circuit.pdk_instance(
        "mn",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
    )

    with pytest.raises(ValueError, match="unsupported model reference mode"):
        Library(lib_dir).project_pdk_instances(
            circuit,
            registry=registry,
            reference_mode=cast(Any, "portable"),
        )


def test_symbol_view_is_valid_source_identity_but_not_runtime_projectable(tmp_path):
    _write_minimal_techlib(tmp_path)
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit("symbol source")
    source = circuit.pdk_instance(
        "mn",
        lib="PTM_MG",
        cell="nfet",
        view="symbol",
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
    )

    validated = registry.validate_instance(source, attachments=["PTM_MG"])

    assert validated.view.primitive == "symbol"
    assert validated.ordered_nets == ("out", "in", "0", "0")
    with pytest.raises(TechlibError, match="source-only"):
        validated.project()
    with pytest.raises(TechlibError, match="source-only"):
        registry.validate_instance(
            source,
            attachments=["PTM_MG"],
            require_projectable=True,
        )


@pytest.mark.parametrize(
    ("pins", "params", "match"),
    [
        ({"d": "out", "g": "in", "s": "0"}, {}, "missing pins"),
        ({"d": "out", "g": "in", "s": "0", "b": "0", "x": "bad"}, {}, "unknown pins"),
        ({"d": "out", "g": "in", "s": "0", "b": "0"}, {"bad": 1}, "unknown params"),
    ],
)
def test_pdk_instance_validation_reports_targeted_diagnostics(tmp_path, pins, params, match):
    _write_minimal_techlib(tmp_path)
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit()
    source = circuit.pdk_instance(
        "mn",
        lib="PTM_MG",
        cell="nfet",
        view="ngspice",
        pins=pins,
        params=params,
    )

    with pytest.raises(TechlibError, match=match):
        registry.validate_instance(source, attachments=["PTM_MG"])


@pytest.mark.parametrize(
    ("lib_name", "cell", "view", "match"),
    [
        ("UNKNOWN", "nfet", "ngspice", "unattached techlib"),
        ("PTM_MG", "unknown", "ngspice", "unknown device cell"),
        ("PTM_MG", "nfet", "unknown", "unknown device view"),
    ],
)
def test_pdk_instance_validation_reports_unknown_target_diagnostics(
    tmp_path,
    lib_name,
    cell,
    view,
    match,
):
    _write_minimal_techlib(tmp_path)
    registry = TechlibRegistry(search_paths=[tmp_path], auto_discover=False)
    circuit = Circuit()
    source = circuit.pdk_instance(
        "mn",
        lib=lib_name,
        cell=cell,
        view=view,
        pins={"d": "out", "g": "in", "s": "0", "b": "0"},
    )

    with pytest.raises(TechlibError, match=match):
        registry.validate_instance(source, attachments=["PTM_MG"])
