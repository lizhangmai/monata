from pathlib import Path

import pytest

from monata.models.diagnostics import ModelDiagnosticError
from monata.models.flow import ModelFlowRecipe, SimulationModelConfig
from monata.models.resolver import resolve_model_flow
from monata.models.registry import ModelRegistry
from monata.sim.capabilities import CapabilityState, native_level_profile, ngspice_profile
from monata.corner import OperatingCorner
from monata.techlib.registry import Techlib
from monata.techlib.schema import ModelDeck, TechlibError
from support.model_cases import _flow_test_techlib
pytestmark = pytest.mark.slow

def test_model_flow_recipe_metadata_serialization_round_trip():
    recipe = ModelFlowRecipe(
        name="ngspice-osdi",
        model_deck="ptm_mg",
        output="ngspice_osdi",
        next_action="Compile the bundled source before simulation.",
        metadata={"owner": "techlib", "priority": 1},
    )

    restored = ModelFlowRecipe.from_dict(recipe.to_dict())

    assert restored.next_action == "Compile the bundled source before simulation."
    assert restored.metadata == {"owner": "techlib", "priority": 1}
    assert restored.to_dict() == recipe.to_dict()


def test_model_flow_recipe_rejects_unknown_serialized_fields():
    with pytest.raises(TypeError, match="unknown model flow recipe fields: unexpected"):
        ModelFlowRecipe.from_dict({
            "name": "ngspice-osdi",
            "model_deck": "ptm_mg",
            "output": "ngspice_osdi",
            "unexpected": True,
        })


def test_model_registry_register_resolve_and_osdi_paths(tmp_path):
    mos = tmp_path / "bsim4.osdi"
    bjt = tmp_path / "vbic.osdi"
    mos.write_text("mos")
    bjt.write_text("bjt")
    registry = ModelRegistry(auto_discover=False)

    registry.register("mos", mos, module_name="bsim4", level=14, version="4.8")
    registry.register("mos", mos, module_name="bsim4-default", level=54)
    registry.register("bjt", bjt, module_name="vbic")

    resolved_mos = registry.resolve("mos", level=14, version="4.8")
    resolved_default = registry.resolve("mos", level=54, version="latest")
    assert resolved_mos is not None
    assert resolved_default is not None
    assert resolved_mos.module_name == "bsim4"
    assert resolved_default.module_name == "bsim4-default"
    assert registry.resolve_osdi("bjt") == str(bjt)
    assert registry.resolve("d") is None
    assert registry.osdi_paths("mos", level=14, version="4.8") == [str(mos)]
    assert registry.osdi_paths("missing") == []
    assert set(registry.osdi_paths()) == {str(mos), str(bjt)}
    mos_entries = registry.list_models("mos")
    assert len(mos_entries) == 2
    assert {entry.family for entry in mos_entries} == {"mos"}


def test_model_resolver_selects_osdi_flow_when_native_level_is_unavailable(tmp_path):
    techlib = _flow_test_techlib(tmp_path)
    profile = ngspice_profile(
        osdi=CapabilityState.SUPPORTED,
        supports_subckt_wrappers=CapabilityState.SUPPORTED,
        osdi_api_versions=("0.4",),
        probes={"pre_osdi": "passed"},
    )

    resolved = resolve_model_flow(
        techlib,
        "tt",
        simulator_profile=profile,
        model_config=SimulationModelConfig(
            simulator_profile=profile,
            allow_precompiled_package_artifacts=True,
        ),
    )

    assert resolved.flow_name == "osdi-openvaf"
    assert resolved.model_selection.includes == [str(tmp_path / "models" / "converted.mod")]
    assert resolved.model_selection.osdi_paths == [str(tmp_path / "models" / "bsimcmg.osdi")]
    assert resolved.generated_artifacts[0].kind == "converted_model_card"


def test_model_resolver_converts_ptm_mg_osdi_card_to_ngspice_ready_section(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "ptm_mg_models.mod").write_text(
        ".LIB ptm20hp\n"
        ".subckt nfet d g s x l=lg nfin=1\n"
        ".include 'ptm_mg_20nm_hp_nmos.mod'\n"
        "mnfet d g s x nfet L=l NFIN=nfin\n"
        ".ends nfet\n"
        ".ENDL ptm20hp\n"
        ".LIB ptm14hp\n"
        ".subckt nfet d g s x l=lg nfin=1\n"
        ".include 'ptm_mg_14nm_hp_nmos.mod'\n"
        "mnfet d g s x nfet L=l NFIN=nfin\n"
        ".ends nfet\n"
        ".subckt pfet d g s x l=lg nfin=1\n"
        ".include 'ptm_mg_14nm_hp_pmos.mod'\n"
        "mpfet d g s x pfet L=l NFIN=nfin\n"
        ".ends pfet\n"
        ".lib 'ptm_mg_param.mod' 14nm\n"
        ".ENDL ptm14hp\n"
    )
    (model_dir / "ptm_mg_param.mod").write_text(
        ".lib 20nm\n"
        ".param lg=24n\n"
        ".endl\n"
        ".lib 14nm\n"
        ".param lg=18n\n"
        ".param vdd=0.8\n"
        ".endl\n"
    )
    (model_dir / "ptm_mg_20nm_hp_nmos.mod").write_text(".model nfet nmos level = 72\n")
    (model_dir / "ptm_mg_14nm_hp_nmos.mod").write_text(".model nfet nmos level = 72\n")
    (model_dir / "ptm_mg_14nm_hp_pmos.mod").write_text(".model pfet pmos level = 72\n")
    (model_dir / "bsimcmg.osdi").write_text("compiled")
    techlib = Techlib(
        name="PTM_MG",
        root=tmp_path,
        model_decks=[ModelDeck(name="ptm_mg", path="models/ptm_mg_models.mod")],
        model_flows=[
            ModelFlowRecipe(
                name="ptm-mg-ngspice-osdi",
                model_deck="ptm_mg",
                output="ngspice_osdi",
                requires={"osdi": True, "supports_subckt_wrappers": True},
                dialects=("ngspice",),
                module_name="bsimcmg_va",
                osdi_path="models/bsimcmg.osdi",
                converter="ptm_mg_level72_to_bsimcmg",
            )
        ],
        corners=[OperatingCorner(name="ptm14hp", model_deck="ptm_mg", section="ptm14hp")],
    )

    resolved = resolve_model_flow(
        techlib,
        "ptm14hp",
        simulator_profile=ngspice_profile(
            osdi=CapabilityState.SUPPORTED,
            supports_subckt_wrappers=CapabilityState.SUPPORTED,
        ),
        model_config=SimulationModelConfig(
            policy="osdi-first",
            allow_precompiled_package_artifacts=True,
            cache_dir=tmp_path / "cache",
        ),
    )

    converted = Path(resolved.model_selection.includes[0])
    converted_text = converted.read_text()

    assert ".LIB ptm14hp" not in converted_text
    assert ".lib 'ptm_mg_param.mod' 14nm" not in converted_text
    assert ".param lg=18n" in converted_text
    assert ".param lg=24n" not in converted_text
    assert "Nnfet d g s x nfet L=l NFIN=nfin" in converted_text
    assert "Npfet d g s x pfet L=l NFIN=nfin" in converted_text
    assert "ptm_mg_20nm_hp_nmos.mod" not in converted_text
    assert "bsimcmg_va" in (converted.parent / "ptm_mg_14nm_hp_nmos.mod").read_text()
    assert "TYPE = 0" in (converted.parent / "ptm_mg_14nm_hp_pmos.mod").read_text()


def test_model_resolver_explicit_api_requires_simulator_profile(tmp_path):
    techlib = _flow_test_techlib(tmp_path)

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(techlib, "tt", model_config=SimulationModelConfig())

    assert excinfo.value.diagnostic.code == "simulator_profile_required"


def test_model_resolver_requires_declared_model_flow(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    raw = model_dir / "raw.mod"
    raw.write_text(".LIB tt\n.model n nmos level=72\n.ENDL tt\n")
    techlib = Techlib(
        name="FAKE_PTM",
        root=tmp_path,
        model_decks=[ModelDeck(name="raw", path="models/raw.mod")],
        corners=[OperatingCorner(name="tt", model_deck="raw", section="tt")],
    )
    profile = native_level_profile(levels=(72,))

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(simulator_profile=profile),
        )

    assert excinfo.value.diagnostic.code == "model_flow_missing"
    assert techlib.model_selection("tt").lib_sections == [(str(raw), "tt")]


def test_model_resolver_explicit_api_uses_flow_resolution_with_explicit_profile(tmp_path):
    techlib = _flow_test_techlib(tmp_path)
    profile = ngspice_profile()

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(simulator_profile=profile),
        )

    assert excinfo.value.diagnostic.code == "model_flow_unsupported_by_simulator"
    assert excinfo.value.diagnostic.context["attempted_flow"] == "raw-level72"


def test_model_resolver_strict_pinned_raw_flow_rejects_ngspice(tmp_path):
    techlib = _flow_test_techlib(tmp_path)
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(
                simulator_profile=profile,
                policy="strict",
                pinned_flow="raw-level72",
            ),
        )

    assert excinfo.value.diagnostic.code == "model_flow_unsupported_by_simulator"
    assert excinfo.value.diagnostic.context["attempted_flow"] == "raw-level72"


def test_model_resolver_reads_bsimcmg_source_path_from_environment(tmp_path, monkeypatch):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    source = tmp_path / "bsimcmg.va"
    source.write_text("module bsimcmg; endmodule\n")
    monkeypatch.setenv("MONATA_BSIMCMG_SOURCE", str(source))
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)
    config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        allow_external_osdi=False,
        allow_compile=False,
    )

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)

    assert excinfo.value.diagnostic.code == "model_cache_missing"


def test_model_resolver_rejects_recipe_paths_that_escape_techlib_root(tmp_path):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False)
    bad_recipe = ModelFlowRecipe(
        name="escape-osdi",
        model_deck="raw",
        output="ngspice_osdi",
        requires={"osdi": True},
        dialects=("ngspice",),
        module_name="bsimcmg",
        osdi_path="../escape.osdi",
        converted_model_card="models/converted.mod",
    )
    techlib = Techlib(
        name=techlib.name,
        root=techlib.root,
        model_decks=techlib.model_decks.values(),
        model_flows=[bad_recipe],
        corners=techlib.corners.values(),
    )
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)

    with pytest.raises(TechlibError, match="relative to the techlib root"):
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(
                simulator_profile=profile,
                policy="osdi-first",
                allow_precompiled_package_artifacts=True,
            ),
        )


@pytest.mark.parametrize("source_include", ["../defs.include", "/tmp/defs.include"])
def test_model_resolver_rejects_source_includes_that_escape_source_root(tmp_path, source_include):
    source_dir = tmp_path / "sources" / "bsimcmg"
    source_dir.mkdir(parents=True)
    (source_dir / "bsimcmg.va").write_text("module bsimcmg; endmodule\n")
    techlib = _flow_test_techlib(
        tmp_path,
        packaged_osdi=False,
        package_policy="bundled_source",
        source_va="sources/bsimcmg/bsimcmg.va",
        source_includes=(source_include,),
    )
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)

    with pytest.raises(TechlibError, match="model flow source include path must be relative to the source_va directory"):
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(
                simulator_profile=profile,
                policy="osdi-first",
                allow_external_osdi=False,
                cache_dir=tmp_path / "cache",
            ),
        )


def test_model_resolver_rejects_unknown_requires_keys(tmp_path):
    techlib = _flow_test_techlib(tmp_path, extra_requires={"future_magic": True})
    profile = native_level_profile(levels=(72,))

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(simulator_profile=profile),
        )

    assert excinfo.value.diagnostic.code == "model_flow_unsupported_by_simulator"
    assert excinfo.value.diagnostic.context["required_capabilities"] == {"future_magic": True}
