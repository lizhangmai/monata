import platform
import sys

import pytest

from monata.models.diagnostics import ModelDiagnosticError
from monata.models.artifacts import ModelArtifact, artifact_sha256
from monata.models.flow import SimulationModelConfig
from monata.models.resolver import resolve_model_flow
from monata.runtime.capabilities import CapabilityState, SimulatorCapabilities, native_level_profile, ngspice_profile
from support.model_cases import _flow_test_techlib, _write_osdi_sidecar
pytestmark = pytest.mark.slow

def test_model_artifact_and_simulator_capabilities_round_trip(tmp_path):
    osdi = tmp_path / "bsimcmg.osdi"
    osdi.write_text("compiled")
    artifact = ModelArtifact(
        kind="osdi",
        path=str(osdi),
        role="compiled",
        requires={"osdi": True},
        content_hash=artifact_sha256(osdi),
        package_policy="external_read_only_artifact",
    )
    capabilities = SimulatorCapabilities(
        name="ngspice",
        dialect="ngspice",
        native_spice_model_levels=frozenset({1, 49, 73}),
        osdi=CapabilityState.SUPPORTED,
    )

    assert ModelArtifact.from_dict(artifact.to_dict()) == artifact
    assert SimulatorCapabilities.from_dict(capabilities.to_dict()) == capabilities
    assert capabilities.supports_native_level(49)
    assert not capabilities.supports_native_level(72)


def test_simulator_capability_payloads_reject_unknown_serialized_fields():
    capabilities = SimulatorCapabilities(name="ngspice", dialect="ngspice")
    capabilities_payload = capabilities.to_dict()
    capabilities_payload["unexpected"] = True

    with pytest.raises(TypeError, match="unknown simulator capabilities fields: unexpected"):
        SimulatorCapabilities.from_dict(capabilities_payload)

    profile = ngspice_profile()
    profile_payload = profile.to_dict()
    profile_payload["unexpected"] = True

    with pytest.raises(TypeError, match="unknown simulator profile fields: unexpected"):
        type(profile).from_dict(profile_payload)


def test_model_artifact_rejects_unknown_serialized_fields(tmp_path):
    osdi = tmp_path / "bsimcmg.osdi"
    osdi.write_text("compiled")

    with pytest.raises(TypeError, match="unknown model artifact fields: unexpected"):
        ModelArtifact.from_dict({
            "kind": "osdi",
            "path": str(osdi),
            "role": "compiled",
            "unexpected": True,
        })


def test_model_resolver_selects_native_flow_by_capability_not_simulator_name(tmp_path):
    techlib = _flow_test_techlib(tmp_path)
    profile = native_level_profile(name="brand-new-simulator", dialect="future-spice", levels=(72,))

    resolved = resolve_model_flow(
        techlib,
        "tt",
        simulator_profile=profile,
        model_config=SimulationModelConfig(simulator_profile=profile),
    )

    assert resolved.flow_name == "raw-level72"
    assert resolved.model_selection.lib_sections == [(str(tmp_path / "models" / "raw.mod"), "tt")]
    assert resolved.model_selection.osdi_paths == []
    assert resolved.reuse_signature.startswith("sha256:")


def test_model_resolver_external_osdi_requires_trusted_sidecar(tmp_path):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    external = tmp_path / "external" / "bsimcmg.osdi"
    external.parent.mkdir()
    external.write_text("external")
    profile = ngspice_profile(
        osdi=CapabilityState.SUPPORTED,
        osdi_api_versions=("0.4",),
    )
    config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        external_osdi_paths=(str(external.parent),),
    )

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)
    assert excinfo.value.diagnostic.code == "external_osdi_untrusted"

    _write_osdi_sidecar(external)

    resolved = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)

    assert resolved.flow_name == "osdi-openvaf"
    assert resolved.model_selection.osdi_paths == [str(external)]
    assert resolved.artifacts[0].validation["validation_identity"] == "probe:test"


def test_model_resolver_allows_external_osdi_with_probe_validation(tmp_path):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    external = tmp_path / "external" / "bsimcmg.osdi"
    external.parent.mkdir()
    external.write_text("external")
    profile = ngspice_profile(
        osdi=CapabilityState.SUPPORTED,
        probes={
            "external_osdi_validation": {
                "status": "passed",
                "artifact_path": str(external),
                "validation_identity": "probe:unit-test",
            }
        },
    )
    config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        external_osdi_paths=(str(external.parent),),
    )

    resolved = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)

    assert resolved.model_selection.osdi_paths == [str(external)]
    assert resolved.artifacts[0].validation["validation_identity"] == "probe:unit-test"


def test_model_resolver_rejects_external_osdi_with_unbound_probe_validation(tmp_path):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    external = tmp_path / "external" / "bsimcmg.osdi"
    external.parent.mkdir()
    external.write_text("external")
    profile = ngspice_profile(
        osdi=CapabilityState.SUPPORTED,
        probes={"external_osdi_validation": "passed"},
    )
    config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        external_osdi_paths=(str(external.parent),),
    )

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)

    assert excinfo.value.diagnostic.code == "external_osdi_untrusted"


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"schema_version": 999}, "external_osdi_untrusted"),
        ({"artifact_path": "/wrong/path/bsimcmg.osdi"}, "external_osdi_untrusted"),
        ({"target_platform": "darwin-test"}, "osdi_artifact_incompatible"),
        ({"target_platform": sys.platform}, "osdi_artifact_incompatible"),
        ({"target_platform": {"sys_platform": sys.platform}}, "osdi_artifact_incompatible"),
        (
            {
                "target_platform": {
                    "sys_platform": sys.platform,
                    "machine": "arm64" if platform.machine().lower() != "arm64" else "x86_64",
                }
            },
            "osdi_artifact_incompatible",
        ),
        ({"osdi_api": "9.9"}, "osdi_artifact_incompatible"),
        ({"simulator_profiles": {"ngspice-subprocess": {"dialect": "xyce"}}}, "osdi_artifact_incompatible"),
    ],
)
def test_model_resolver_rejects_untrusted_external_osdi_sidecar(tmp_path, override, code):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False)
    external = tmp_path / "external" / "bsimcmg.osdi"
    external.parent.mkdir()
    external.write_text("external")
    _write_osdi_sidecar(external, override)
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED, osdi_api_versions=("0.4",))
    config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        external_osdi_paths=(str(external.parent),),
    )

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)

    assert excinfo.value.diagnostic.code == code


def test_model_resolver_reads_external_osdi_search_path_from_environment(tmp_path, monkeypatch):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False)
    external = tmp_path / "external" / "bsimcmg.osdi"
    external.parent.mkdir()
    external.write_text("external")
    _write_osdi_sidecar(external)
    monkeypatch.setenv("MONATA_OSDI_PATH", str(external.parent))
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED, osdi_api_versions=("0.4",))
    config = SimulationModelConfig(simulator_profile=profile, policy="osdi-first")

    resolved = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)

    assert resolved.model_selection.osdi_paths == [str(external)]


def test_model_resolver_enforces_subckt_wrapper_capability(tmp_path):
    techlib = _flow_test_techlib(tmp_path, extra_requires={"supports_subckt_wrappers": True})
    profile = native_level_profile(levels=(72,))

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(simulator_profile=profile),
        )

    assert excinfo.value.diagnostic.code == "model_flow_unsupported_by_simulator"
    assert excinfo.value.diagnostic.context["available_capabilities"] == {
        "supports_subckt_wrappers": "unsupported"
    }


def test_model_resolver_bundled_only_ignores_external_osdi_and_user_source(tmp_path, monkeypatch):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    source = tmp_path / "bsimcmg.va"
    source.write_text("module bsimcmg; endmodule\n")
    external = tmp_path / "external" / "bsimcmg.osdi"
    external.parent.mkdir()
    external.write_text("external")
    _write_osdi_sidecar(external)
    monkeypatch.setenv("MONATA_BSIMCMG_SOURCE", str(source))
    monkeypatch.setenv("MONATA_OSDI_PATH", str(external.parent))
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED, osdi_api_versions=("0.4",))

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(
                simulator_profile=profile,
                policy="bundled-only",
                pinned_flow="osdi-openvaf",
            ),
        )

    assert excinfo.value.diagnostic.code == "model_cache_missing"
