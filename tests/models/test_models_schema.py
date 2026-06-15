
import pytest

from monata.models.diagnostics import ModelDiagnostic, ModelDiagnosticError
from monata.models.manifest import DeviceMetadata, ModelManifest
from monata.models.registry import ModelEntry, ModelRegistry
from monata.netlist import Circuit
pytestmark = pytest.mark.slow

def test_model_entry_core_serialization_round_trip(tmp_path):
    osdi = tmp_path / "bsim4.osdi"
    source = tmp_path / "bsim4.va"
    entry = ModelEntry(
        name="bsim4",
        family="mos",
        osdi_path=osdi,
        module_name="bsim4",
        level=14,
        version="4.8",
        source_va=source,
    )

    data = entry.to_dict()
    restored = ModelEntry.from_dict(data)

    assert data == {
        "name": "bsim4",
        "family": "mos",
        "level": 14,
        "version": "4.8",
        "osdi_path": str(osdi),
        "module_name": "bsim4",
        "source_va": str(source),
    }
    assert restored.to_dict() == data


def test_model_entry_optional_metadata_serialization_round_trip(tmp_path):
    entry = ModelEntry(
        name="nmos",
        family="mos",
        module_name="bsim4",
        model_file=tmp_path / "models.lib",
        lib_section="tt",
        provenance={"manifest": "models.toml"},
        parameters={"w": "device width"},
        include_paths=[tmp_path / "constants.va"],
    )

    data = entry.to_dict()
    restored = ModelEntry.from_dict(data)

    assert data["module_name"] == "bsim4"
    assert data["model_file"] == str(tmp_path / "models.lib")
    assert data["lib_section"] == "tt"
    assert data["provenance"] == {"manifest": "models.toml"}
    assert data["parameters"] == {"w": "device width"}
    assert data["include_paths"] == [str(tmp_path / "constants.va")]
    assert "level" not in data
    assert "version" not in data
    assert "osdi_path" not in data
    assert "source_va" not in data
    assert restored.to_dict() == data


def test_model_entry_rejects_unknown_serialized_fields():
    with pytest.raises(TypeError, match="unknown model entry fields: unexpected"):
        ModelEntry.from_dict({
            "name": "nmos",
            "family": "mos",
            "module_name": "bsim4",
            "unexpected": True,
        })


def test_device_metadata_rejects_unknown_serialized_fields():
    with pytest.raises(TypeError, match="unknown device metadata fields: unexpected"):
        DeviceMetadata.from_dict({
            "name": "nmos",
            "family": "mos",
            "unexpected": True,
        })


def test_model_manifest_reports_unknown_record_fields():
    with pytest.raises(ModelDiagnosticError) as model_exc:
        ModelManifest.from_dict({
            "models": [
                {
                    "name": "nmos",
                    "family": "mos",
                    "module_name": "bsim4",
                    "unexpected": True,
                }
            ]
        })
    assert model_exc.value.diagnostic.code == "model_manifest_invalid"
    assert model_exc.value.diagnostic.message == (
        "models[0] has invalid fields: unknown model entry fields: unexpected"
    )

    with pytest.raises(ModelDiagnosticError) as device_exc:
        ModelManifest.from_dict({
            "devices": [
                {
                    "name": "nmos",
                    "family": "mos",
                    "unexpected": True,
                }
            ]
        })
    assert device_exc.value.diagnostic.code == "model_manifest_invalid"
    assert device_exc.value.diagnostic.message == (
        "devices[0] has invalid fields: unknown device metadata fields: unexpected"
    )


def test_model_manifest_reports_unknown_top_level_fields():
    with pytest.raises(ModelDiagnosticError) as exc:
        ModelManifest.from_dict({"models": [], "unexpected": True})

    assert exc.value.diagnostic.code == "model_manifest_invalid"
    assert exc.value.diagnostic.message == "manifest has unknown fields: unexpected"


def test_model_diagnostic_round_trip_and_error_payload():
    diagnostic = ModelDiagnostic(
        code="model_missing",
        message="OSDI model file not found",
        context={"path": "models/bsim4.osdi", "family": "mos"},
    )

    restored = ModelDiagnostic.from_dict(diagnostic.to_dict())
    error = ModelDiagnosticError(restored)

    assert restored == diagnostic
    assert str(error) == "model_missing: OSDI model file not found"
    assert error.to_dict() == {
        "code": "model_missing",
        "message": "OSDI model file not found",
        "context": {"path": "models/bsim4.osdi", "family": "mos"},
    }


def test_model_diagnostic_rejects_unknown_serialized_fields():
    with pytest.raises(TypeError, match="unknown model diagnostic fields: unexpected"):
        ModelDiagnostic.from_dict({
            "code": "model_missing",
            "message": "OSDI model file not found",
            "unexpected": True,
        })


def test_model_registry_preserves_duplicate_candidates_and_diagnoses_ambiguity(tmp_path):
    first = tmp_path / "first.osdi"
    second = tmp_path / "second.osdi"
    first.write_text("first")
    second.write_text("second")
    registry = ModelRegistry(auto_discover=False)

    registry.register("mos", first, module_name="bsim4-a", level=14, version="4.8")
    registry.register("mos", second, module_name="bsim4-b", level=14, version="4.8")

    assert [entry.module_name for entry in registry.resolve_candidates("mos", level=14, version="4.8")] == [
        "bsim4-a",
        "bsim4-b",
    ]
    resolved = registry.resolve("mos", level=14, version="4.8")
    assert resolved is not None
    assert resolved.module_name == "bsim4-a"
    with pytest.raises(ModelDiagnosticError) as excinfo:
        registry.resolve_strict("mos", level=14, version="4.8")
    assert excinfo.value.diagnostic.code == "model_selection_ambiguous"
    assert [
        entry["module_name"]
        for entry in excinfo.value.diagnostic.context["candidates"]
    ] == ["bsim4-a", "bsim4-b"]
    with pytest.raises(ModelDiagnosticError):
        registry.resolve_osdi_strict("mos", level=14, version="4.8")


def test_model_registry_auto_discovery_preserves_duplicate_candidates(tmp_path):
    first = tmp_path / "a" / "bsim4.osdi"
    second = tmp_path / "b" / "bsim4.osdi"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first")
    second.write_text("second")

    registry = ModelRegistry(search_paths=[tmp_path], auto_discover=True)

    candidates = registry.resolve_candidates("mos")
    assert [entry.osdi_path for entry in candidates] == [str(first), str(second)]
    resolved = registry.resolve("mos")
    assert resolved is not None
    assert resolved.osdi_path == str(first)
    with pytest.raises(ModelDiagnosticError):
        registry.resolve_strict("mos")


def test_model_registry_serialized_entries_round_trip_preserves_candidates(tmp_path):
    first = tmp_path / "first.osdi"
    second = tmp_path / "second.osdi"
    first.write_text("first")
    second.write_text("second")
    registry = ModelRegistry(auto_discover=False)
    registry.load_entries([
        {"name": "bsim4-a", "family": "mos", "module_name": "bsim4-a", "osdi_path": str(first)},
        {"name": "bsim4-b", "family": "mos", "module_name": "bsim4-b", "osdi_path": str(second)},
    ])

    data = registry.to_dict()
    restored = ModelRegistry(auto_discover=False)
    restored.load_entries(data["entries"])

    assert [entry.module_name for entry in restored.list_models("mos")] == ["bsim4-a", "bsim4-b"]
    assert set(restored.osdi_paths()) == {str(first), str(second)}


def test_model_registry_discovers_env_osdi_paths(tmp_path, monkeypatch):
    osdi_dir = tmp_path / "osdi"
    osdi_dir.mkdir()
    (osdi_dir / "bsimcmg.osdi").write_text("model")
    monkeypatch.setenv("MONATA_OSDI_PATH", str(osdi_dir))

    registry = ModelRegistry()

    assert registry.resolve_osdi("mos") == str(osdi_dir / "bsimcmg.osdi")


def test_model_manifest_selection_projects_concrete_circuit_directives(tmp_path):
    include = tmp_path / "common.inc"
    lib = tmp_path / "models.lib"
    osdi = tmp_path / "bsim4.osdi"
    include.write_text(".model d d\n")
    lib.write_text(".lib tt\n.endl\n")
    osdi.write_text("compiled")
    manifest = ModelManifest([
        ModelEntry(name="diodes", family="d", module_name="d", model_file=include),
        ModelEntry(
            name="nmos_tt",
            family="mos",
            module_name="bsim4",
            osdi_path=osdi,
            model_file=lib,
            lib_section="tt",
        ),
    ])
    circuit = Circuit("selection")

    selection = manifest.resolve(family="mos")
    selection.apply_to_circuit(circuit)

    assert selection.osdi_paths == [str(osdi)]
    assert selection.includes == []
    assert selection.lib_sections == [(str(lib), "tt")]
    assert circuit.directives[0].name == "lib"
    assert circuit.directives[0].args == (str(lib), "tt")
    assert selection.metadata["models"][0]["name"] == "nmos_tt"
