import json
from pathlib import Path

import pytest

from monata.models.cache import CACHE_MARKER_FILENAME, ModelCache
from monata.models.cache import reset_default_cache_dir_for_tests, resolve_model_cache_dir
from monata.models.compiler import ModelCompiler
from monata.models.diagnostics import ModelDiagnosticError
from monata.models.flow import SimulationModelConfig
from monata.models.resolver import resolve_model_flow
from monata.models.registry import ModelRegistry
from monata.runtime.capabilities import CapabilityState, ngspice_profile
from support.model_cases import _flow_test_techlib, _write_fake_openvaf
pytestmark = pytest.mark.slow

def test_model_cache_lookup_store_and_metadata(tmp_path):
    va = tmp_path / "diode.va"
    va.write_text("module diode; endmodule\n")
    built = tmp_path / "diode.osdi"
    built.write_text("compiled")
    cache = ModelCache(tmp_path / "cache")

    assert cache.lookup(va) is None
    cached = cache.store(va, built)

    assert cached.exists()
    assert cached.read_text() == "compiled"
    assert cache.lookup(va) == cached
    assert (cached.parent / "meta.json").exists()

    va.write_text("module diode2; endmodule\n")
    assert cache.lookup(va) is None


def test_model_cache_resolution_honors_monata_home_and_specific_override(tmp_path, monkeypatch):
    monkeypatch.delenv("MONATA_MODEL_CACHE", raising=False)
    monkeypatch.setenv("MONATA_HOME", str(tmp_path / "home"))
    reset_default_cache_dir_for_tests()

    assert resolve_model_cache_dir() == tmp_path / "home" / "cache" / "models"

    monkeypatch.setenv("MONATA_MODEL_CACHE", str(tmp_path / "specific"))
    reset_default_cache_dir_for_tests()

    assert resolve_model_cache_dir(project_config=tmp_path / "project-cache") == tmp_path / "specific"

    monkeypatch.delenv("MONATA_MODEL_CACHE", raising=False)
    reset_default_cache_dir_for_tests()

    assert resolve_model_cache_dir(project_config=tmp_path / "project-cache") == tmp_path / "project-cache"


def test_model_cache_default_stays_under_default_monata_home(tmp_path, monkeypatch):
    monkeypatch.delenv("MONATA_HOME", raising=False)
    monkeypatch.delenv("MONATA_MODEL_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reset_default_cache_dir_for_tests()

    assert resolve_model_cache_dir() == tmp_path / "home" / ".monata/cache/models"


def test_model_cache_marks_new_cache_directory(tmp_path):
    cache = ModelCache(tmp_path / "cache")

    assert (cache.path / CACHE_MARKER_FILENAME).read_text() == "monata model cache\n"


def test_model_cache_clear_requires_marker_and_preserves_cache_directory(tmp_path):
    va = tmp_path / "diode.va"
    built = tmp_path / "diode.osdi"
    va.write_text("module diode; endmodule\n")
    built.write_text("compiled")
    cache = ModelCache(tmp_path / "cache")
    cached = cache.store(va, built)

    cache.clear()

    assert cache.path.is_dir()
    assert (cache.path / CACHE_MARKER_FILENAME).is_file()
    assert not cached.exists()


def test_model_cache_clear_refuses_unmarked_existing_directory(tmp_path):
    cache_dir = tmp_path / "existing"
    cache_dir.mkdir()
    protected = cache_dir / "user-data.txt"
    protected.write_text("do not delete")
    cache = ModelCache(cache_dir)

    with pytest.raises(ModelDiagnosticError) as excinfo:
        cache.clear()

    assert excinfo.value.diagnostic.code == "unsafe_cache_clear"
    assert excinfo.value.diagnostic.context["required_marker"] == CACHE_MARKER_FILENAME
    assert protected.read_text() == "do not delete"


def test_model_cache_clear_refuses_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = ModelCache(tmp_path)

    with pytest.raises(ModelDiagnosticError) as excinfo:
        cache.clear()

    assert excinfo.value.diagnostic.code == "unsafe_cache_clear"
    assert tmp_path.is_dir()


def test_model_cache_include_content_changes_cache_key(tmp_path):
    va = tmp_path / "diode.va"
    include = tmp_path / "constants.va"
    built = tmp_path / "diode.osdi"
    va.write_text("`include \"constants.va\"\nmodule diode; endmodule\n")
    include.write_text("parameter real is = 1e-12;\n")
    built.write_text("compiled")
    cache = ModelCache(tmp_path / "cache")

    cached = cache.store(va, built, include_paths=[include])

    assert cache.lookup(va, include_paths=[include]) == cached
    include.write_text("parameter real is = 2e-12;\n")
    assert cache.lookup(va, include_paths=[include]) is None


def test_model_cache_include_path_identity_changes_cache_key(tmp_path):
    va = tmp_path / "diode.va"
    include_a = tmp_path / "a" / "constants.va"
    include_b = tmp_path / "b" / "constants.va"
    built = tmp_path / "diode.osdi"
    include_a.parent.mkdir()
    include_b.parent.mkdir()
    va.write_text("`include \"constants.va\"\nmodule diode; endmodule\n")
    include_a.write_text("parameter real is = 1e-12;\n")
    include_b.write_text(include_a.read_text())
    built.write_text("compiled")
    cache = ModelCache(tmp_path / "cache")

    cache.store(va, built, include_paths=[include_a])

    assert cache.lookup(va, include_paths=[include_b]) is None


def test_model_cache_context_identity_changes_cache_key(tmp_path):
    va = tmp_path / "diode.va"
    built = tmp_path / "diode.osdi"
    va.write_text("module diode; endmodule\n")
    built.write_text("compiled")
    cache = ModelCache(tmp_path / "cache")

    cached = cache.store(va, built, context={"flow": "a", "compiler": {"sha": "1"}})

    assert cache.lookup(va, context={"flow": "a", "compiler": {"sha": "1"}}) == cached
    assert cache.lookup(va, context={"flow": "a", "compiler": {"sha": "2"}}) is None
    assert cache.lookup_compatible(va, required_context={"flow": "a"}) == cached


def test_model_cache_ignores_partial_entries_without_metadata(tmp_path):
    va = tmp_path / "diode.va"
    va.write_text("module diode; endmodule\n")
    cache = ModelCache(tmp_path / "cache")
    source_hash = cache._hash_files([va.resolve()])
    entry = cache._entry_dir(source_hash)
    entry.mkdir(parents=True)
    (entry / "diode.osdi").write_text("partial")

    assert cache.lookup(va) is None


def test_model_cache_framed_inputs_avoid_concatenation_collision(tmp_path):
    va = tmp_path / "diode.va"
    include_ab = tmp_path / "ab.inc"
    include_a = tmp_path / "a.inc"
    include_b = tmp_path / "b.inc"
    built = tmp_path / "diode.osdi"
    va.write_text("module diode; endmodule\n")
    include_ab.write_text("ab")
    include_a.write_text("a")
    include_b.write_text("b")
    built.write_text("compiled")
    cache = ModelCache(tmp_path / "cache")

    cache.store(va, built, include_paths=[include_ab])

    assert cache.lookup(va, include_paths=[include_a, include_b]) is None


def test_model_cache_missing_include_raises_diagnostic(tmp_path):
    va = tmp_path / "diode.va"
    va.write_text("`include \"missing.va\"\nmodule diode; endmodule\n")
    cache = ModelCache(tmp_path / "cache")

    with pytest.raises(ModelDiagnosticError) as excinfo:
        cache.lookup(va, include_paths=[tmp_path / "missing.va"])

    assert excinfo.value.diagnostic.code == "model_source_missing"
    assert excinfo.value.diagnostic.context["missing"] == [str(tmp_path / "missing.va")]


def test_model_cache_require_cached_reports_stale_artifact(tmp_path):
    va = tmp_path / "diode.va"
    va.write_text("module diode; endmodule\n")
    cache = ModelCache(tmp_path / "cache")

    with pytest.raises(ModelDiagnosticError) as excinfo:
        cache.require_cached(va)

    assert excinfo.value.diagnostic.code == "model_cache_missing"
    assert excinfo.value.diagnostic.context["source"] == str(va)


def test_register_va_registers_cached_osdi_path(tmp_path):
    openvaf = tmp_path / "openvaf"
    openvaf.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = '-o' ]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "printf 'compiled' > \"$out\"\n"
    )
    openvaf.chmod(0o755)
    va = tmp_path / "bsim4.va"
    va.write_text("module bsim4; endmodule\n")
    registry = ModelRegistry(auto_discover=False)
    registry._compiler = ModelCompiler(openvaf_bin=openvaf)
    registry._cache = ModelCache(tmp_path / "cache")

    osdi = registry.register_va(va, family="mos", module_name="bsim4")

    assert osdi.exists()
    assert tmp_path / "cache" in osdi.parents
    assert registry.resolve_osdi("mos") == str(osdi)


def test_register_va_forwards_include_paths_to_cache_metadata(tmp_path, monkeypatch):
    openvaf = tmp_path / "openvaf"
    openvaf.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$MONATA_OPENVAF_ARGS\"\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = '-o' ]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "printf 'compiled' > \"$out\"\n"
    )
    openvaf.chmod(0o755)
    va = tmp_path / "bsim4.va"
    include = tmp_path / "constants.va"
    va.write_text("`include \"constants.va\"\nmodule bsim4; endmodule\n")
    include.write_text("parameter real tox = 1e-9;\n")
    registry = ModelRegistry(auto_discover=False)
    registry._compiler = ModelCompiler(openvaf_bin=openvaf)
    registry._cache = ModelCache(tmp_path / "cache")
    args_log = tmp_path / "openvaf.args"
    monkeypatch.setenv("MONATA_OPENVAF_ARGS", str(args_log))

    osdi = registry.register_va(
        va,
        family="mos",
        module_name="bsim4",
        include_paths=[include],
    )

    meta = json.loads((osdi.parent / "meta.json").read_text())
    entry = registry.resolve("mos")
    assert entry is not None
    assert meta["includes"] == [str(include.resolve())]
    assert str(include.resolve()) in meta["inputs"]
    assert entry.include_paths == [str(include.resolve())]
    args = args_log.read_text().splitlines()
    assert "-I" in args
    assert str(include.parent.resolve()) in args


def test_model_resolver_compiles_user_source_through_cache(tmp_path, monkeypatch):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    source = tmp_path / "bsimcmg.va"
    source.write_text("module bsimcmg; endmodule\n")
    _write_fake_openvaf(tmp_path / "openvaf", "compiled")
    monkeypatch.setenv("PATH", str(tmp_path))
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)
    config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        allow_external_osdi=False,
        cache_dir=str(tmp_path / "cache"),
        source_paths={"bsimcmg": str(source)},
    )

    first = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)
    second = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)

    assert first.cache_hits == {"osdi": False}
    assert second.cache_hits == {"osdi": True}
    assert first.model_selection.osdi_paths[0].endswith("bsimcmg.osdi")
    assert Path(first.model_selection.osdi_paths[0]).is_file()


def test_model_resolver_uses_compatible_cache_when_compile_is_disabled(tmp_path, monkeypatch):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    source = tmp_path / "bsimcmg.va"
    source.write_text("module bsimcmg; endmodule\n")
    _write_fake_openvaf(tmp_path / "openvaf", "compiled")
    monkeypatch.setenv("PATH", str(tmp_path))
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)
    base_config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        allow_external_osdi=False,
        cache_dir=str(tmp_path / "cache"),
        source_paths={"bsimcmg": str(source)},
    )

    first = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=base_config)
    monkeypatch.setenv("PATH", "")
    cached_config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        allow_external_osdi=False,
        allow_compile=False,
        cache_dir=str(tmp_path / "cache"),
        source_paths={"bsimcmg": str(source)},
    )
    second = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=cached_config)

    assert first.cache_hits == {"osdi": False}
    assert second.cache_hits == {"osdi": True}
    assert second.model_selection.osdi_paths == first.model_selection.osdi_paths


def test_model_resolver_rejects_compatible_cache_from_other_platform(tmp_path, monkeypatch):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    source = tmp_path / "bsimcmg.va"
    source.write_text("module bsimcmg; endmodule\n")
    built = tmp_path / "bsimcmg.osdi"
    built.write_text("compiled-for-other-platform")
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)
    cache = ModelCache(tmp_path / "cache", namespace="FAKE_PTM/osdi-openvaf")
    cache.store(
        source,
        built,
        context={
            "schema": "monata-osdi-cache-v1",
            "techlib": "FAKE_PTM",
            "flow": "osdi-openvaf",
            "output": "ngspice_osdi",
            "module_name": "bsimcmg",
            "simulator_profile": profile.to_dict(),
            "target_platform": {"sys_platform": "darwin", "machine": "arm64"},
            "wrapper_schema": "monata-osdi-wrapper-v1",
            "compiler": {"backend": "openvaf", "available": True},
        },
    )
    monkeypatch.setenv("PATH", "")

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(
                simulator_profile=profile,
                policy="osdi-first",
                allow_external_osdi=False,
                allow_compile=False,
                cache_dir=str(tmp_path / "cache"),
                source_paths={"bsimcmg": str(source)},
            ),
        )

    assert excinfo.value.diagnostic.code == "model_cache_missing"
