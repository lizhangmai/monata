import json
from pathlib import Path
import subprocess

import pytest

from monata.models.compiler import CompilationError, ModelCompiler
from monata.models.diagnostics import ModelDiagnosticError
from monata.models.flow import SimulationModelConfig
from monata.models.resolver import resolve_model_flow
from monata.runtime.capabilities import CapabilityState, ngspice_profile
from support.model_cases import _flow_test_techlib, _write_fake_openvaf
pytestmark = pytest.mark.slow

def test_model_compiler_compile_osdi_with_fake_openvaf(tmp_path):
    openvaf = tmp_path / "openvaf"
    openvaf.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = '-o' ]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "[ -n \"$out\" ] || exit 2\n"
        "printf 'compiled' > \"$out\"\n"
    )
    openvaf.chmod(0o755)
    va = tmp_path / "resistor.va"
    va.write_text("module resistor; endmodule\n")

    osdi = ModelCompiler(openvaf_bin=openvaf).compile_osdi(va, output_dir=tmp_path / "out")

    assert osdi == tmp_path / "out" / "resistor.osdi"
    assert osdi.read_text() == "compiled"


def test_model_compiler_failure_raises_compilation_error(tmp_path):
    openvaf = tmp_path / "openvaf"
    openvaf.write_text("#!/bin/sh\nprintf 'bad model' >&2\nexit 3\n")
    openvaf.chmod(0o755)
    va = tmp_path / "bad.va"
    va.write_text("bad\n")

    with pytest.raises(CompilationError, match="bad model"):
        ModelCompiler(openvaf_bin=openvaf).compile_osdi(va)


def test_model_compiler_timeout_raises_diagnostic(tmp_path, monkeypatch):
    openvaf = tmp_path / "openvaf"
    openvaf.write_text("#!/bin/sh\nexit 0\n")
    openvaf.chmod(0o755)
    va = tmp_path / "slow.va"
    va.write_text("module slow; endmodule\n")

    def timeout_run(cmd, **kwargs):
        assert kwargs["timeout"] == 0.5
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], output="partial", stderr="late")

    monkeypatch.setattr("monata.models.compiler.subprocess.run", timeout_run)

    with pytest.raises(ModelDiagnosticError) as excinfo:
        ModelCompiler(openvaf_bin=openvaf, timeout=0.5).compile_osdi(va)

    assert excinfo.value.diagnostic.code == "compiler_timeout"
    assert excinfo.value.diagnostic.context["backend"] == "openvaf"
    assert excinfo.value.diagnostic.context["timeout_seconds"] == 0.5
    assert excinfo.value.diagnostic.context["stdout"] == "partial"
    assert excinfo.value.diagnostic.context["stderr"] == "late"


def test_model_compiler_missing_openvaf_raises_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    va = tmp_path / "diode.va"
    va.write_text("module diode; endmodule\n")

    with pytest.raises(ModelDiagnosticError) as excinfo:
        ModelCompiler().compile_osdi(va)

    assert excinfo.value.diagnostic.code == "compiler_missing"
    assert excinfo.value.diagnostic.context["backend"] == "openvaf"


def test_model_compiler_honors_openvaf_bin_environment(tmp_path, monkeypatch):
    openvaf = _write_fake_openvaf(tmp_path / "openvaf-env", "compiled-env")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("OPENVAF_BIN", str(openvaf))
    va = tmp_path / "diode.va"
    va.write_text("module diode; endmodule\n")

    osdi = ModelCompiler().compile_osdi(va, output_dir=tmp_path / "out")

    assert osdi.read_text() == "compiled-env"


def test_model_resolver_rejects_packaged_osdi_when_precompiled_artifacts_are_disabled(tmp_path):
    techlib = _flow_test_techlib(tmp_path)
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(
                simulator_profile=profile,
                policy="osdi-first",
            ),
        )

    assert excinfo.value.diagnostic.code == "precompiled_package_artifact_disallowed"


def test_model_resolver_rejects_unsupported_compiler_requirement(tmp_path):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, osdi_requires={"compiler": "spectre"})
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(simulator_profile=profile, policy="osdi-first"),
        )

    assert excinfo.value.diagnostic.code == "compiler_unsupported"
    assert excinfo.value.diagnostic.context["required_capabilities"] == {"compiler": "spectre"}


def test_model_resolver_reports_missing_required_openvaf_before_compile(tmp_path, monkeypatch):
    techlib = _flow_test_techlib(
        tmp_path,
        packaged_osdi=False,
        package_policy="user_provided_source",
        osdi_requires={"compiler": "openvaf"},
    )
    source = tmp_path / "bsimcmg.va"
    source.write_text("module bsimcmg; endmodule\n")
    monkeypatch.setenv("PATH", "")
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)

    with pytest.raises(ModelDiagnosticError) as excinfo:
        resolve_model_flow(
            techlib,
            "tt",
            simulator_profile=profile,
            model_config=SimulationModelConfig(
                simulator_profile=profile,
                policy="osdi-first",
                allow_external_osdi=False,
                source_paths={"bsimcmg": str(source)},
            ),
        )

    assert excinfo.value.diagnostic.code == "compiler_missing"
    assert excinfo.value.diagnostic.context["backend"] == "openvaf"


def test_model_resolver_compiles_bundled_source_with_declared_includes_and_args(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources" / "bsimcmg"
    source_dir.mkdir(parents=True)
    source = source_dir / "bsimcmg.va"
    include = source_dir / "defs.include"
    source.write_text('`include "defs.include"\nmodule bsimcmg; endmodule\n')
    include.write_text("parameter real tox = 1e-9;\n")
    techlib = _flow_test_techlib(
        tmp_path,
        packaged_osdi=False,
        package_policy="bundled_source",
        source_va="sources/bsimcmg/bsimcmg.va",
        source_includes=("defs.include",),
        compiler_args=("-D__NGSPICE__",),
    )
    _write_fake_openvaf(tmp_path / "openvaf", "compiled")
    args_log = tmp_path / "openvaf.args"
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("MONATA_OPENVAF_ARGS", str(args_log))
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)
    config = SimulationModelConfig(
        simulator_profile=profile,
        policy="osdi-first",
        allow_external_osdi=False,
        cache_dir=str(tmp_path / "cache"),
    )

    first = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)
    second = resolve_model_flow(techlib, "tt", simulator_profile=profile, model_config=config)

    osdi = Path(first.model_selection.osdi_paths[0])
    meta = json.loads((osdi.parent / "meta.json").read_text())
    args = args_log.read_text().splitlines()
    assert first.cache_hits == {"osdi": False}
    assert second.cache_hits == {"osdi": True}
    assert osdi.read_text() == "compiled"
    assert meta["includes"] == [str(include.resolve())]
    assert str(include.resolve()) in meta["inputs"]
    assert meta["context"]["compiler_args"] == ["-D__NGSPICE__"]
    assert "-D__NGSPICE__" in args
    assert "-I" in args
    assert str(source_dir.resolve()) in args


def test_model_resolver_recompiles_when_openvaf_identity_changes(tmp_path):
    techlib = _flow_test_techlib(tmp_path, packaged_osdi=False, package_policy="user_provided_source")
    source = tmp_path / "bsimcmg.va"
    source.write_text("module bsimcmg; endmodule\n")
    first_openvaf = _write_fake_openvaf(tmp_path / "openvaf-a", "compiled-a")
    second_openvaf = _write_fake_openvaf(tmp_path / "openvaf-b", "compiled-b")
    profile = ngspice_profile(osdi=CapabilityState.SUPPORTED)

    first = resolve_model_flow(
        techlib,
        "tt",
        simulator_profile=profile,
        model_config=SimulationModelConfig(
            simulator_profile=profile,
            policy="osdi-first",
            allow_external_osdi=False,
            cache_dir=str(tmp_path / "cache"),
            openvaf_bin=str(first_openvaf),
            source_paths={"bsimcmg": str(source)},
        ),
    )
    second = resolve_model_flow(
        techlib,
        "tt",
        simulator_profile=profile,
        model_config=SimulationModelConfig(
            simulator_profile=profile,
            policy="osdi-first",
            allow_external_osdi=False,
            cache_dir=str(tmp_path / "cache"),
            openvaf_bin=str(second_openvaf),
            source_paths={"bsimcmg": str(source)},
        ),
    )

    assert first.cache_hits == {"osdi": False}
    assert second.cache_hits == {"osdi": False}
    assert Path(first.model_selection.osdi_paths[0]).read_text() == "compiled-a"
    assert Path(second.model_selection.osdi_paths[0]).read_text() == "compiled-b"
