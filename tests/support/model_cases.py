import json
import platform
from pathlib import Path
import sys


from monata.models.artifacts import artifact_sha256
from monata.models.flow import ModelFlowRecipe
from monata.corner import OperatingCorner
from monata.techlib.registry import Techlib
from monata.techlib.schema import ModelDeck

def _write_osdi_sidecar(path: Path, overrides: dict | None = None) -> None:
    payload = {
        "schema_version": 1,
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha256(path),
        "source_sha256": "source",
        "compiler": "openvaf",
        "compiler_sha256": "compiler",
        "compiler_version": "test",
        "target_platform": {"sys_platform": sys.platform, "machine": platform.machine()},
        "osdi_api": "0.4",
        "simulator_profiles": {"ngspice-subprocess": {"dialect": "ngspice"}},
        "created_by": "unit-test",
        "validation_identity": "probe:test",
    }
    payload.update(overrides or {})
    path.with_name(path.name + ".monata-osdi.json").write_text(json.dumps(payload))


def _write_fake_openvaf(path: Path, payload: str) -> Path:
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '--version' ]; then printf 'fake-openvaf\\n'; exit 0; fi\n"
        "if [ -n \"$MONATA_OPENVAF_ARGS\" ]; then printf '%s\\n' \"$@\" > \"$MONATA_OPENVAF_ARGS\"; fi\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = '-o' ]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "[ -n \"$out\" ] || exit 2\n"
        f"printf '{payload}' > \"$out\"\n"
    )
    path.chmod(0o755)
    return path


def _flow_test_techlib(
    tmp_path,
    *,
    packaged_osdi: bool = True,
    package_policy: str | None = None,
    extra_requires: dict | None = None,
    osdi_requires: dict | None = None,
    source_va: str | None = None,
    source_includes: tuple[str, ...] = (),
    compiler_args: tuple[str, ...] = (),
) -> Techlib:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    raw = model_dir / "raw.mod"
    converted = model_dir / "converted.mod"
    osdi = model_dir / "bsimcmg.osdi"
    raw.write_text(".LIB tt\n.model n nmos level=72\n.ENDL tt\n")
    converted.write_text(".model n bsimcmg\n")
    osdi.write_text("compiled")
    return Techlib(
        name="FAKE_PTM",
        root=tmp_path,
        model_decks=[
            ModelDeck(name="raw", path="models/raw.mod"),
        ],
        model_flows=[
            ModelFlowRecipe(
                name="raw-level72",
                model_deck="raw",
                output="native_spice_lib",
                requires={"native_spice_model_levels": [72], **(extra_requires or {})},
            ),
            ModelFlowRecipe(
                name="osdi-openvaf",
                model_deck="raw",
                output="ngspice_osdi",
                requires={"osdi": True, **(osdi_requires or {})},
                dialects=("ngspice",),
                module_name="bsimcmg",
                source_name="bsimcmg",
                source_va=source_va,
                source_includes=source_includes,
                compiler_args=compiler_args,
                converted_model_card="models/converted.mod",
                osdi_path="models/bsimcmg.osdi" if packaged_osdi else None,
                package_policy=package_policy,
            ),
        ],
        corners=[OperatingCorner(name="tt", model_deck="raw", section="tt")],
    )
