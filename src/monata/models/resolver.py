"""Simulator-aware model-flow resolver."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

from monata._paths import resolve_relative_path
from monata.models.artifacts import ModelArtifact, artifact_sha256
from monata.models.cache import ModelCache, resolve_model_cache_dir
from monata.models.compiler import ModelCompiler
from monata.models.diagnostics import ModelDiagnostic, ModelDiagnosticError
from monata.models.flow import (
    ModelFlowError,
    ModelFlowRecipe,
    ResolvedModelFlow,
    SimulationModelConfig,
    path_from_recipe,
)
from monata.models.manifest import ModelSelection
from monata.models.registry import ModelEntry
from monata.runtime.capabilities import CapabilityState, SimulatorProfile


@dataclass(frozen=True)
class _Candidate:
    recipe: ModelFlowRecipe
    selection: ModelSelection
    artifacts: tuple[ModelArtifact, ...]
    generated_artifacts: tuple[ModelArtifact, ...]
    cache_hits: dict[str, bool]
    warnings: tuple[ModelDiagnostic, ...]


class ModelResolver:
    """Resolve a technology/corner to simulator-ready model artifacts."""

    def __init__(self, techlib, *, model_config: SimulationModelConfig | None = None):
        self.techlib = techlib
        self.config = model_config or SimulationModelConfig()

    def resolve(self, corner=None, *, simulator_profile: SimulatorProfile | None = None) -> ResolvedModelFlow:
        profile = simulator_profile or self.config.simulator_profile
        if profile is None:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="simulator_profile_required",
                    message="model-flow resolution requires an explicit simulator profile",
                    context={
                        "techlib": getattr(self.techlib, "name", None),
                        "corner": getattr(corner, "name", corner),
                    },
                )
            )
        recipes = tuple(getattr(self.techlib, "model_flows", ()))
        if not recipes:
            raise ModelDiagnosticError(
                _diagnostic(
                    "model_flow_missing",
                    "no model flow recipe is available",
                    self.techlib,
                    corner,
                    profile,
                )
            )

        diagnostics: list[ModelDiagnostic] = []
        candidates: list[_Candidate] = []
        for recipe in _ordered_recipes(recipes, self.config):
            if self.config.pinned_flow and recipe.name != self.config.pinned_flow:
                continue
            try:
                candidates.append(self._candidate(recipe, corner, profile))
            except ModelDiagnosticError as exc:
                diagnostics.append(exc.diagnostic)

        if candidates:
            selected = candidates[0]
            flow = ResolvedModelFlow(
                flow_name=selected.recipe.name,
                policy=self.config.policy,
                simulator_profile=profile,
                model_selection=selected.selection,
                artifacts=selected.artifacts,
                generated_artifacts=selected.generated_artifacts,
                cache_hits=selected.cache_hits,
                warnings=selected.warnings,
                diagnostics=tuple(diagnostics),
            )
            return flow.with_reuse_signature()

        if not diagnostics:
            diagnostics.append(_diagnostic("model_flow_missing", "no model flow recipe is available", self.techlib, corner, profile))
        raise ModelDiagnosticError(diagnostics[0])

    def _candidate(self, recipe: ModelFlowRecipe, corner, profile: SimulatorProfile) -> _Candidate:
        resolved_corner = self.techlib.corner(corner)
        if not recipe.enabled:
            raise ModelDiagnosticError(
                _diagnostic(
                    recipe.disabled_diagnostic or "model_syntax_unverified",
                    "model flow is declared but not enabled",
                    self.techlib,
                    resolved_corner,
                    profile,
                    attempted_flow=recipe.name,
                    next_action=recipe.next_action,
                )
            )
        _validate_capabilities(self.techlib, resolved_corner, recipe, profile)
        if recipe.is_osdi_flow:
            return self._osdi_candidate(recipe, resolved_corner, profile)
        return self._native_candidate(recipe, resolved_corner, profile)

    def _native_candidate(self, recipe: ModelFlowRecipe, corner, profile: SimulatorProfile) -> _Candidate:
        deck = self.techlib.model_deck(recipe.model_deck)
        deck_path = deck.resolve_path(self.techlib.root)
        selection = ModelSelection([
            ModelEntry(
                name=corner.name,
                family=self.techlib.name,
                module_name=deck.name,
                model_file=deck_path,
                lib_section=corner.section,
                provenance=self.techlib.provenance,
            )
        ]).validate_files()
        artifact = ModelArtifact(
            kind="spice_lib",
            path=str(deck_path),
            role="model_card",
            requires=dict(recipe.requires),
            provenance=self.techlib.provenance,
            content_hash=artifact_sha256(deck_path),
            package_policy=recipe.package_policy or "bundled_source",
        )
        return _Candidate(recipe, selection, (artifact,), (), {}, ())

    def _osdi_candidate(self, recipe: ModelFlowRecipe, corner, profile: SimulatorProfile) -> _Candidate:
        source_path = self._source_path(recipe)
        osdi_path = path_from_recipe(self.techlib.root, recipe.osdi_path)
        osdi_origin = "package" if osdi_path is not None else None
        external_validation: dict[str, Any] | None = None
        cache_hits: dict[str, bool] = {}
        generated_artifacts: tuple[ModelArtifact, ...] = ()
        if osdi_path is not None and not self.config.allow_precompiled_package_artifacts:
            raise ModelDiagnosticError(
                _diagnostic(
                    "precompiled_package_artifact_disallowed",
                    "package-provided precompiled OSDI artifacts are disabled by policy",
                    self.techlib,
                    corner,
                    profile,
                    attempted_flow=recipe.name,
                    next_action="Enable allow_precompiled_package_artifacts only for trusted package artifacts.",
                )
            )
        if osdi_path is None:
            if _external_osdi_allowed(self.config):
                osdi_path, external_validation = _find_external_osdi(recipe, self.config, profile)
                if osdi_path is not None:
                    osdi_origin = "external"
        if osdi_path is None:
            if source_path:
                osdi_path, cache_hits = self._cached_or_compiled_osdi(recipe, profile, Path(source_path))
                osdi_origin = "compiled" if not cache_hits.get("osdi") else "cache"
            else:
                if recipe.package_policy == "user_provided_source" and self.config.policy != "bundled-only":
                    raise ModelDiagnosticError(
                        _diagnostic(
                            "model_source_missing",
                            "model flow requires a user-provided Verilog-A source path or trusted external OSDI",
                            self.techlib,
                            corner,
                            profile,
                            attempted_flow=recipe.name,
                            env_vars=["MONATA_BSIMCMG_SOURCE", "MONATA_OSDI_PATH"],
                            next_action="Set MONATA_BSIMCMG_SOURCE, project model source config, or a trusted MONATA_OSDI_PATH artifact for BSIM-CMG.",
                        )
                    )
                code = "model_flow_requires_compile" if not self.config.allow_compile else "model_cache_missing"
                raise ModelDiagnosticError(
                    _diagnostic(
                        code,
                        "model flow requires OSDI compilation or a compatible cache hit",
                        self.techlib,
                        corner,
                        profile,
                        attempted_flow=recipe.name,
                        env_vars=["OPENVAF_BIN", "MONATA_MODEL_CACHE"],
                    )
                )
        model_card = self._converted_model_card(recipe, corner)
        if model_card is None:
            raise ModelDiagnosticError(
                _diagnostic(
                    "converted_model_card_failed",
                    "OSDI flow has no converted simulator-ready model card",
                    self.techlib,
                    corner,
                    profile,
                    attempted_flow=recipe.name,
                )
            )
        selection = ModelSelection([
            ModelEntry(
                name=corner.name,
                family=self.techlib.name,
                module_name=recipe.module_name or recipe.name,
                osdi_path=osdi_path,
                model_file=model_card,
                provenance=self.techlib.provenance,
            )
        ]).validate_files()
        osdi_artifact = ModelArtifact(
            kind="osdi",
            path=str(osdi_path),
            role="compiled",
            requires=dict(recipe.requires),
            provenance=self.techlib.provenance,
            content_hash=artifact_sha256(osdi_path),
            package_policy=_osdi_package_policy(recipe, osdi_origin),
            validation=external_validation or dict(recipe.validation),
        )
        card_artifact = ModelArtifact(
            kind="converted_model_card",
            path=str(model_card),
            role="generated",
            requires=dict(recipe.requires),
            provenance=self.techlib.provenance,
            content_hash=artifact_sha256(model_card),
            generated_from=[str(self.techlib.model_deck(recipe.model_deck).resolve_path(self.techlib.root))],
            toolchain={"converter": recipe.converter} if recipe.converter else {},
            package_policy="generated_cache_artifact",
        )
        if osdi_origin == "compiled":
            generated_artifacts = (osdi_artifact, card_artifact)
        else:
            generated_artifacts = (card_artifact,)
        return _Candidate(recipe, selection, (osdi_artifact, card_artifact), generated_artifacts, cache_hits, ())

    def _converted_model_card(self, recipe: ModelFlowRecipe, corner) -> Path | None:
        model_card = path_from_recipe(self.techlib.root, recipe.converted_model_card)
        if model_card is not None:
            return model_card
        if recipe.converter == "ptm_mg_level72_to_bsimcmg":
            return _generate_ptm_mg_bsimcmg_model_card(self.techlib, recipe, self.config, corner)
        return None

    def _source_path(self, recipe: ModelFlowRecipe) -> str | None:
        allow_user_source = self.config.policy != "bundled-only"
        if allow_user_source and recipe.source_name and recipe.source_name in self.config.source_paths:
            return self.config.source_paths[recipe.source_name]
        if allow_user_source and recipe.source_name:
            env_path = os.environ.get(f"MONATA_{recipe.source_name.upper()}_SOURCE")
            if env_path:
                return env_path
        if recipe.source_va:
            source = path_from_recipe(self.techlib.root, recipe.source_va)
            if source and source.exists():
                return str(source)
        return None

    def _cached_or_compiled_osdi(
        self,
        recipe: ModelFlowRecipe,
        profile: SimulatorProfile,
        source_path: Path,
    ) -> tuple[Path, dict[str, bool]]:
        include_paths = _source_include_paths(recipe, source_path)
        required_compiler = _compiler_requirement(recipe)
        if required_compiler and required_compiler != "openvaf":
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="compiler_unsupported",
                    message="model flow requires an unsupported compiler backend",
                    context={"required_compiler": required_compiler, "attempted_flow": recipe.name},
                )
            )
        cache = ModelCache(
            resolve_model_cache_dir(explicit=self.config.cache_dir),
            namespace=f"{getattr(self.techlib, 'name', 'techlib')}/{recipe.name}",
        )
        required_context = _cache_required_context(self.techlib, recipe, profile)
        if not self.config.allow_compile:
            cached = cache.lookup_compatible(
                source_path,
                include_paths=include_paths,
                required_context=required_context,
            )
            if cached is not None:
                return cached, {"osdi": True}
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="model_cache_missing",
                    message="compiled model artifact is missing or stale",
                    context={"source": str(source_path), "attempted_flow": recipe.name},
                )
            )
        compiler = ModelCompiler(openvaf_bin=self.config.openvaf_bin)
        store_context = _cache_store_context(self.techlib, recipe, profile, compiler)
        cached = cache.lookup(source_path, include_paths=include_paths, context=store_context)
        if cached is not None:
            return cached, {"osdi": True}
        compatible_cached = cache.lookup_compatible(
            source_path,
            include_paths=include_paths,
            required_context=required_context,
        )
        if compatible_cached is not None and not compiler.has_openvaf:
            return compatible_cached, {"osdi": True}
        if not compiler.has_openvaf:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="compiler_missing",
                    message="OpenVAF not found in PATH. Install openvaf-r or set openvaf_bin.",
                    context={"backend": "openvaf", "source": str(source_path), "attempted_flow": recipe.name},
                )
            )
        build_dir = cache.path / "_build" / recipe.name
        compiled = compiler.compile_osdi(
            source_path,
            output_dir=build_dir,
            extra_args=recipe.compiler_args,
            include_paths=include_paths,
        )
        return cache.store(
            source_path,
            compiled,
            include_paths=include_paths,
            context=store_context,
        ), {"osdi": False}


def resolve_model_flow(
    techlib,
    corner=None,
    *,
    simulator_profile: SimulatorProfile | None = None,
    model_config: SimulationModelConfig | None = None,
) -> ResolvedModelFlow:
    return ModelResolver(techlib, model_config=model_config).resolve(corner, simulator_profile=simulator_profile)


def _osdi_package_policy(recipe: ModelFlowRecipe, origin: str | None) -> str:
    if origin == "external":
        return "external_read_only_artifact"
    if origin in {"compiled", "cache"}:
        return "generated_cache_artifact"
    return recipe.package_policy or "precompiled_package_artifact"


def _ordered_recipes(recipes: tuple[ModelFlowRecipe, ...], config: SimulationModelConfig) -> tuple[ModelFlowRecipe, ...]:
    if not recipes:
        return ()
    if config.policy == "osdi-first":
        return tuple(sorted(recipes, key=lambda recipe: (not recipe.is_osdi_flow, recipe.name)))
    if config.policy in {"auto", "native-first"}:
        return tuple(sorted(recipes, key=lambda recipe: (not recipe.is_native_flow, recipe.name)))
    if config.policy == "source-compile":
        return tuple(sorted(recipes, key=lambda recipe: (not recipe.is_osdi_flow, recipe.name)))
    return recipes


def _external_osdi_allowed(config: SimulationModelConfig) -> bool:
    return config.allow_external_osdi and config.policy != "bundled-only"


def _cache_required_context(techlib, recipe: ModelFlowRecipe, profile: SimulatorProfile) -> dict[str, Any]:
    return {
        "schema": "monata-osdi-cache-v1",
        "techlib": getattr(techlib, "name", "techlib"),
        "flow": recipe.name,
        "output": recipe.output,
        "module_name": recipe.module_name or recipe.name,
        "compiler_args": list(recipe.compiler_args),
        "simulator_profile": profile.to_dict(),
        "target_platform": _target_platform_identity(),
        "wrapper_schema": "monata-osdi-wrapper-v1",
    }


def _cache_store_context(
    techlib,
    recipe: ModelFlowRecipe,
    profile: SimulatorProfile,
    compiler: ModelCompiler,
) -> dict[str, Any]:
    context = _cache_required_context(techlib, recipe, profile)
    context.update(
        {
            "compiler": compiler.openvaf_identity(),
            "osdi_api_versions": list(profile.capabilities.osdi_api_versions),
        }
    )
    return context


def _source_include_paths(recipe: ModelFlowRecipe, source_path: Path) -> tuple[Path, ...]:
    """Resolve recipe-declared source dependencies relative to the source file."""

    source_dir = source_path.resolve().parent
    includes: list[Path] = []
    seen: set[Path] = set()
    for value in recipe.source_includes:
        try:
            path = resolve_relative_path(
                source_dir,
                str(value),
                label="model flow source include path",
                root_label="source_va directory",
            )
        except ValueError as exc:
            raise ModelFlowError(str(exc)) from exc
        if path not in seen:
            seen.add(path)
            includes.append(path)
    return tuple(includes)


def _target_platform_identity() -> dict[str, str]:
    return {
        "sys_platform": sys.platform.lower(),
        "machine": platform.machine().lower(),
    }


def _generate_ptm_mg_bsimcmg_model_card(techlib, recipe: ModelFlowRecipe, config: SimulationModelConfig, corner) -> Path:
    deck = techlib.model_deck(recipe.model_deck).resolve_path(techlib.root)
    section = getattr(corner, "section", None) or getattr(corner, "name", None)
    if not section:
        raise ModelDiagnosticError(
            ModelDiagnostic(
                code="converted_model_card_failed",
                message="PTM_MG BSIM-CMG conversion requires a corner section",
                context={"attempted_flow": recipe.name},
            )
        )
    digest = artifact_sha256(deck)[:16]
    target_dir = (
        resolve_model_cache_dir(explicit=config.cache_dir)
        / "converted"
        / getattr(techlib, "name", "techlib")
        / recipe.name
        / digest
        / section
    )
    target = target_dir / "ptm_mg_bsimcmg_osdi.mod"

    target_dir.mkdir(parents=True, exist_ok=True)
    source_text = deck.read_text()
    converted_text = _extract_spice_lib_section(source_text, section, deck)
    for include_name in _ptm_mg_includes(source_text):
        source = deck.parent / include_name
        if _is_ptm_mg_device_model(include_name):
            converted = target_dir / include_name
            _write_converted_ptm_mg_device_model(source, converted)
            converted_text = _replace_quoted_path(converted_text, include_name, converted)
    converted_text = _inline_spice_lib_references(converted_text, deck.parent)
    converted_text = re.sub(r"(?m)^mnfet\b", "Nnfet", converted_text)
    converted_text = re.sub(r"(?m)^mpfet\b", "Npfet", converted_text)
    target.write_text(converted_text)
    return target


def _extract_spice_lib_section(text: str, section: str, source_path: Path) -> str:
    selected: list[str] | None = None
    for line in text.splitlines():
        tokens = _spice_directive_tokens(line)
        if selected is None:
            if len(tokens) == 2 and tokens[0].lower() == ".lib" and tokens[1].lower() == section.lower():
                selected = []
            continue
        if tokens and tokens[0].lower() == ".endl":
            if len(tokens) == 1 or tokens[1].lower() == section.lower():
                return "\n".join(selected).strip() + "\n"
        selected.append(line)
    raise ModelDiagnosticError(
        ModelDiagnostic(
            code="converted_model_card_failed",
            message="PTM_MG BSIM-CMG conversion could not find the requested .lib section",
            context={"section": section, "source": str(source_path)},
        )
    )


def _inline_spice_lib_references(text: str, base_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        path_text = match.group("path")
        section = match.group("section")
        path = Path(path_text)
        if not path.is_absolute():
            path = base_dir / path
        return _extract_spice_lib_section(path.read_text(), section, path).rstrip()

    return re.sub(
        r"(?im)^\s*\.lib\s+['\"](?P<path>[^'\"]+)['\"]\s+(?P<section>\S+)\s*$",
        replace,
        text,
    )


def _spice_directive_tokens(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("*"):
        return []
    try:
        return shlex.split(stripped, posix=True)
    except ValueError:
        return stripped.split()


def _ptm_mg_includes(text: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in re.finditer(r"\.include\s+['\"]([^'\"]+)['\"]", text, re.IGNORECASE))


def _is_ptm_mg_device_model(path: str) -> bool:
    return bool(re.match(r"ptm_mg_\d+nm_(?:hp|lstp)_[np]mos\.mod\Z", Path(path).name))


def _replace_quoted_path(text: str, original: str, replacement: Path) -> str:
    return text.replace(f"'{original}'", f"'{replacement.resolve()}'").replace(
        f'"{original}"',
        f'"{replacement.resolve()}"',
    )


def _write_converted_ptm_mg_device_model(source: Path, target: Path) -> None:
    text = source.read_text()
    text = re.sub(
        r"(?m)^\.model\s+nfet\s+nmos\s+level\s*=\s*72\s*$",
        ".model nfet bsimcmg_va\n+ TYPE = 1",
        text,
    )
    text = re.sub(
        r"(?m)^\.model\s+pfet\s+pmos\s+level\s*=\s*72\s*$",
        ".model pfet bsimcmg_va\n+ TYPE = 0",
        text,
    )
    text = re.sub(r"(?im)^\+version\s*=.*$", lambda match: f"* {match.group(0)[1:].strip()}", text)
    target.write_text(text)


def _compiler_requirement(recipe: ModelFlowRecipe) -> str | None:
    raw = recipe.requires.get("compiler")
    if raw in (None, False):
        return None
    return str(raw)


def _validate_compiler_requirement(techlib, corner, recipe: ModelFlowRecipe, profile: SimulatorProfile) -> None:
    required_compiler = _compiler_requirement(recipe)
    if required_compiler is None:
        return
    if required_compiler != "openvaf":
        raise ModelDiagnosticError(
            _diagnostic(
                "compiler_unsupported",
                "model flow requires an unsupported compiler backend",
                techlib,
                corner,
                profile,
                attempted_flow=recipe.name,
                required_capabilities={"compiler": required_compiler},
            )
        )


def _validate_capabilities(techlib, corner, recipe: ModelFlowRecipe, profile: SimulatorProfile) -> None:
    supported_requires = {
        "native_spice_model_levels",
        "osdi",
        "compiler",
        "spice_lib",
        "supports_subckt_wrappers",
    }
    unknown_requires = set(recipe.requires) - supported_requires
    if unknown_requires:
        raise ModelDiagnosticError(
            _diagnostic(
                "model_flow_unsupported_by_simulator",
                "model flow declares unsupported capability requirements",
                techlib,
                corner,
                profile,
                attempted_flow=recipe.name,
                required_capabilities={key: recipe.requires[key] for key in sorted(unknown_requires)},
            )
        )
    if recipe.dialects and profile.dialect not in recipe.dialects:
        raise ModelDiagnosticError(
            _diagnostic(
                "model_flow_unsupported_by_simulator",
                "model flow dialect constraints do not match the simulator profile",
                techlib,
                corner,
                profile,
                attempted_flow=recipe.name,
                required_capabilities={"dialects": list(recipe.dialects)},
                available_capabilities={"dialect": profile.dialect},
            )
        )
    _validate_compiler_requirement(techlib, corner, recipe, profile)
    levels = recipe.requires.get("native_spice_model_levels", ())
    if isinstance(levels, int):
        levels = (levels,)
    for level in levels:
        if not profile.capabilities.supports_native_level(int(level)):
            raise ModelDiagnosticError(
                _diagnostic(
                    "model_flow_unsupported_by_simulator",
                    "simulator profile does not support the required native SPICE model level",
                    techlib,
                    corner,
                    profile,
                    attempted_flow=recipe.name,
                    required_capabilities={"native_spice_model_levels": [int(level)]},
                    available_capabilities={
                        "native_spice_model_levels": sorted(profile.capabilities.native_spice_model_levels)
                    },
                )
            )
    if recipe.requires.get("osdi"):
        if profile.capabilities.osdi == CapabilityState.UNKNOWN:
            raise ModelDiagnosticError(
                _diagnostic(
                    "simulator_capability_unknown",
                    "simulator OSDI capability is unknown",
                    techlib,
                    corner,
                    profile,
                    attempted_flow=recipe.name,
                    required_capabilities={"osdi": "supported"},
                    available_capabilities={"osdi": "unknown"},
                    next_action="Run an OSDI probe or pass a simulator profile with OSDI support evidence.",
                )
            )
        if profile.capabilities.osdi != CapabilityState.SUPPORTED:
            raise ModelDiagnosticError(
                _diagnostic(
                    "model_flow_unsupported_by_simulator",
                    "simulator profile does not support OSDI artifacts",
                    techlib,
                    corner,
                    profile,
                    attempted_flow=recipe.name,
                    required_capabilities={"osdi": "supported"},
                    available_capabilities={"osdi": profile.capabilities.osdi.value},
                )
            )
    if recipe.requires.get("supports_subckt_wrappers"):
        state = profile.capabilities.supports_subckt_wrappers
        if state == CapabilityState.UNKNOWN:
            raise ModelDiagnosticError(
                _diagnostic(
                    "simulator_capability_unknown",
                    "simulator subckt-wrapper capability is unknown",
                    techlib,
                    corner,
                    profile,
                    attempted_flow=recipe.name,
                    required_capabilities={"supports_subckt_wrappers": "supported"},
                    available_capabilities={"supports_subckt_wrappers": "unknown"},
                )
            )
        if state != CapabilityState.SUPPORTED:
            raise ModelDiagnosticError(
                _diagnostic(
                    "model_flow_unsupported_by_simulator",
                    "simulator profile does not support generated subckt wrappers",
                    techlib,
                    corner,
                    profile,
                    attempted_flow=recipe.name,
                    required_capabilities={"supports_subckt_wrappers": "supported"},
                    available_capabilities={"supports_subckt_wrappers": state.value},
                )
            )


def _find_external_osdi(
    recipe: ModelFlowRecipe,
    config: SimulationModelConfig,
    profile: SimulatorProfile,
) -> tuple[Path | None, dict[str, Any] | None]:
    module_name = recipe.module_name or recipe.name
    for root in _external_osdi_search_paths(config):
        root_path = Path(root)
        candidates = [root_path / f"{module_name}.osdi"] if root_path.is_dir() else [root_path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            return candidate, _validate_external_osdi_sidecar(candidate, profile)
    return None, None


def _external_osdi_search_paths(config: SimulationModelConfig) -> tuple[str, ...]:
    env_paths = tuple(path for path in os.environ.get("MONATA_OSDI_PATH", "").split(os.pathsep) if path)
    return (*config.external_osdi_paths, *env_paths)


def _validate_external_osdi_sidecar(path: Path, profile: SimulatorProfile) -> dict[str, Any]:
    sidecars = (
        path.with_name(path.name + ".monata-osdi.json"),
        path.with_suffix(".monata-osdi.json"),
    )
    for sidecar in sidecars:
        if not sidecar.is_file():
            continue
        try:
            data = json.loads(sidecar.read_text())
        except json.JSONDecodeError as exc:
            raise _external_osdi_untrusted(path, profile, "sidecar is not valid JSON", metadata_path=sidecar) from exc
        required = {
            "schema_version",
            "artifact_path",
            "artifact_sha256",
            "source_sha256",
            "compiler",
            "compiler_sha256",
            "compiler_version",
            "target_platform",
            "osdi_api",
            "simulator_profiles",
            "created_by",
            "validation_identity",
        }
        missing = sorted(required - set(data))
        if missing:
            raise _external_osdi_untrusted(
                path,
                profile,
                "sidecar is missing required fields",
                metadata_path=sidecar,
                missing_fields=missing,
            )
        if data["schema_version"] != 1:
            raise _external_osdi_untrusted(
                path,
                profile,
                "sidecar schema version is unsupported",
                metadata_path=sidecar,
                schema_version=data["schema_version"],
            )
        artifact_path = _sidecar_artifact_path(sidecar, data["artifact_path"])
        if artifact_path.resolve() != path.resolve():
            raise _external_osdi_untrusted(
                path,
                profile,
                "sidecar artifact_path does not match the selected OSDI artifact",
                metadata_path=sidecar,
                artifact_path=str(artifact_path),
            )
        if data["artifact_sha256"] != artifact_sha256(path):
            raise _external_osdi_untrusted(
                path,
                profile,
                "sidecar artifact hash does not match the selected OSDI artifact",
                metadata_path=sidecar,
            )
        _require_nonempty_sidecar_fields(
            path,
            profile,
            sidecar,
            data,
            ("source_sha256", "compiler", "compiler_sha256", "compiler_version", "created_by", "validation_identity"),
        )
        _validate_sidecar_platform(path, profile, sidecar, data)
        _validate_sidecar_osdi_api(path, profile, sidecar, data)
        _validate_sidecar_simulator_profile(path, profile, sidecar, data)
        return {
            "metadata_path": str(sidecar),
            "validation_identity": data["validation_identity"],
            "schema_version": data["schema_version"],
            "source_sha256": data["source_sha256"],
            "compiler": data["compiler"],
            "compiler_sha256": data["compiler_sha256"],
            "compiler_version": data["compiler_version"],
            "target_platform": data["target_platform"],
            "osdi_api": data["osdi_api"],
        }
    probe_validation = _external_osdi_probe_validation(path, profile)
    if probe_validation is not None:
        return probe_validation
    raise _external_osdi_untrusted(path, profile, "external OSDI sidecar metadata is missing")


def _external_osdi_probe_validation(path: Path, profile: SimulatorProfile) -> dict[str, Any] | None:
    if profile.capabilities.osdi != CapabilityState.SUPPORTED:
        return None
    probes = (profile.probes, profile.capabilities.probes)
    for probe_set in probes:
        for key in ("external_osdi_validation", "osdi_artifact_probe"):
            raw = probe_set.get(key)
            if not isinstance(raw, dict):
                continue
            status = str(raw.get("status", raw.get("result", ""))).lower()
            if status not in {"passed", "ok"}:
                continue
            artifact_path = raw.get("artifact_path")
            if not artifact_path:
                continue
            if Path(str(artifact_path)).resolve() != path.resolve():
                continue
            return {
                "validation_identity": str(raw.get("validation_identity") or f"probe:{key}"),
                "probe": key,
                "metadata_path": None,
                "artifact_path": str(path),
            }
    return None


def _sidecar_artifact_path(sidecar: Path, raw_path: Any) -> Path:
    candidate = Path(str(raw_path))
    return candidate if candidate.is_absolute() else sidecar.parent / candidate


def _require_nonempty_sidecar_fields(
    path: Path,
    profile: SimulatorProfile,
    sidecar: Path,
    data: dict[str, Any],
    fields: tuple[str, ...],
) -> None:
    empty = [field for field in fields if not str(data.get(field, "")).strip()]
    if empty:
        raise _external_osdi_untrusted(
            path,
            profile,
            "sidecar has empty trust identity fields",
            metadata_path=sidecar,
            empty_fields=empty,
        )


def _validate_sidecar_platform(path: Path, profile: SimulatorProfile, sidecar: Path, data: dict[str, Any]) -> None:
    target = _normalize_platform_identity(data["target_platform"])
    current = _target_platform_identity()
    compatible = target == {"any": True} or target == current
    if not compatible:
        raise _osdi_artifact_incompatible(
            path,
            profile,
            "sidecar target platform is incompatible",
            metadata_path=sidecar,
            target_platform=data["target_platform"],
            current_platform=current,
        )


def _normalize_platform_identity(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        normalized = {
            "sys_platform": str(value.get("sys_platform", "")).lower(),
            "machine": str(value.get("machine", "")).lower(),
        }
        if normalized["sys_platform"] in {"any", "*"} and normalized["machine"] in {"", "any", "*"}:
            return {"any": True}
        return normalized
    text = str(value).lower()
    if text in {"any", "*"}:
        return {"any": True}
    parts = text.split("-", 1)
    if len(parts) == 2:
        return {"sys_platform": parts[0], "machine": parts[1]}
    return {"sys_platform": text, "machine": ""}


def _validate_sidecar_osdi_api(path: Path, profile: SimulatorProfile, sidecar: Path, data: dict[str, Any]) -> None:
    supported = set(profile.capabilities.osdi_api_versions)
    if supported and str(data["osdi_api"]) not in supported:
        raise _osdi_artifact_incompatible(
            path,
            profile,
            "sidecar OSDI API is incompatible with the simulator profile",
            metadata_path=sidecar,
            osdi_api=data["osdi_api"],
            supported_osdi_api_versions=sorted(supported),
        )


def _validate_sidecar_simulator_profile(path: Path, profile: SimulatorProfile, sidecar: Path, data: dict[str, Any]) -> None:
    profiles = data.get("simulator_profiles", {})
    profile_data: Any | None = None
    if isinstance(profiles, dict):
        profile_data = profiles.get(profile.backend_name, profiles.get(profile.dialect))
    elif isinstance(profiles, list):
        profile_data = profile.backend_name if profile.backend_name in profiles else profile.dialect if profile.dialect in profiles else None
    if profile_data is None:
        raise _osdi_artifact_incompatible(
            path,
            profile,
            "sidecar simulator profile is incompatible",
            metadata_path=sidecar,
            simulator_profiles=profiles,
        )
    if isinstance(profile_data, dict):
        dialect = profile_data.get("dialect")
        if dialect and dialect != profile.dialect:
            raise _osdi_artifact_incompatible(
                path,
                profile,
                "sidecar simulator dialect does not match the selected profile",
                metadata_path=sidecar,
                sidecar_dialect=dialect,
                profile_dialect=profile.dialect,
            )


def _external_osdi_untrusted(
    path: Path,
    profile: SimulatorProfile,
    message: str,
    **context: Any,
) -> ModelDiagnosticError:
    return ModelDiagnosticError(
        ModelDiagnostic(
            code="external_osdi_untrusted",
            message=message,
            context={
                "path": str(path),
                "simulator": profile.display_name,
                **{key: str(value) if isinstance(value, Path) else value for key, value in context.items()},
            },
        )
    )


def _osdi_artifact_incompatible(
    path: Path,
    profile: SimulatorProfile,
    message: str,
    **context: Any,
) -> ModelDiagnosticError:
    return ModelDiagnosticError(
        ModelDiagnostic(
            code="osdi_artifact_incompatible",
            message=message,
            context={
                "path": str(path),
                "simulator": profile.display_name,
                **{key: str(value) if isinstance(value, Path) else value for key, value in context.items()},
            },
        )
    )


def _diagnostic(
    code: str,
    message: str,
    techlib,
    corner,
    profile: SimulatorProfile,
    **extra: Any,
) -> ModelDiagnostic:
    corner_name = getattr(corner, "name", corner)
    context = {
        "techlib": getattr(techlib, "name", None),
        "corner": corner_name,
        "simulator": profile.display_name,
        "profile": profile.to_dict(),
    }
    context.update({key: value for key, value in extra.items() if value not in (None, [], {})})
    return ModelDiagnostic(code=code, message=message, context=context)
