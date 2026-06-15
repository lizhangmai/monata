"""Model-flow recipes and resolved model-flow records."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from monata._paths import resolve_relative_path
from monata.models.artifacts import ModelArtifact
from monata.models.diagnostics import ModelDiagnostic
from monata.models.manifest import ModelSelection
from monata.sim.capabilities import SimulatorProfile
from monata.techlib.schema import TechlibError


ModelPolicyName = Literal[
    "auto",
    "native-first",
    "osdi-first",
    "source-compile",
    "bundled-only",
    "strict",
]

_MODEL_FLOW_RECIPE_FIELDS = frozenset({
    "name",
    "model_deck",
    "output",
    "requires",
    "dialects",
    "enabled",
    "disabled_diagnostic",
    "next_action",
    "source_va",
    "source_name",
    "source_includes",
    "source_metadata",
    "module_name",
    "compiler_args",
    "converter",
    "converted_model_card",
    "osdi_path",
    "package_policy",
    "validation",
    "metadata",
})


@dataclass(frozen=True)
class SimulationModelConfig:
    """Policy and permission inputs for simulator-aware model resolution."""

    simulator_profile: SimulatorProfile | None = None
    policy: ModelPolicyName = "auto"
    pinned_flow: str | None = None
    allow_compile: bool = True
    allow_external_osdi: bool = True
    allow_precompiled_package_artifacts: bool = False
    cache_dir: str | None = None
    openvaf_bin: str | None = None
    source_paths: dict[str, str] = field(default_factory=dict)
    external_osdi_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulator_profile": self.simulator_profile.to_dict() if self.simulator_profile is not None else None,
            "policy": self.policy,
            "pinned_flow": self.pinned_flow,
            "allow_compile": self.allow_compile,
            "allow_external_osdi": self.allow_external_osdi,
            "allow_precompiled_package_artifacts": self.allow_precompiled_package_artifacts,
            "cache_dir": self.cache_dir,
            "openvaf_bin": self.openvaf_bin,
            "source_paths": dict(self.source_paths),
            "external_osdi_paths": list(self.external_osdi_paths),
        }


@dataclass(frozen=True)
class ModelFlowRecipe:
    """Technology-declared recipe for consuming a model deck."""

    name: str
    model_deck: str
    output: str
    requires: dict[str, Any] = field(default_factory=dict)
    dialects: tuple[str, ...] = ()
    enabled: bool = True
    disabled_diagnostic: str | None = None
    next_action: str | None = None
    source_va: str | None = None
    source_name: str | None = None
    source_includes: tuple[str, ...] = ()
    source_metadata: str | None = None
    module_name: str | None = None
    compiler_args: tuple[str, ...] = ()
    converter: str | None = None
    converted_model_card: str | None = None
    osdi_path: str | None = None
    package_policy: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelFlowRecipe":
        unknown = sorted(key for key in data if key not in _MODEL_FLOW_RECIPE_FIELDS)
        if unknown:
            raise TypeError(f"unknown model flow recipe fields: {', '.join(unknown)}")
        dialects = data.get("dialects", ())
        if isinstance(dialects, str):
            dialects = (dialects,)
        source_includes = data.get("source_includes", ())
        if isinstance(source_includes, str):
            source_includes = (source_includes,)
        compiler_args = data.get("compiler_args", ())
        if isinstance(compiler_args, str):
            compiler_args = (compiler_args,)
        return cls(
            name=str(data["name"]),
            model_deck=str(data["model_deck"]),
            output=str(data["output"]),
            requires=dict(data.get("requires", {})),
            dialects=tuple(str(value) for value in dialects),
            enabled=bool(data.get("enabled", True)),
            disabled_diagnostic=data.get("disabled_diagnostic"),
            next_action=data.get("next_action"),
            source_va=data.get("source_va"),
            source_name=data.get("source_name"),
            source_includes=tuple(str(value) for value in source_includes),
            source_metadata=data.get("source_metadata"),
            module_name=data.get("module_name"),
            compiler_args=tuple(str(value) for value in compiler_args),
            converter=data.get("converter"),
            converted_model_card=data.get("converted_model_card"),
            osdi_path=data.get("osdi_path"),
            package_policy=data.get("package_policy"),
            validation=dict(data.get("validation", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "model_deck": self.model_deck,
            "output": self.output,
            "requires": dict(self.requires),
            "enabled": self.enabled,
        }
        optional = {
            "dialects": list(self.dialects),
            "disabled_diagnostic": self.disabled_diagnostic,
            "next_action": self.next_action,
            "source_va": self.source_va,
            "source_name": self.source_name,
            "source_includes": list(self.source_includes),
            "source_metadata": self.source_metadata,
            "module_name": self.module_name,
            "compiler_args": list(self.compiler_args),
            "converter": self.converter,
            "converted_model_card": self.converted_model_card,
            "osdi_path": self.osdi_path,
            "package_policy": self.package_policy,
            "validation": self.validation,
            "metadata": self.metadata,
        }
        for key, value in optional.items():
            if value not in (None, {}, []):
                data[key] = value
        return data

    @property
    def is_osdi_flow(self) -> bool:
        return bool(self.requires.get("osdi")) or self.output in {"osdi", "ngspice_osdi", "converted_osdi"}

    @property
    def is_native_flow(self) -> bool:
        return bool(self.requires.get("native_spice_model_levels")) or self.output in {"native_spice_lib", "spice_lib"}


@dataclass(frozen=True)
class ResolvedModelFlow:
    """Canonical resolver output, with concrete ModelSelection nested inside."""

    flow_name: str
    policy: str
    simulator_profile: SimulatorProfile | None
    model_selection: ModelSelection
    artifacts: tuple[ModelArtifact, ...] = ()
    generated_artifacts: tuple[ModelArtifact, ...] = ()
    cache_hits: dict[str, bool] = field(default_factory=dict)
    warnings: tuple[ModelDiagnostic, ...] = ()
    diagnostics: tuple[ModelDiagnostic, ...] = ()
    reuse_signature: str = ""

    def with_reuse_signature(self) -> "ResolvedModelFlow":
        payload = self.to_dict(include_reuse_signature=False)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return ResolvedModelFlow(
            flow_name=self.flow_name,
            policy=self.policy,
            simulator_profile=self.simulator_profile,
            model_selection=self.model_selection,
            artifacts=self.artifacts,
            generated_artifacts=self.generated_artifacts,
            cache_hits=dict(self.cache_hits),
            warnings=self.warnings,
            diagnostics=self.diagnostics,
            reuse_signature=f"sha256:{digest}",
        )

    def to_dict(self, *, include_reuse_signature: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "flow_name": self.flow_name,
            "policy": self.policy,
            "simulator_profile": self.simulator_profile.to_dict() if self.simulator_profile else None,
            "model_selection": self.model_selection.metadata,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "generated_artifacts": [artifact.to_dict() for artifact in self.generated_artifacts],
            "cache_hits": dict(self.cache_hits),
            "warnings": [diagnostic.to_dict() for diagnostic in self.warnings],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }
        if include_reuse_signature:
            data["reuse_signature"] = self.reuse_signature
        return data


def path_from_recipe(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    try:
        return resolve_relative_path(
            root,
            str(value),
            label="model flow path",
            root_label="techlib root",
        )
    except ValueError as exc:
        raise TechlibError(str(exc)) from exc
