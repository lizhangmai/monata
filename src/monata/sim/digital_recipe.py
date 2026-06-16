"""Structured digital simulation recipes for data-only cellviews."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from monata.sim.digital_claims import DigitalTransientObservation
from monata.sim.digital_projection import PdkDeviceProjection, PdkModelProjectionLibrary
from monata.sim.task import SimArtifactOptions


__all__ = [
    "DigitalLoadRecipe",
    "DigitalModelCardRecipe",
    "DigitalModelProfile",
    "DigitalProjectionRecipe",
    "DigitalResolvedSimulationRecipe",
    "DigitalSimulationRecipe",
    "DigitalTimingRecipe",
]


_RECIPE_FIELDS = frozenset({
    "schema_version",
    "view_type",
    "analysis",
    "timing",
    "load",
    "observation",
    "model_profiles",
    "backend_options",
    "metadata",
})

_TIMING_FIELDS = frozenset({
    "period",
    "step",
    "truth_table_step",
    "transition",
    "skew_step",
})

_LOAD_FIELDS = frozenset({"capacitance"})

_PROFILE_FIELDS = frozenset({
    "projection",
    "spice_options",
    "models",
    "backend_options",
    "metadata",
})

_PROJECTION_FIELDS = frozenset({
    "source",
    "lib",
    "view",
    "devices",
})

_DEVICE_PROJECTION_FIELDS = frozenset({
    "model",
    "pins",
    "kind",
    "params",
})

_MODEL_CARD_FIELDS = frozenset({
    "name",
    "type",
    "params",
})


@dataclass(frozen=True)
class DigitalTimingRecipe:
    period: float
    step: float | None = None
    truth_table_step: float | None = None
    transition: float = 0.0
    skew_step: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DigitalTimingRecipe":
        _reject_unknown(payload, _TIMING_FIELDS, "digital timing recipe")
        if "period" not in payload:
            raise ValueError("digital timing recipe requires period")
        return cls(
            period=float(payload["period"]),
            step=_optional_float(payload.get("step")),
            truth_table_step=_optional_float(payload.get("truth_table_step")),
            transition=float(payload.get("transition", 0.0)),
            skew_step=float(payload.get("skew_step", 0.0)),
        )

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.step is not None and self.step <= 0:
            raise ValueError("step must be positive")
        if self.truth_table_step is not None and self.truth_table_step <= 0:
            raise ValueError("truth_table_step must be positive")
        if self.transition < 0:
            raise ValueError("transition must be non-negative")
        if self.skew_step < 0:
            raise ValueError("skew_step must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        data: dict[str, object] = {
            "period": self.period,
            "transition": self.transition,
            "skew_step": self.skew_step,
        }
        if self.step is not None:
            data["step"] = self.step
        if self.truth_table_step is not None:
            data["truth_table_step"] = self.truth_table_step
        return data


@dataclass(frozen=True)
class DigitalLoadRecipe:
    capacitance: str | float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DigitalLoadRecipe":
        if payload is None:
            return cls()
        _reject_unknown(payload, _LOAD_FIELDS, "digital load recipe")
        value = payload.get("capacitance")
        if value is not None and not isinstance(value, (str, int, float)):
            raise TypeError("load.capacitance must be a string, number, or null")
        return cls(capacitance=value)

    def to_mapping(self) -> dict[str, object]:
        return {} if self.capacitance is None else {"capacitance": self.capacitance}


@dataclass(frozen=True)
class DigitalModelCardRecipe:
    name: str
    model_type: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DigitalModelCardRecipe":
        allowed = _MODEL_CARD_FIELDS | frozenset({"lambda", "level", "vto", "kp", "gamma"})
        _reject_unknown(payload, allowed, "digital model card")
        params = dict(_optional_mapping(payload.get("params"), "model params"))
        for key, value in payload.items():
            if key not in _MODEL_CARD_FIELDS:
                params[key] = value
        return cls(
            name=_required_string(payload.get("name"), "model.name"),
            model_type=_required_string(payload.get("type"), "model.type"),
            params=params,
        )

    def apply_to_circuit(self, circuit: Any) -> None:
        circuit.model(self.name, self.model_type, **_python_param_names(self.params))

    def to_mapping(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "type": self.model_type,
        }
        for key, value in self.params.items():
            json_key = key[:-1] if key.endswith("_") and key[:-1] == "lambda" else key
            data[str(json_key)] = value
        return data


@dataclass(frozen=True)
class DigitalProjectionRecipe:
    source: str | None = None
    lib: str | None = None
    view: str | None = None
    devices: Mapping[str, PdkDeviceProjection] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DigitalProjectionRecipe":
        if payload is None:
            return cls()
        _reject_unknown(payload, _PROJECTION_FIELDS, "digital projection recipe")
        source = payload.get("source")
        if source is not None:
            source_text = str(source)
            if source_text != "library":
                raise ValueError(f"unsupported projection source: {source_text}")
            return cls(source=source_text)
        devices_payload = _optional_mapping(payload.get("devices"), "projection.devices")
        devices = {
            str(name): _device_projection_from_mapping(data, label=f"projection.devices.{name}")
            for name, data in devices_payload.items()
        }
        if not devices:
            return cls()
        return cls(
            lib=_required_string(payload.get("lib"), "projection.lib"),
            view=_required_string(payload.get("view"), "projection.view"),
            devices=devices,
        )

    def resolve_projection_library(self, library: Any) -> Any:
        if self.source == "library":
            if library is None:
                raise ValueError("projection source 'library' requires a library object")
            return library
        if self.devices:
            return PdkModelProjectionLibrary(
                lib=str(self.lib),
                view=str(self.view),
                devices=self.devices,
            )
        return None

    def to_mapping(self) -> dict[str, object]:
        if self.source is not None:
            return {"source": self.source}
        if not self.devices:
            return {}
        return {
            "lib": self.lib,
            "view": self.view,
            "devices": {
                name: {
                    "model": projection.model,
                    "pins": list(projection.pins),
                    "kind": projection.kind,
                    "params": dict(projection.params),
                }
                for name, projection in self.devices.items()
            },
        }


@dataclass(frozen=True)
class DigitalModelProfile:
    projection: DigitalProjectionRecipe = field(default_factory=DigitalProjectionRecipe)
    spice_options: Mapping[str, Any] = field(default_factory=dict)
    models: tuple[DigitalModelCardRecipe, ...] = ()
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DigitalModelProfile":
        _reject_unknown(payload, _PROFILE_FIELDS, "digital model profile")
        models_payload = payload.get("models", ())
        if isinstance(models_payload, (str, bytes)) or not isinstance(models_payload, (tuple, list)):
            raise TypeError("model profile models must be a list")
        return cls(
            projection=DigitalProjectionRecipe.from_mapping(
                _optional_mapping_or_none(payload.get("projection"), "projection")
            ),
            spice_options=dict(_optional_mapping(payload.get("spice_options"), "spice_options")),
            models=tuple(
                DigitalModelCardRecipe.from_mapping(_required_mapping(item, "model"))
                for item in models_payload
            ),
            backend_options=dict(_optional_mapping(payload.get("backend_options"), "backend_options")),
            metadata=dict(_optional_mapping(payload.get("metadata"), "metadata")),
        )

    def setup(self):
        if not self.spice_options and not self.models:
            return None

        def _setup(circuit: Any) -> None:
            if self.spice_options:
                circuit.options(**dict(self.spice_options))
            for model in self.models:
                model.apply_to_circuit(circuit)

        return _setup

    def to_mapping(self) -> dict[str, object]:
        data: dict[str, object] = {}
        projection = self.projection.to_mapping()
        if projection:
            data["projection"] = projection
        if self.spice_options:
            data["spice_options"] = dict(self.spice_options)
        if self.models:
            data["models"] = [model.to_mapping() for model in self.models]
        if self.backend_options:
            data["backend_options"] = dict(self.backend_options)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class DigitalResolvedSimulationRecipe:
    recipe: "DigitalSimulationRecipe"
    selected_profile: str
    profile: DigitalModelProfile
    observation: DigitalTransientObservation
    run_config: Any
    builder_kwargs: Mapping[str, Any]

    def metadata(self) -> dict[str, object]:
        return {
            "simulation_analysis": self.recipe.analysis,
            "model_profile": self.selected_profile,
            "observation": self.observation.as_dict(),
        }


@dataclass(frozen=True)
class DigitalSimulationRecipe:
    timing: DigitalTimingRecipe
    model_profiles: Mapping[str, DigitalModelProfile]
    observation: DigitalTransientObservation = field(default_factory=DigitalTransientObservation)
    load: DigitalLoadRecipe = field(default_factory=DigitalLoadRecipe)
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    view_type: str = "monata-simulation"
    analysis: str = "transient"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DigitalSimulationRecipe":
        _reject_unknown(payload, _RECIPE_FIELDS, "digital simulation recipe")
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ValueError(f"unsupported digital simulation schema_version: {schema_version}")
        view_type = str(payload.get("view_type", ""))
        if view_type != "monata-simulation":
            raise ValueError(f"unsupported digital simulation view_type: {view_type}")
        analysis = str(payload.get("analysis", ""))
        if analysis not in ("transient", "clocked_transient"):
            raise ValueError(f"unsupported digital simulation analysis: {analysis}")
        if analysis == "clocked_transient":
            analysis = "transient"
        profiles = _required_mapping(payload.get("model_profiles"), "model_profiles")
        if not profiles:
            raise ValueError("digital simulation recipe requires at least one model profile")
        return cls(
            timing=DigitalTimingRecipe.from_mapping(
                _required_mapping(payload.get("timing"), "timing")
            ),
            load=DigitalLoadRecipe.from_mapping(
                _optional_mapping_or_none(payload.get("load"), "load")
            ),
            observation=DigitalTransientObservation.from_dict(
                _optional_mapping(payload.get("observation"), "observation")
            ),
            model_profiles={
                str(name): DigitalModelProfile.from_mapping(
                    _required_mapping(profile, f"model_profiles.{name}")
                )
                for name, profile in profiles.items()
            },
            backend_options=dict(_optional_mapping(payload.get("backend_options"), "backend_options")),
            metadata=dict(_optional_mapping(payload.get("metadata"), "metadata")),
            analysis=analysis,
        )

    def select_profile(self, run_config: Any) -> tuple[str, DigitalModelProfile]:
        profile_name = getattr(run_config, "model", None)
        if profile_name is None:
            raise ValueError("digital simulation recipe selection requires run_config.model")
        name = str(profile_name)
        try:
            return name, self.model_profiles[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.model_profiles))
            raise ValueError(f"missing digital simulation model profile {name!r}; available: {available}") from exc

    def resolve(
        self,
        *,
        library: Any,
        run_config: Any,
        observation: DigitalTransientObservation | Mapping[str, Any] | None = None,
        cycles_per_vector: int | None = None,
        slots_per_task: int | None = None,
        artifacts: Any = None,
    ) -> DigitalResolvedSimulationRecipe:
        profile_name, profile = self.select_profile(run_config)
        resolved_observation = DigitalTransientObservation.resolve(
            self.observation if observation is None else observation,
            cycles_per_vector=cycles_per_vector,
            slots_per_task=slots_per_task,
        )
        effective_artifacts = SimArtifactOptions.coerce(artifacts)
        metadata = {
            "simulation_analysis": self.analysis,
            "model_profile": profile_name,
            "observation": resolved_observation.as_dict(),
            **dict(self.metadata),
            **dict(profile.metadata),
        }
        builder_kwargs = {
            "mode": self.analysis,
            "setup": profile.setup(),
            "projection_library": profile.projection.resolve_projection_library(library),
            "period": self.timing.period,
            "step": self.timing.step,
            "truth_table_step": self.timing.truth_table_step,
            "cycles_per_vector": (
                resolved_observation.cycles_per_vector
                if resolved_observation.cycles_per_vector is not None
                else 2
            ),
            "slots_per_task": resolved_observation.slots_per_task,
            "transition": self.timing.transition,
            "skew_step": self.timing.skew_step,
            "load_cap": self.load.capacitance,
            "metadata": metadata,
            "backend_options": {**dict(self.backend_options), **dict(profile.backend_options)},
            "artifacts": effective_artifacts,
        }
        return DigitalResolvedSimulationRecipe(
            recipe=self,
            selected_profile=profile_name,
            profile=profile,
            observation=resolved_observation,
            run_config=run_config,
            builder_kwargs=builder_kwargs,
        )

    def to_mapping(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema_version": self.schema_version,
            "view_type": self.view_type,
            "analysis": self.analysis,
            "timing": self.timing.to_mapping(),
            "load": self.load.to_mapping(),
            "observation": self.observation.as_dict(),
            "model_profiles": {
                name: profile.to_mapping()
                for name, profile in self.model_profiles.items()
            },
        }
        if self.backend_options:
            data["backend_options"] = dict(self.backend_options)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


def _device_projection_from_mapping(payload: Any, *, label: str) -> PdkDeviceProjection:
    data = _required_mapping(payload, label)
    _reject_unknown(data, _DEVICE_PROJECTION_FIELDS, label)
    return PdkDeviceProjection(
        model=_required_string(data.get("model"), f"{label}.model"),
        pins=_string_tuple(data.get("pins"), f"{label}.pins", require_nonempty=True),
        kind=str(data.get("kind", "M")),
        params=dict(_optional_mapping(data.get("params"), f"{label}.params")),
    )


def _reject_unknown(payload: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(key for key in payload if key not in allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _optional_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _required_mapping(value, label)


def _optional_mapping_or_none(value: Any, label: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _required_mapping(value, label)


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_tuple(value: Any, label: str, *, require_nonempty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a list of strings")
    result = tuple(_required_string(item, f"{label}[]") for item in value)
    if require_nonempty and not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _python_param_names(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lambda_" if key == "lambda" else str(key): value
        for key, value in params.items()
    }
