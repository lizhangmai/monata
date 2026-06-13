"""Typed simulation task contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar

from monata.corner import coerce_operating_corner
from monata.sim._frozen import frozen_mapping
from monata.sim.analysis_spec import AnalysisSpec

DEFAULT_SIMULATOR = "ngspice-subprocess"
DEFAULT_SIM_TIMEOUT_SECONDS = 300.0
SimPayload: TypeAlias = Mapping[str, Any]
ParamOverrides: TypeAlias = Mapping[str, Any]
TAnalysisSpec = TypeVar("TAnalysisSpec", bound=AnalysisSpec)


@dataclass(frozen=True, init=False)
class SimArtifactOptions:
    """Execution artifact persistence options for a simulation task."""

    directory: Path | None
    overwrite: bool

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        object.__setattr__(self, "directory", None if directory is None else Path(directory))
        object.__setattr__(self, "overwrite", bool(overwrite))

    @classmethod
    def coerce(cls, value: ArtifactOptionsLike = None) -> "SimArtifactOptions":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, (str, Path)):
            return cls(value)
        if isinstance(value, Mapping):
            return cls(
                directory=value.get("directory"),
                overwrite=bool(value.get("overwrite", False)),
            )
        raise TypeError("artifacts must be a SimArtifactOptions, mapping, path, string, or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": None if self.directory is None else str(self.directory),
            "overwrite": self.overwrite,
        }


ArtifactOptionsLike: TypeAlias = SimArtifactOptions | Mapping[str, Any] | str | Path | None


@dataclass(frozen=True, init=False)
class SimTask(Generic[TAnalysisSpec]):
    circuit: Any
    analysis_spec: TAnalysisSpec
    simulator: str
    corner: Any
    param_overrides: SimPayload
    output_names: tuple[str, ...]
    osdi_paths: tuple[Path, ...]
    metadata: SimPayload
    backend_options: SimPayload
    artifacts: SimArtifactOptions
    timeout: float | None

    def __init__(
        self,
        circuit: Any,
        analysis_spec: TAnalysisSpec,
        simulator: str = DEFAULT_SIMULATOR,
        corner: Any = None,
        param_overrides: ParamOverrides | None = None,
        output_names: Iterable[str] | None = None,
        osdi_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
        metadata: SimPayload | None = None,
        timeout: float | int | None = DEFAULT_SIM_TIMEOUT_SECONDS,
        backend_options: SimPayload | None = None,
        artifacts: ArtifactOptionsLike = None,
    ) -> None:
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(self, "analysis_spec", analysis_spec)
        object.__setattr__(self, "simulator", str(simulator))
        object.__setattr__(self, "corner", coerce_operating_corner(corner))
        object.__setattr__(self, "param_overrides", _read_only_mapping(param_overrides))
        object.__setattr__(self, "output_names", normalize_output_names(output_names))
        object.__setattr__(self, "osdi_paths", tuple(Path(path) for path in (osdi_paths or ())))
        object.__setattr__(self, "metadata", _read_only_mapping(metadata))
        object.__setattr__(self, "backend_options", _read_only_mapping(backend_options))
        object.__setattr__(self, "artifacts", SimArtifactOptions.coerce(artifacts))
        object.__setattr__(self, "timeout", _coerce_timeout(timeout))


def normalize_output_names(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError("output_names must be an iterable of strings, not a string")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _coerce_output_name(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _coerce_output_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("output_names must contain strings")
    if not value:
        raise ValueError("output name is required")
    if any(char.isspace() for char in value) or any(char in value for char in ";'\""):
        raise ValueError("invalid output name: whitespace, quotes, and command separators are not allowed")
    if value.count("(") != value.count(")") or value.count("[") != value.count("]"):
        raise ValueError("invalid output name: delimiters are not balanced")
    return value


def _coerce_timeout(value: float | int | None) -> float | None:
    if value is None:
        return None
    timeout = float(value)
    if timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds or None")
    return timeout


def _read_only_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return frozen_mapping(values)


__all__ = [
    "ArtifactOptionsLike",
    "DEFAULT_SIMULATOR",
    "DEFAULT_SIM_TIMEOUT_SECONDS",
    "ParamOverrides",
    "SimArtifactOptions",
    "SimPayload",
    "SimTask",
    "normalize_output_names",
]
