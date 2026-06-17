"""Simulator capability profiles used by model-flow resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable


class CapabilityState(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: "CapabilityState | str | bool | None") -> "CapabilityState":
        if isinstance(value, cls):
            return value
        if value is True:
            return cls.SUPPORTED
        if value is False:
            return cls.UNSUPPORTED
        if value is None:
            return cls.UNKNOWN
        return cls(str(value).lower())


_SIMULATOR_CAPABILITIES_FIELDS = frozenset({
    "name",
    "dialect",
    "native_spice_model_levels",
    "osdi",
    "supports_runtime_verilog_a",
    "supports_xyce_plugins",
    "supports_subckt_wrappers",
    "osdi_api_versions",
    "probes",
})

_SIMULATOR_PROFILE_FIELDS = frozenset({
    "backend_name",
    "display_name",
    "dialect",
    "executable",
    "capabilities",
    "probes",
})


@dataclass(frozen=True)
class SimulatorCapabilities:
    name: str
    dialect: str
    native_spice_model_levels: frozenset[int] = field(default_factory=frozenset)
    osdi: CapabilityState = CapabilityState.UNKNOWN
    supports_runtime_verilog_a: bool = False
    supports_xyce_plugins: bool = False
    supports_subckt_wrappers: CapabilityState = CapabilityState.UNKNOWN
    osdi_api_versions: tuple[str, ...] = ()
    probes: dict[str, Any] = field(default_factory=dict)

    def supports_native_level(self, level: int) -> bool:
        return int(level) in self.native_spice_model_levels

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dialect": self.dialect,
            "native_spice_model_levels": sorted(self.native_spice_model_levels),
            "osdi": self.osdi.value,
            "supports_runtime_verilog_a": self.supports_runtime_verilog_a,
            "supports_xyce_plugins": self.supports_xyce_plugins,
            "supports_subckt_wrappers": self.supports_subckt_wrappers.value,
            "osdi_api_versions": list(self.osdi_api_versions),
            "probes": dict(self.probes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulatorCapabilities":
        unknown = sorted(key for key in data if key not in _SIMULATOR_CAPABILITIES_FIELDS)
        if unknown:
            raise TypeError(f"unknown simulator capabilities fields: {', '.join(unknown)}")
        return cls(
            name=str(data["name"]),
            dialect=str(data.get("dialect", data["name"])),
            native_spice_model_levels=frozenset(int(level) for level in data.get("native_spice_model_levels", ())),
            osdi=CapabilityState.coerce(data.get("osdi")),
            supports_runtime_verilog_a=bool(data.get("supports_runtime_verilog_a", False)),
            supports_xyce_plugins=bool(data.get("supports_xyce_plugins", False)),
            supports_subckt_wrappers=CapabilityState.coerce(data.get("supports_subckt_wrappers")),
            osdi_api_versions=tuple(str(version) for version in data.get("osdi_api_versions", ())),
            probes=dict(data.get("probes", {})),
        )


@dataclass(frozen=True)
class SimulatorProfile:
    backend_name: str
    display_name: str
    dialect: str
    executable: str | None
    capabilities: SimulatorCapabilities
    probes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "display_name": self.display_name,
            "dialect": self.dialect,
            "executable": self.executable,
            "capabilities": self.capabilities.to_dict(),
            "probes": dict(self.probes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulatorProfile":
        unknown = sorted(key for key in data if key not in _SIMULATOR_PROFILE_FIELDS)
        if unknown:
            raise TypeError(f"unknown simulator profile fields: {', '.join(unknown)}")
        capabilities = SimulatorCapabilities.from_dict(data["capabilities"])
        return cls(
            backend_name=str(data["backend_name"]),
            display_name=str(data.get("display_name", data["backend_name"])),
            dialect=str(data.get("dialect", capabilities.dialect)),
            executable=data.get("executable"),
            capabilities=capabilities,
            probes=dict(data.get("probes", {})),
        )


def ngspice_profile(
    *,
    osdi: CapabilityState | str | bool | None = CapabilityState.UNKNOWN,
    supports_subckt_wrappers: CapabilityState | str | bool | None = CapabilityState.UNKNOWN,
    osdi_api_versions: tuple[str, ...] = (),
    probes: dict[str, Any] | None = None,
) -> SimulatorProfile:
    """Return the static ngspice baseline profile used before runtime probes."""

    state = CapabilityState.coerce(osdi)
    return SimulatorProfile(
        backend_name="ngspice-subprocess",
        display_name="ngspice",
        dialect="ngspice",
        executable="ngspice",
        capabilities=SimulatorCapabilities(
            name="ngspice",
            dialect="ngspice",
            native_spice_model_levels=frozenset({1, 2, 3, 8, 49, 54, 55, 56, 57, 58, 60, 68, 73}),
            osdi=state,
            supports_runtime_verilog_a=False,
            supports_xyce_plugins=False,
            supports_subckt_wrappers=CapabilityState.coerce(supports_subckt_wrappers),
            osdi_api_versions=tuple(str(version) for version in osdi_api_versions),
            probes=probes or {},
        ),
        probes=probes or {},
    )


def native_level_profile(
    *,
    name: str = "fake-native",
    dialect: str = "spice",
    levels: Iterable[int] = (72,),
) -> SimulatorProfile:
    """Return a deterministic native-capable profile for tests and integrations."""

    capabilities = SimulatorCapabilities(
        name=name,
        dialect=dialect,
        native_spice_model_levels=frozenset(int(level) for level in levels),
        osdi=CapabilityState.UNSUPPORTED,
        supports_subckt_wrappers=CapabilityState.UNSUPPORTED,
    )
    return SimulatorProfile(
        backend_name=name,
        display_name=name,
        dialect=dialect,
        executable=None,
        capabilities=capabilities,
    )


__all__ = [
    "CapabilityState",
    "SimulatorCapabilities",
    "SimulatorProfile",
    "native_level_profile",
    "ngspice_profile",
]
