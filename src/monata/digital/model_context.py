"""Model projection context for digital task construction."""

from __future__ import annotations

from dataclasses import dataclass

from monata.corner import OperatingCorner
from monata.digital.projection import PdkProjectionOwner
from monata.models.flow import ResolvedModelFlow, SimulationModelConfig
from monata.netlist import Circuit


@dataclass(frozen=True)
class DigitalModelContext:
    """Explicit model/projection context used while building digital circuits."""

    projection_library: PdkProjectionOwner | None = None
    corner: OperatingCorner | None = None
    model_flow: ResolvedModelFlow | None = None

    @property
    def osdi_paths(self) -> tuple[str, ...]:
        if self.model_flow is None:
            return ()
        return tuple(getattr(self.model_flow.model_selection, "osdi_paths", ()))

    def project_circuit(self, circuit: Circuit) -> Circuit:
        if self.projection_library is None:
            return circuit
        if self.model_flow is None:
            return self.projection_library.project_pdk_instances(circuit, corner=self.corner)
        projected = self.projection_library.project_pdk_instances(
            circuit,
            corner=self.corner,
            include_models=False,
        )
        self.model_flow.model_selection.apply_to_circuit(projected)
        return projected


def resolve_digital_model_context(
    *,
    projection_library: PdkProjectionOwner | None,
    corner: OperatingCorner | None,
    model_config: SimulationModelConfig | None,
) -> DigitalModelContext:
    return DigitalModelContext(
        projection_library=projection_library,
        corner=corner,
        model_flow=resolve_digital_model_flow(
            projection_library=projection_library,
            corner=corner,
            model_config=model_config,
        ),
    )


def resolve_digital_model_flow(
    *,
    projection_library: PdkProjectionOwner | None,
    corner: OperatingCorner | None,
    model_config: SimulationModelConfig | None,
) -> ResolvedModelFlow | None:
    if projection_library is None or corner is None or model_config is None:
        return None
    techlib_name = getattr(corner, "techlib", None)
    if not techlib_name:
        return None
    from monata.techlib.registry import TechlibRegistry

    techlib = TechlibRegistry()[techlib_name]
    return techlib.resolve_model_flow(
        corner,
        model_config=model_config,
        simulator_profile=model_config.simulator_profile,
    )


__all__ = [
    "DigitalModelContext",
    "resolve_digital_model_context",
    "resolve_digital_model_flow",
]
