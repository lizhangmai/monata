"""Configuration resolution for digital truth-table simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from monata.corner import CornerLike, OperatingCorner, coerce_operating_corner
from monata.models.flow import ResolvedModelFlow, SimulationModelConfig
from monata.netlist import Circuit, SubCircuit
from monata.sim.digital_claims import (
    DigitalComparator,
    DigitalOutputTolerance,
    DigitalVerificationClaim,
)
from monata.sim.digital_projection import PdkProjectionOwner
from monata.sim.digital_spec import ExpectedLike, ExpectedTable
from monata.sim.task import SimArtifactOptions


SetupFn = Callable[[Circuit], object]
SubCircuitInput = type[SubCircuit] | SubCircuit


@dataclass(frozen=True)
class _DigitalTruthTableConfig:
    dut: SubCircuitInput
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    expected: ExpectedLike | None
    dependencies: tuple[SubCircuitInput, ...]
    rails: tuple[str, str]
    complement_inputs: tuple[str, ...]
    vdd: float
    threshold: float
    period: float
    step: float | None
    truth_table_step: float | None
    cycles_per_vector: int
    slots_per_task: int | None
    transition: float
    skew_step: float
    sample_fraction: float
    load_cap: str | float | None
    tolerance: float | None
    comparator: DigitalComparator | DigitalOutputTolerance | None
    setup: SetupFn | None
    library: PdkProjectionOwner | None
    corner: OperatingCorner | None
    metadata: dict
    backend_options: dict[str, Any]
    artifacts: SimArtifactOptions
    claim: DigitalVerificationClaim
    model_config: SimulationModelConfig | None
    model_flow: ResolvedModelFlow | None

    @classmethod
    def resolve(
        cls,
        *,
        dut: SubCircuitInput,
        inputs: Iterable[str],
        outputs: Iterable[str],
        expected: ExpectedLike | None,
        oracle: str | None,
        claim: DigitalVerificationClaim | Mapping[str, Any] | None,
        dependencies: Iterable[SubCircuitInput],
        rails: tuple[str, str],
        complement_inputs: Iterable[str],
        vdd: float,
        threshold: float | None,
        period: float,
        step: float | None,
        truth_table_step: float | None,
        cycles_per_vector: int,
        slots_per_task: int | None,
        transition: float,
        skew_step: float,
        sample_fraction: float,
        load_cap: str | float | None,
        tolerance: float | None,
        comparator: DigitalComparator | DigitalOutputTolerance | None,
        setup: SetupFn | None,
        library: PdkProjectionOwner | None,
        corner: CornerLike,
        model_config: SimulationModelConfig | None,
        metadata: dict | None,
        backend_options: Mapping[str, Any] | None,
        artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None,
    ) -> _DigitalTruthTableConfig:
        resolved_metadata = dict(metadata or {})
        oracle_hint = oracle or resolved_metadata.get("oracle")
        if claim is None and oracle_hint is None:
            if tolerance is not None:
                oracle_hint = "toleranced"
            elif comparator is not None:
                oracle_hint = "custom"
        resolved_claim = DigitalVerificationClaim.resolve(
            expected=expected,
            oracle=oracle_hint,
            claim=claim,
        )
        resolved_comparator = _resolve_digital_comparator(
            claim=resolved_claim,
            tolerance=tolerance,
            comparator=comparator,
        )
        resolved_metadata.setdefault("oracle", resolved_claim.oracle)
        resolved_metadata.setdefault("claim", resolved_claim.as_dict())
        resolved_corner = _resolve_digital_corner(library, corner)
        model_flow = _resolve_model_flow(library, resolved_corner, model_config)
        if model_flow is not None:
            resolved_metadata.setdefault("model_flow", model_flow.to_dict())
        return cls(
            dut=dut,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            expected=expected,
            dependencies=tuple(dependencies),
            rails=rails,
            complement_inputs=tuple(complement_inputs),
            vdd=vdd,
            threshold=vdd / 2 if threshold is None else threshold,
            period=period,
            step=step,
            truth_table_step=truth_table_step,
            cycles_per_vector=int(cycles_per_vector),
            slots_per_task=None if slots_per_task is None else int(slots_per_task),
            transition=transition,
            skew_step=skew_step,
            sample_fraction=sample_fraction,
            load_cap=load_cap,
            tolerance=tolerance,
            comparator=resolved_comparator,
            setup=setup,
            library=library,
            corner=resolved_corner,
            metadata=resolved_metadata,
            backend_options=dict(backend_options or {}),
            artifacts=SimArtifactOptions.coerce(artifacts),
            claim=resolved_claim,
            model_config=model_config,
            model_flow=model_flow,
        )

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("DigitalTruthTable requires at least one input")
        if not self.outputs:
            raise ValueError("DigitalTruthTable requires at least one output")
        if self.complement_inputs and len(self.complement_inputs) != len(self.inputs):
            raise ValueError("complement_inputs must be empty or match inputs length")
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
        if self.cycles_per_vector < 1:
            raise ValueError("cycles_per_vector must be positive")
        if self.slots_per_task is not None and self.slots_per_task < 1:
            raise ValueError("slots_per_task must be positive")
        if not 0 < self.sample_fraction < 1:
            raise ValueError("sample_fraction must be between 0 and 1")
        if isinstance(self.expected, ExpectedTable):
            self.expected.validate(
                input_width=len(self.inputs),
                output_width=len(self.outputs),
            )


def _resolve_model_flow(
    library: PdkProjectionOwner | None,
    corner: OperatingCorner | None,
    model_config: SimulationModelConfig | None,
) -> ResolvedModelFlow | None:
    if library is None or corner is None or model_config is None:
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


def _resolve_digital_corner(
    library: PdkProjectionOwner | None,
    corner: CornerLike,
) -> OperatingCorner | None:
    if corner is None:
        return None
    resolver = getattr(library, "resolve_pdk_corner", None)
    if callable(resolver):
        resolved = resolver(corner)
        if isinstance(resolved, OperatingCorner):
            return resolved
        if resolved is not None:
            return coerce_operating_corner(resolved)
    return coerce_operating_corner(corner)


def _resolve_digital_comparator(
    *,
    claim: DigitalVerificationClaim,
    tolerance: float | None,
    comparator: DigitalComparator | DigitalOutputTolerance | None,
) -> DigitalComparator | DigitalOutputTolerance | None:
    if claim.oracle == "toleranced":
        if comparator is not None:
            return comparator
        if tolerance is None:
            raise ValueError("toleranced digital oracle requires tolerance or comparator")
        return DigitalOutputTolerance(float(tolerance))
    if claim.oracle == "custom":
        if comparator is None:
            raise ValueError("custom digital oracle requires comparator")
        return comparator
    if comparator is not None or tolerance is not None:
        raise ValueError("digital comparators require custom or toleranced oracle")
    return None


__all__ = ["SetupFn", "SubCircuitInput"]
