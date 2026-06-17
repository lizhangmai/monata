"""Digital verification claims, comparators, and observation settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, cast


ExpectedFn = Callable[[tuple[int, ...]], tuple[int, ...]]
DigitalOracle = Literal["exact", "observed", "toleranced", "custom"]
DigitalClaimStrength = Literal["exact", "observation", "toleranced", "custom"]

_DIGITAL_VERIFICATION_CLAIM_FIELDS = frozenset({
    "oracle",
    "claim_strength",
    "assertion",
    "expected_required",
    "correctness_claim",
})

_DIGITAL_TRANSIENT_OBSERVATION_FIELDS = frozenset({
    "stop",
    "uic",
    "cycles_per_vector",
    "slots_per_task",
    "clock_period",
})


@dataclass(frozen=True)
class DigitalVerificationClaim:
    oracle: DigitalOracle
    claim_strength: DigitalClaimStrength
    assertion: str
    expected_required: bool
    correctness_claim: str

    @classmethod
    def resolve(
        cls,
        *,
        expected: object | None,
        oracle: str | None = None,
        claim: "DigitalVerificationClaim | Mapping[str, Any] | None" = None,
    ) -> "DigitalVerificationClaim":
        if isinstance(claim, DigitalVerificationClaim):
            resolved = claim
        elif claim is not None:
            resolved = cls.from_dict(claim)
        else:
            resolved = cls.from_oracle(oracle or ("exact" if expected is not None else "observed"))
        if resolved.expected_required and expected is None:
            raise ValueError(f"{resolved.oracle} digital verification claims require an expected oracle")
        return resolved

    @classmethod
    def from_oracle(cls, oracle: str) -> "DigitalVerificationClaim":
        text = str(oracle)
        if text == "exact":
            return cls(
                oracle="exact",
                claim_strength="exact",
                assertion="actual output bits equal expected oracle bits",
                expected_required=True,
                correctness_claim="functional_truth_table",
            )
        if text == "observed":
            return cls(
                oracle="observed",
                claim_strength="observation",
                assertion="source outputs were observed for each input vector",
                expected_required=False,
                correctness_claim="none",
            )
        if text == "toleranced":
            return cls(
                oracle="toleranced",
                claim_strength="toleranced",
                assertion="sampled output voltages satisfy the declared tolerance comparator",
                expected_required=True,
                correctness_claim="toleranced_truth_table",
            )
        if text == "custom":
            return cls(
                oracle="custom",
                claim_strength="custom",
                assertion="custom comparator accepted the observed row",
                expected_required=False,
                correctness_claim="custom_comparator",
            )
        raise ValueError(f"unsupported digital verification oracle: {oracle}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DigitalVerificationClaim":
        unknown = sorted(key for key in payload if key not in _DIGITAL_VERIFICATION_CLAIM_FIELDS)
        if unknown:
            raise ValueError(f"unknown digital verification claim fields: {', '.join(unknown)}")
        raw_oracle = str(payload["oracle"])
        if raw_oracle not in {"exact", "observed", "toleranced", "custom"}:
            raise ValueError(f"unsupported digital verification claim oracle: {raw_oracle}")
        claim_strength = str(payload["claim_strength"])
        if claim_strength not in {"exact", "observation", "toleranced", "custom"}:
            raise ValueError(
                f"unsupported digital verification claim strength: {claim_strength}"
            )
        return cls(
            oracle=cast(DigitalOracle, raw_oracle),
            claim_strength=cast(DigitalClaimStrength, claim_strength),
            assertion=str(payload["assertion"]),
            expected_required=bool(payload["expected_required"]),
            correctness_claim=str(payload["correctness_claim"]),
        )

    def summary(self) -> dict[str, object]:
        if self.expected_required and self.claim_strength == "exact":
            label = "exact-functional-truth-table"
            pass_meaning = "PASS means actual output bits matched the expected oracle truth table."
            warning = None
        elif self.claim_strength == "toleranced":
            label = "toleranced-truth-table"
            pass_meaning = "PASS means sampled output voltages satisfied the declared tolerance comparator."
            warning = None
        elif self.claim_strength == "custom":
            label = "custom-comparator"
            pass_meaning = "PASS means the declared custom comparator accepted the observed row."
            warning = None
        elif not self.expected_required:
            label = f"{self.oracle}-only"
            pass_meaning = (
                "PASS means source outputs were observed for every input vector; "
                "it does not prove exact arithmetic or product correctness."
            )
            warning = "No exact product correctness claim."
        else:
            label = f"{self.oracle}-{self.claim_strength}"
            pass_meaning = "PASS means the result satisfied the declared verification oracle."
            warning = None
        return {
            "label": label,
            "claim_strength": self.claim_strength,
            "correctness_claim": self.correctness_claim,
            "expected_required": self.expected_required,
            "pass_meaning": pass_meaning,
            "warning": warning,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "oracle": self.oracle,
            "claim_strength": self.claim_strength,
            "assertion": self.assertion,
            "expected_required": self.expected_required,
            "correctness_claim": self.correctness_claim,
        }


@dataclass(frozen=True)
class DigitalComparisonContext:
    inputs: tuple[int, ...]
    outputs: tuple[str, ...]
    actual: tuple[int, ...]
    expected: tuple[int, ...] | None
    samples: Mapping[str, float]
    vdd: float
    threshold: float
    claim: DigitalVerificationClaim


@dataclass(frozen=True)
class DigitalComparisonResult:
    matched: bool
    reason: str | None = None
    details: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"matched": self.matched}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.details is not None:
            payload["details"] = dict(self.details)
        return payload


DigitalComparator = Callable[[DigitalComparisonContext], bool | DigitalComparisonResult]


@dataclass(frozen=True)
class DigitalOutputTolerance:
    voltage_tolerance: float

    def __post_init__(self) -> None:
        if self.voltage_tolerance < 0:
            raise ValueError("voltage_tolerance must be non-negative")

    def compare(self, context: DigitalComparisonContext) -> DigitalComparisonResult:
        if context.expected is None:
            return DigitalComparisonResult(False, reason="missing_expected")
        if len(context.expected) != len(context.outputs) or len(context.actual) != len(context.outputs):
            return DigitalComparisonResult(
                False,
                reason="shape_mismatch",
                details={
                    "output_count": len(context.outputs),
                    "actual_count": len(context.actual),
                    "expected_count": len(context.expected),
                },
            )
        details: dict[str, object] = {}
        matched = True
        for output, expected_bit in zip(context.outputs, context.expected):
            if output not in context.samples:
                details[output] = {"missing_sample": True}
                matched = False
                continue
            sample = float(context.samples[output])
            target = context.vdd if expected_bit else 0.0
            delta = abs(sample - target)
            within = delta <= self.voltage_tolerance
            matched = matched and within
            details[output] = {
                "sample": sample,
                "target": target,
                "delta": delta,
                "tolerance": self.voltage_tolerance,
                "within_tolerance": within,
            }
        return DigitalComparisonResult(
            matched,
            reason="within_tolerance" if matched else "outside_tolerance",
            details=details,
        )


@dataclass(frozen=True)
class DigitalTransientObservation:
    """Configuration for digital transient vector-sequence runs."""

    stop: float | None = None
    uic: bool = False
    cycles_per_vector: int | None = None
    slots_per_task: int | None = None
    clock_period: float | None = None

    @classmethod
    def resolve(
        cls,
        observation: "DigitalTransientObservation | Mapping[str, Any] | None" = None,
        *,
        stop: float | None = None,
        uic: bool | None = None,
        cycles_per_vector: int | None = None,
        slots_per_task: int | None = None,
        clock_period: float | None = None,
    ) -> "DigitalTransientObservation":
        if isinstance(observation, DigitalTransientObservation):
            resolved = observation
        elif observation is None:
            resolved = cls()
        else:
            resolved = cls.from_dict(observation)
        if (
            stop is not None
            or uic is not None
            or cycles_per_vector is not None
            or slots_per_task is not None
            or clock_period is not None
        ):
            resolved = cls(
                stop=resolved.stop if stop is None else stop,
                uic=resolved.uic if uic is None else uic,
                cycles_per_vector=(
                    resolved.cycles_per_vector
                    if cycles_per_vector is None
                    else cycles_per_vector
                ),
                slots_per_task=(
                    resolved.slots_per_task
                    if slots_per_task is None
                    else slots_per_task
                ),
                clock_period=(
                    resolved.clock_period
                    if clock_period is None
                    else clock_period
                ),
            )
        resolved._validate()
        return resolved

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DigitalTransientObservation":
        unknown = sorted(key for key in payload if key not in _DIGITAL_TRANSIENT_OBSERVATION_FIELDS)
        if unknown:
            raise ValueError(f"unknown digital transient observation fields: {', '.join(unknown)}")
        raw_stop = payload.get("stop")
        raw_cycles = payload.get("cycles_per_vector")
        raw_slots = payload.get("slots_per_task")
        raw_clock = payload.get("clock_period")
        return cls(
            stop=None if raw_stop is None else float(raw_stop),
            uic=bool(payload.get("uic", False)),
            cycles_per_vector=None if raw_cycles is None else int(raw_cycles),
            slots_per_task=None if raw_slots is None else int(raw_slots),
            clock_period=None if raw_clock is None else float(raw_clock),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "stop": self.stop,
            "uic": self.uic,
            "cycles_per_vector": self.cycles_per_vector,
            "slots_per_task": self.slots_per_task,
            "clock_period": self.clock_period,
        }

    def _validate(self) -> None:
        if self.stop is not None and self.stop <= 0:
            raise ValueError("transient observation stop must be positive")
        if self.cycles_per_vector is not None and self.cycles_per_vector < 1:
            raise ValueError("cycles_per_vector must be positive")
        if self.slots_per_task is not None and self.slots_per_task < 1:
            raise ValueError("slots_per_task must be positive")
        if self.clock_period is not None and self.clock_period <= 0:
            raise ValueError("clock_period must be positive")
