"""Digital truth-table result records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from monata.sim._digital_bits import bits_to_text
from monata.sim.digital_claims import DigitalComparisonResult, DigitalVerificationClaim
from monata.sim.results import SimResult

__all__ = [
    "DigitalPropagationDelayRow",
    "DigitalTruthTableResult",
    "DigitalTruthTableRow",
]


@dataclass(frozen=True)
class DigitalTruthTableRow:
    inputs: tuple[int, ...]
    actual: tuple[int, ...]
    expected: tuple[int, ...] | None = None
    samples: dict[str, float] | None = None
    sample_time: float | None = None
    claim: DigitalVerificationClaim | None = None
    comparison: DigitalComparisonResult | None = None

    @property
    def passed(self) -> bool:
        if self.comparison is not None:
            return self.comparison.matched
        return self.expected is None or self.actual == self.expected

    def as_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "inputs": bits_to_text(self.inputs),
            "actual": bits_to_text(self.actual),
            "status": "PASS" if self.passed else "FAIL",
        }
        if self.expected is None:
            if self.comparison is None:
                row["status"] = "OBSERVED_NON_EXACT"
        else:
            row["expected"] = bits_to_text(self.expected)
        if self.claim is not None:
            row["claim"] = self.claim.as_dict()
        if self.comparison is not None:
            row["comparison"] = self.comparison.as_dict()
        if self.sample_time is not None:
            row["sample_time"] = self.sample_time
        if self.samples is not None:
            row["samples"] = dict(self.samples)
        return row


@dataclass(frozen=True)
class DigitalPropagationDelayRow:
    from_inputs: tuple[int, ...]
    to_inputs: tuple[int, ...]
    input_name: str
    output_name: str
    input_edge: Literal["rise", "fall"]
    output_edge: Literal["rise", "fall"]
    input_crossing: float
    output_crossing: float
    delay: float

    def as_dict(self) -> dict[str, object]:
        return {
            "from_inputs": bits_to_text(self.from_inputs),
            "to_inputs": bits_to_text(self.to_inputs),
            "input": self.input_name,
            "output": self.output_name,
            "input_edge": self.input_edge,
            "output_edge": self.output_edge,
            "input_crossing": self.input_crossing,
            "output_crossing": self.output_crossing,
            "delay": self.delay,
        }


class DigitalTruthTableResult:
    def __init__(
        self,
        rows: Iterable[DigitalTruthTableRow],
        sim_result: SimResult,
        mode: str,
        claim: DigitalVerificationClaim | None = None,
        sim_results: Iterable[SimResult] | None = None,
        propagation_delays: Iterable[DigitalPropagationDelayRow] | None = None,
        propagation_delay_sim_result: SimResult | None = None,
        propagation_delay_sim_results: Iterable[SimResult] | None = None,
        propagation_delay_coverage: Mapping[str, Any] | None = None,
    ):
        self.rows = list(rows)
        self.sim_result = sim_result
        self.sim_results = tuple(sim_results) if sim_results is not None else (sim_result,)
        if not self.sim_results:
            raise ValueError("DigitalTruthTableResult requires at least one SimResult")
        self.mode = mode
        self.claim = claim
        self.propagation_delays = tuple(propagation_delays or ())
        if propagation_delay_sim_results is not None:
            self.propagation_delay_sim_results = tuple(propagation_delay_sim_results)
        elif propagation_delay_sim_result is not None:
            self.propagation_delay_sim_results = (propagation_delay_sim_result,)
        else:
            self.propagation_delay_sim_results = ()
        self.propagation_delay_sim_result = (
            self.propagation_delay_sim_results[0]
            if self.propagation_delay_sim_results
            else None
        )
        self.propagation_delay_coverage = dict(propagation_delay_coverage or {})

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def failed(self) -> list[DigitalTruthTableRow]:
        return [row for row in self.rows if not row.passed]

    @property
    def max_propagation_delay(self) -> float | None:
        row = self.max_propagation_delay_arc
        return None if row is None else row.delay

    @property
    def max_propagation_delay_arc(self) -> DigitalPropagationDelayRow | None:
        if not self.propagation_delays:
            return None
        return max(self.propagation_delays, key=lambda row: row.delay)

    def with_propagation_delays(
        self,
        propagation_delays: Iterable[DigitalPropagationDelayRow],
        *,
        sim_result: SimResult | None = None,
        sim_results: Iterable[SimResult] | None = None,
    ) -> "DigitalTruthTableResult":
        resolved_sim_results = tuple(sim_results) if sim_results is not None else ()
        if sim_result is not None:
            resolved_sim_results = (sim_result, *resolved_sim_results)
        if not resolved_sim_results:
            raise ValueError("propagation delay results require at least one SimResult")
        return DigitalTruthTableResult(
            self.rows,
            self.sim_result,
            self.mode,
            claim=self.claim,
            sim_results=self.sim_results,
            propagation_delays=propagation_delays,
            propagation_delay_sim_results=resolved_sim_results,
            propagation_delay_coverage=self.propagation_delay_coverage,
        )

    def as_dicts(self) -> list[dict[str, object]]:
        return [row.as_dict() for row in self.rows]

    def measurements_as_dict(self) -> dict[str, object]:
        failed_rows = len(self.failed)
        measurements: dict[str, object] = {
            "truth_table": {
                "status": "PASS" if failed_rows == 0 else "FAIL",
                "rows": len(self.rows),
                "failed_rows": failed_rows,
                "results": self.as_dicts(),
            }
        }
        worst_arc = self.max_propagation_delay_arc
        if worst_arc is not None:
            measurements["max_propagation_delay"] = {
                "value": worst_arc.delay,
                "unit": "s",
                "arc": worst_arc.as_dict(),
            }
            if self.propagation_delay_coverage:
                measurements["max_propagation_delay"]["coverage"] = dict(
                    self.propagation_delay_coverage
                )
        return measurements
