"""Digital truth-table specification data types."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from monata.sim._digital_bits import _coerce_bits, _expected_row_parts, bit_combinations, bits_to_text
from monata.sim.digital_claims import (
    DigitalTransientObservation,
    DigitalVerificationClaim,
    ExpectedFn,
)


__all__ = [
    "DigitalTruthTableSpec",
    "ExpectedLike",
    "ExpectedTable",
]


@dataclass(frozen=True)
class ExpectedTable:
    """User-supplied expected output table for digital truth-table verification."""

    rows: Mapping[tuple[int, ...], tuple[int, ...]]

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[
            tuple[Sequence[int] | str, Sequence[int] | str]
            | Mapping[str, Sequence[int] | str]
        ],
    ) -> "ExpectedTable":
        resolved: dict[tuple[int, ...], tuple[int, ...]] = {}
        for row in rows:
            inputs, expected = _expected_row_parts(row)
            input_bits = _coerce_bits(inputs, "inputs")
            expected_bits = _coerce_bits(expected, "expected")
            if input_bits in resolved:
                raise ValueError(f"duplicate expected row for inputs {bits_to_text(input_bits)}")
            resolved[input_bits] = expected_bits
        return cls(resolved)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExpectedTable":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            rows = payload.get("rows")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("expected table JSON must contain a 'rows' list")
        return cls.from_rows(rows)

    def __post_init__(self) -> None:
        normalized: dict[tuple[int, ...], tuple[int, ...]] = {}
        for inputs, expected in self.rows.items():
            input_bits = _coerce_bits(inputs, "inputs")
            expected_bits = _coerce_bits(expected, "expected")
            if input_bits in normalized:
                raise ValueError(f"duplicate expected row for inputs {bits_to_text(input_bits)}")
            normalized[input_bits] = expected_bits
        object.__setattr__(self, "rows", normalized)

    def __call__(self, inputs: tuple[int, ...]) -> tuple[int, ...]:
        try:
            return self.rows[tuple(inputs)]
        except KeyError as exc:
            raise KeyError(f"missing expected row for inputs {bits_to_text(inputs)}") from exc

    def validate(
        self,
        *,
        input_width: int,
        output_width: int,
        require_complete: bool = True,
    ) -> None:
        for inputs, expected in self.rows.items():
            if len(inputs) != input_width:
                raise ValueError(
                    f"expected table row {bits_to_text(inputs)} has {len(inputs)} inputs, "
                    f"expected {input_width}"
                )
            if len(expected) != output_width:
                raise ValueError(
                    f"expected table row {bits_to_text(inputs)} has {len(expected)} outputs, "
                    f"expected {output_width}"
                )
        if require_complete:
            missing = [
                bits_to_text(bits)
                for bits in bit_combinations(input_width)
                if bits not in self.rows
            ]
            if missing:
                raise ValueError(
                    "expected table is missing input vectors: " + ", ".join(missing)
                )

    def as_dicts(self) -> list[dict[str, str]]:
        return [
            {"inputs": bits_to_text(inputs), "expected": bits_to_text(expected)}
            for inputs, expected in sorted(self.rows.items())
        ]


ExpectedLike = ExpectedFn | ExpectedTable


@dataclass(frozen=True)
class DigitalTruthTableSpec:
    """Project-declared truth-table view data.

    The spec carries user-owned facts: DUT identity, pins, rails, dependencies,
    simulation preference, and a user-supplied expected table. It intentionally
    does not derive logic semantics.
    """

    dut: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    expected: ExpectedLike | None = None
    oracle: str = "exact"
    dependencies: tuple[str, ...] = ()
    rails: tuple[str, str] = ("vdd", "0")
    complement_inputs: tuple[str, ...] = ()
    simulation_mode: str = "transient"
    transient_observation: DigitalTransientObservation | None = None
    stage: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return 2 ** len(self.inputs)

    @property
    def claim(self) -> dict[str, object]:
        return DigitalVerificationClaim.from_oracle(self.oracle).as_dict()

    @property
    def claim_summary(self) -> dict[str, object]:
        return DigitalVerificationClaim.from_dict(self.claim).summary()

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "complement_inputs", tuple(self.complement_inputs))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if isinstance(self.expected, ExpectedTable):
            self.expected.validate(
                input_width=len(self.inputs),
                output_width=len(self.outputs),
            )
