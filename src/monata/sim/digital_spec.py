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
    "ExpectedTableReference",
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


_DIGITAL_TRUTH_TABLE_FIELDS = frozenset({
    "schema_version",
    "view_type",
    "dut",
    "stage",
    "simulation_mode",
    "inputs",
    "outputs",
    "dependencies",
    "rails",
    "complement_inputs",
    "oracle",
    "expected",
    "metadata",
})

_EXPECTED_TABLE_REFERENCE_FIELDS = frozenset({
    "entry",
    "format",
})


@dataclass(frozen=True)
class ExpectedTableReference:
    """A path-agnostic reference to an expected-table data file."""

    entry: str
    format: str = "monata-expected-table-json"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExpectedTableReference":
        _reject_unknown(payload, _EXPECTED_TABLE_REFERENCE_FIELDS, "expected table reference")
        if "entry" not in payload:
            raise ValueError("expected table reference requires entry")
        entry = _required_string(payload["entry"], "expected.entry")
        format_name = str(payload.get("format", "monata-expected-table-json"))
        if format_name != "monata-expected-table-json":
            raise ValueError(f"unsupported expected table format: {format_name}")
        return cls(entry=entry, format=format_name)

    def to_mapping(self) -> dict[str, object]:
        return {"entry": self.entry, "format": self.format}


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
    expected_ref: ExpectedTableReference | Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        expected: ExpectedLike | None = None,
    ) -> "DigitalTruthTableSpec":
        """Parse the canonical data-only digital truth-table JSON shape.

        The parser deliberately stays filesystem-agnostic. Loader code resolves
        ``expected.entry`` and passes the parsed table through ``expected``.
        """

        _reject_unknown(payload, _DIGITAL_TRUTH_TABLE_FIELDS, "digital truth-table spec")
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ValueError(f"unsupported digital truth-table schema_version: {schema_version}")
        view_type = str(payload.get("view_type", ""))
        if view_type != "monata-digital-truth-table":
            raise ValueError(f"unsupported digital truth-table view_type: {view_type}")
        if "expected" not in payload:
            raise ValueError("digital truth-table spec requires expected")
        expected_ref = ExpectedTableReference.from_mapping(
            _required_mapping(payload["expected"], "expected")
        )
        inputs = _string_tuple(payload.get("inputs"), "inputs", require_nonempty=True)
        complement_inputs = _complement_inputs_from_mapping(
            payload.get("complement_inputs", {}),
            inputs=inputs,
        )
        metadata = dict(_optional_mapping(payload.get("metadata"), "metadata"))
        if "simulation_view" in metadata:
            raise ValueError("digital truth-table metadata cannot include simulation_view")
        return cls(
            dut=_required_string(payload.get("dut"), "dut"),
            inputs=inputs,
            outputs=_string_tuple(payload.get("outputs"), "outputs", require_nonempty=True),
            expected=expected,
            oracle=str(payload.get("oracle", "exact")),
            dependencies=_string_tuple(payload.get("dependencies", ()), "dependencies"),
            rails=_rails_from_mapping(payload.get("rails", {"vdd": "vdd", "vss": "0"})),
            complement_inputs=complement_inputs,
            simulation_mode=str(payload.get("simulation_mode", "transient")),
            stage=str(payload.get("stage", "custom")),
            metadata=metadata,
            expected_ref=expected_ref,
        )

    @property
    def row_count(self) -> int:
        return 2 ** len(self.inputs)

    @property
    def claim(self) -> dict[str, object]:
        return DigitalVerificationClaim.from_oracle(self.oracle).as_dict()

    @property
    def claim_summary(self) -> dict[str, object]:
        return DigitalVerificationClaim.from_dict(self.claim).summary()

    def to_mapping(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema_version": 1,
            "view_type": "monata-digital-truth-table",
            "dut": self.dut,
            "stage": self.stage,
            "simulation_mode": self.simulation_mode,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "dependencies": list(self.dependencies),
            "rails": {"vdd": self.rails[0], "vss": self.rails[1]},
            "complement_inputs": _complement_inputs_to_mapping(
                self.inputs,
                self.complement_inputs,
            ),
            "oracle": self.oracle,
        }
        metadata = dict(self.metadata)
        if self.expected_ref is not None:
            data["expected"] = _expected_ref_to_mapping(self.expected_ref)
        if metadata:
            data["metadata"] = metadata
        return data

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "complement_inputs", tuple(self.complement_inputs))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.expected_ref is not None and not isinstance(self.expected_ref, ExpectedTableReference):
            object.__setattr__(
                self,
                "expected_ref",
                ExpectedTableReference.from_mapping(self.expected_ref),
            )
        if isinstance(self.expected, ExpectedTable):
            self.expected.validate(
                input_width=len(self.inputs),
                output_width=len(self.outputs),
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


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_tuple(
    value: Any,
    label: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    if value is None:
        if require_nonempty:
            raise ValueError(f"{label} must not be empty")
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a list of strings")
    result = tuple(_required_string(item, f"{label}[]") for item in value)
    if require_nonempty and not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _rails_from_mapping(value: Any) -> tuple[str, str]:
    payload = _required_mapping(value, "rails")
    _reject_unknown(payload, frozenset({"vdd", "vss"}), "rails")
    return (
        _required_string(payload.get("vdd"), "rails.vdd"),
        _required_string(payload.get("vss"), "rails.vss"),
    )


def _complement_inputs_from_mapping(
    value: Any,
    *,
    inputs: tuple[str, ...],
) -> tuple[str, ...]:
    payload = _required_mapping(value, "complement_inputs")
    if not payload:
        return ()
    unknown = sorted(key for key in payload if key not in inputs)
    if unknown:
        raise ValueError(f"unknown complement input names: {', '.join(unknown)}")
    missing = [name for name in inputs if name not in payload]
    if missing:
        raise ValueError(f"missing complement input names: {', '.join(missing)}")
    return tuple(_required_string(payload[name], f"complement_inputs.{name}") for name in inputs)


def _complement_inputs_to_mapping(
    inputs: tuple[str, ...],
    complements: tuple[str, ...],
) -> dict[str, str]:
    if not complements:
        return {}
    if len(inputs) != len(complements):
        raise ValueError("complement_inputs must be empty or match inputs length")
    return dict(zip(inputs, complements, strict=True))


def _expected_ref_to_mapping(value: ExpectedTableReference | Mapping[str, Any]) -> dict[str, object]:
    if isinstance(value, ExpectedTableReference):
        return value.to_mapping()
    return ExpectedTableReference.from_mapping(value).to_mapping()
