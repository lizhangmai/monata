"""Bit-vector helpers shared by digital simulation modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

__all__ = ["bit_combinations", "bits_to_text"]


def bit_combinations(width: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))
        for value in range(2**width)
    )


def bits_to_text(bits: Iterable[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def _expected_row_parts(
    row: tuple[Sequence[int] | str, Sequence[int] | str]
    | Mapping[str, Sequence[int] | str],
) -> tuple[Sequence[int] | str, Sequence[int] | str]:
    if isinstance(row, Mapping):
        if "inputs" not in row or "expected" not in row:
            raise ValueError("expected table row mappings require 'inputs' and 'expected'")
        return row["inputs"], row["expected"]
    if len(row) != 2:
        raise ValueError("expected table rows must be (inputs, expected) pairs")
    return row


def _coerce_bits(value: Sequence[int] | str, label: str) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_bits: Iterable[str | int] = value.strip()
    else:
        raw_bits = value
    bits = tuple(int(bit) for bit in raw_bits)
    if any(bit not in {0, 1} for bit in bits):
        raise ValueError(f"{label} bits must contain only 0 or 1")
    return bits
