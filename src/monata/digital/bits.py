"""Bit-vector helpers shared by digital simulation modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "bit_combinations",
    "bits_to_text",
    "gray_code_bit_flip",
    "gray_code_chunks",
    "gray_code_sequence",
]


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


def gray_code_sequence(width: int) -> tuple[tuple[int, ...], ...]:
    """Generate the binary-reflected Gray-code traversal of all 2**width vectors.

    Each consecutive pair differs by exactly one bit.  The sequence starts
    at the all-zero vector and is deterministic for a given *width*.

    >>> gray_code_sequence(2)
    ((0, 0), (0, 1), (1, 1), (1, 0))
    """
    if width < 0:
        raise ValueError("width must be non-negative")
    if width == 0:
        return ((),)
    count = 1 << width
    codes: list[tuple[int, ...]] = []
    for value in range(count):
        gray = value ^ (value >> 1)
        codes.append(tuple((gray >> shift) & 1 for shift in range(width - 1, -1, -1)))
    return tuple(codes)


def gray_code_bit_flip(states: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return which input index flipped at each transition in a Gray-code sequence.

    >>> gray_code_bit_flip(((0,0), (0,1), (1,1), (1,0)))
    (1, 0, 1)
    """
    if len(states) < 2:
        return ()
    flipped: list[int] = []
    for prev_state, next_state in zip(states, states[1:]):
        diffs = [i for i, (a, b) in enumerate(zip(prev_state, next_state)) if a != b]
        if len(diffs) != 1:
            raise ValueError(
                f"Gray-code sequence transition flips {len(diffs)} bits "
                f"({bits_to_text(prev_state)} -> {bits_to_text(next_state)}); expected exactly 1"
            )
        flipped.append(diffs[0])
    return tuple(flipped)


def gray_code_chunks(
    width: int,
    *,
    slots_per_chunk: int | None = None,
) -> tuple[tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]], ...]:
    """Split a Gray-code sequence into parallel-simulatable chunks.

    Returns tuples of ``(chunk_index, initial_state, state_subsequence)``
    where each chunk's state_subsequence has at most *slots_per_chunk*
    vectors.  Chunks overlap: the last state of chunk N is the initial
    state of chunk N+1 so the full sequence reconstructs without gaps.
    """
    sequence = gray_code_sequence(width)
    vector_count = len(sequence)
    if vector_count <= 1:
        return ((0, sequence[0], ()),)
    resolved_slots = slots_per_chunk if slots_per_chunk is not None else vector_count - 1
    if resolved_slots < 1:
        raise ValueError("slots_per_chunk must be >= 1")
    chunks: list[tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    index = 0
    chunk_index = 0
    while index < vector_count - 1:
        end = min(index + resolved_slots, vector_count - 1)
        chunks.append((chunk_index, sequence[index], sequence[index + 1 : end + 1]))
        index = end
        chunk_index += 1
    return tuple(chunks)


def _coerce_bits(value: Sequence[int] | str, label: str) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_bits: Iterable[str | int] = value.strip()
    else:
        raw_bits = value
    bits = tuple(int(bit) for bit in raw_bits)
    if any(bit not in {0, 1} for bit in bits):
        raise ValueError(f"{label} bits must contain only 0 or 1")
    return bits
