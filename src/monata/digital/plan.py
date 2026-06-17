"""Internal digital truth-table planning primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from monata.digital.bits import bit_combinations, gray_code_chunks, gray_code_sequence
from monata.sim.results import SimResult


DIGITAL_TASK_METADATA_SCHEMA = "monata.sim.digital-task.v1"
MONATA_METADATA_KEY = "monata"
DIGITAL_TASK_METADATA_KEY = "digital_task_v1"


@dataclass(frozen=True)
class DigitalStimulusMetadata:
    """Typed view of digital stimulus metadata stored on SimTask/SimResult."""

    kind: str
    initial_settle: float
    clock_period: float = 0.0
    chunk_index: int = 0

    def require_kind(self, expected: str, *, dut_name: str) -> None:
        if self.kind != expected:
            raise RuntimeError(
                f"{dut_name} expected digital stimulus metadata kind {expected!r}, got {self.kind!r}"
            )


@dataclass
class DigitalTruthTablePlan:
    """Owns digital sequence planning, timing schedules, and task metadata."""

    table: Any

    def combinations(self) -> tuple[tuple[int, ...], ...]:
        return bit_combinations(len(self.table.inputs))

    def sequence(self) -> tuple[tuple[int, ...], ...]:
        return gray_code_sequence(len(self.table.inputs))

    def sequence_states(
        self,
        *,
        slots_per_chunk: int | None = None,
    ) -> tuple[tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]], ...]:
        resolved_slots = (
            slots_per_chunk if slots_per_chunk is not None else self.table.slots_per_task
        )
        return gray_code_chunks(
            len(self.table.inputs),
            slots_per_chunk=resolved_slots,
        )

    def logic_value(self, value: Any) -> int:
        return int(float(value) > self.table.threshold)

    def task_metadata(
        self,
        task_kind: str,
        *,
        measurements: Iterable[str],
        stimulus: Mapping[str, object] | None = None,
        coverage: Mapping[str, object] | None = None,
        transient: Mapping[str, object] | None = None,
    ) -> dict:
        metadata = dict(self.table.metadata)
        payload = self.digital_task_payload(task_kind, measurements=measurements)
        if stimulus is not None:
            payload["stimulus"] = dict(stimulus)
        if coverage is not None:
            payload["coverage"] = dict(coverage)
        if transient is not None:
            payload["transient"] = dict(transient)
        metadata[MONATA_METADATA_KEY] = _monata_metadata_with_digital_task(
            metadata.get(MONATA_METADATA_KEY),
            payload,
        )
        return metadata

    def digital_task_payload(self, task_kind: str, *, measurements: Iterable[str]) -> dict:
        """Return the versioned digital control payload for tests or custom task assembly."""

        table = self.table
        return {
            "schema": DIGITAL_TASK_METADATA_SCHEMA,
            "measurements": list(measurements),
            "digital_verification": {
                "task_kind": task_kind,
                "dut": table.dut_name,
                "inputs": list(table.inputs),
                "outputs": list(table.outputs),
                "vectors": len(self.combinations()),
                "vdd": table.vdd,
                "threshold": table.threshold,
            },
        }


def sim_result_digital_task_metadata(sim_result: SimResult) -> Mapping[str, Any]:
    return digital_task_metadata(sim_result.metadata)


def sim_result_has_digital_task_metadata(sim_result: SimResult) -> bool:
    namespace = sim_result.metadata.get(MONATA_METADATA_KEY)
    return isinstance(namespace, Mapping) and DIGITAL_TASK_METADATA_KEY in namespace


def digital_task_metadata(task_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    namespace = task_metadata.get(MONATA_METADATA_KEY)
    if not isinstance(namespace, Mapping):
        raise RuntimeError("digital task metadata requires metadata['monata']")
    if DIGITAL_TASK_METADATA_KEY not in namespace:
        raise RuntimeError("digital task metadata missing metadata['monata']['digital_task_v1']")
    payload = namespace[DIGITAL_TASK_METADATA_KEY]
    if not isinstance(payload, Mapping):
        raise RuntimeError("digital task metadata payload must be a mapping")
    schema = payload.get("schema")
    if schema != DIGITAL_TASK_METADATA_SCHEMA:
        raise RuntimeError(f"unsupported digital task metadata schema: {schema}")
    return payload


def digital_stimulus_metadata(
    sim_result: SimResult,
    *,
    default_cycles_per_vector: int = 1,
    default_slot_duration: float | None = None,
) -> DigitalStimulusMetadata:
    task_metadata = sim_result_digital_task_metadata(sim_result)
    raw_stimulus = task_metadata.get("stimulus")
    if not isinstance(raw_stimulus, Mapping):
        raise RuntimeError("digital task metadata missing stimulus")
    stimulus = cast(Mapping[str, object], raw_stimulus)
    return DigitalStimulusMetadata(
        kind=str(stimulus.get("kind", "")),
        initial_settle=_metadata_float(stimulus, "initial_settle", 0.0),
        clock_period=_metadata_float(stimulus, "clock_period", 0.0),
        chunk_index=_metadata_int(stimulus, "chunk_index", 0),
    )


def _metadata_int(metadata: Mapping[str, object], name: str, default: int) -> int:
    try:
        return int(cast(float | int | str, metadata.get(name, default)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"digital stimulus metadata field {name!r} must be an integer") from exc


def _metadata_float(metadata: Mapping[str, object], name: str, default: float) -> float:
    try:
        return float(cast(float | int | str, metadata.get(name, default)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"digital stimulus metadata field {name!r} must be numeric") from exc


def _monata_metadata_with_digital_task(value: object, payload: Mapping[str, object]) -> dict[str, object]:
    if value is None:
        namespace: dict[str, object] = {}
    elif isinstance(value, Mapping):
        namespace = dict(value)
    else:
        raise ValueError("metadata['monata'] is reserved for Monata namespace mappings")
    namespace[DIGITAL_TASK_METADATA_KEY] = dict(payload)
    return namespace
