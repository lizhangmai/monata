from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from monata.sim.digital_table import DigitalVerificationSpec, ExpectedTable
from monata.views.base import View
from monata.views.path_safety import read_cell_json_mapping, resolve_cell_relative_path


class VerificationView(View):
    """Data-only Monata view for verification intent."""

    def __init__(
        self,
        cell,
        entry: str,
        *,
        generated: bool = False,
        view_format: str | None = "monata-verification-json",
        schema_version: int | None = 1,
    ):
        super().__init__(
            view_type="verification",
            cell=cell,
            entry=entry,
            generated=generated,
            format=view_format,
            schema_version=schema_version,
        )

    def read(self) -> DigitalVerificationSpec:
        return self.spec()

    def load(self) -> DigitalVerificationSpec:
        return self.spec()

    def spec(self) -> DigitalVerificationSpec:
        payload = read_cell_json_mapping(
            self.path(),
            self.entry,
            label="verification.entry",
        )
        _validate_schema_version(
            payload,
            self.schema_version,
            label="verification",
        )
        expected = _expected_table_from_spec_payload(self.path(), payload)
        return DigitalVerificationSpec.from_mapping(payload, expected=expected)


def _expected_table_from_spec_payload(root: Path, payload: Mapping[str, Any]) -> ExpectedTable:
    truth_table_measure = _truth_table_measure_payload(payload)
    expected_ref = truth_table_measure.get("expected")
    if not isinstance(expected_ref, Mapping):
        raise TypeError("truth_table measure expected must be an object")
    entry = expected_ref.get("entry")
    if not isinstance(entry, str):
        raise ValueError("truth_table measure expected.entry must be a string")
    path = resolve_cell_relative_path(root, entry, label="expected.entry")
    return ExpectedTable.from_json(path)


def _truth_table_measure_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    measures = payload.get("measures")
    if not isinstance(measures, list):
        raise TypeError("verification measures must be an array")
    for measure in measures:
        if isinstance(measure, Mapping) and measure.get("name") == "truth_table":
            return measure
    raise ValueError("verification measures require a truth_table measure")


def _validate_schema_version(
    payload: Mapping[str, Any],
    expected: int | None,
    *,
    label: str,
) -> None:
    if expected is None:
        return
    actual = payload.get("schema_version")
    if actual != expected:
        raise ValueError(f"{label} schema_version {actual!r} does not match view config {expected}")
