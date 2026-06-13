from __future__ import annotations

from monata.sim.results import SimResult

__all__ = ["assert_failed_result", "assert_ok_result"]


def assert_ok_result(result: SimResult) -> None:
    assert result.status == "ok", result.error_message


def assert_failed_result(result: SimResult, *, reason: str | None = None) -> None:
    assert result.status == "failed"
    if reason is not None:
        assert result.metadata["reason"] == reason
