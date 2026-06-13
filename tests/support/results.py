from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from monata.sim.results import SimResult

__all__ = ["corner_results", "failed_result", "sim_result"]


class _MissingSweep:
    pass


_MISSING: Final = _MissingSweep()


def sim_result(
    *,
    status: str = "ok",
    waveforms: Mapping[str, np.ndarray] | None = None,
    sweep_var: np.ndarray | None | _MissingSweep = _MISSING,
    corner: Any = None,
    metadata: Mapping[str, Any] | None = None,
    error_message: str | None = None,
) -> SimResult:
    result_waveforms = {"out": np.array([1.0])} if waveforms is None else dict(waveforms)
    if isinstance(sweep_var, _MissingSweep):
        result_sweep_var = np.array([0.0]) if result_waveforms else None
    else:
        result_sweep_var = sweep_var
    return SimResult(
        status=status,
        waveforms=result_waveforms,
        sweep_var=result_sweep_var,
        corner=corner,
        metadata=dict(metadata or {}),
        error_message=error_message,
    )


def failed_result(
    *,
    corner: Any = None,
    metadata: Mapping[str, Any] | None = None,
    error_message: str = "fail",
) -> SimResult:
    return sim_result(
        status="failed",
        waveforms={},
        sweep_var=None,
        corner=corner,
        metadata=metadata,
        error_message=error_message,
    )


def corner_results(*corners: Any) -> list[SimResult]:
    return [sim_result(corner=corner) for corner in corners]
