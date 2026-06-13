"""Native simulation backend dispatcher."""

from __future__ import annotations

from monata.sim.backends.base import backend_failure_result, validate_backend_task
from monata.sim.results import SimResult
from monata.sim.task import SimTask


def execute_task(task: SimTask) -> SimResult:
    """Execute a SimTask using the configured simulator backend."""
    from monata.sim.backends import get_backend

    try:
        backend = get_backend(task.simulator)
    except KeyError:
        return backend_failure_result(
            task,
            f"unknown simulator backend: {task.simulator}",
            metadata={
                "simulator": task.simulator,
                "reason": "unknown_simulator",
            },
        )
    preflight = validate_backend_task(backend, task)
    if preflight is not None:
        return preflight
    return backend.run(task)
