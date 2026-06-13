from __future__ import annotations

import numpy as np

from monata.sim.analysis_spec import analysis_name
from monata.sim.backends import BackendCapabilities, BackendTaskPlan, unregister_backend
from monata.sim.results import SimResult
from monata.sim.task import SimTask

from support.results import failed_result, sim_result

__all__ = [
    "DummyBackend",
    "MinimalBackend",
    "PlannedBackend",
    "cleanup_backends",
    "cleanup_test_backends",
]


class DummyBackend:
    name = "dummy-backend"
    capabilities = BackendCapabilities(
        analyses=("tran",),
        result_modes=("table",),
        source_mutations=("output_names",),
        model_artifacts=("include",),
        native_monte_carlo=False,
        renderer_contract="dummy-renderer",
        runner_contract="dummy-runner",
        parser_contract="dummy-parser",
    )

    def run(self, task: SimTask) -> SimResult:
        return sim_result(
            waveforms={"out": np.array([1.0, 2.0])},
            sweep_var=np.array([0.0, 1.0]),
            corner=task.corner,
            metadata={"simulator": self.name},
        )


class PlannedBackend(DummyBackend):
    name = "planned-backend"

    def validate_task(self, task: SimTask) -> SimResult | None:
        if analysis_name(task.analysis_spec) == "tran":
            return None
        return failed_result(
            corner=task.corner,
            metadata={"simulator": self.name, "reason": "custom_validation"},
            error_message="custom validation failed",
        )

    def plan_task(self, task: SimTask) -> BackendTaskPlan:
        return BackendTaskPlan(
            backend_name=self.name,
            analysis_name=analysis_name(task.analysis_spec),
            output_names=task.output_names,
            output_vectors=tuple(f"v({name})" for name in task.output_names),
            metadata={"planned": True},
        )


class MinimalBackend:
    name = "minimal-backend"
    capabilities = BackendCapabilities()

    def run(self, task: SimTask) -> SimResult:
        return sim_result(
            waveforms={},
            sweep_var=None,
            corner=task.corner,
            metadata={"simulator": self.name},
        )


def cleanup_backends(*names: str) -> None:
    for name in names:
        unregister_backend(name)


def cleanup_test_backends() -> None:
    cleanup_backends(DummyBackend.name, PlannedBackend.name, MinimalBackend.name)
