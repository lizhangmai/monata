from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future

from monata.sim.results import SimResult
from monata.sim.task import SimTask

from support.results import sim_result

ResultFactory = Callable[[SimTask, int], SimResult]

__all__ = [
    "CapturingExecutor",
    "ImmediateExecutor",
    "RecordingExecutor",
    "ResultFactory",
    "completed_future",
]


def completed_future(result: SimResult) -> Future[SimResult]:
    future: Future[SimResult] = Future()
    future.set_result(result)
    return future


class ImmediateExecutor:
    def __init__(self, result: SimResult):
        self.result = result
        self.task: SimTask | None = None

    def submit(self, task: SimTask) -> Future[SimResult]:
        self.task = task
        return completed_future(self.result)


class RecordingExecutor:
    def __init__(self, result_factory: ResultFactory | None = None):
        self.result_factory = result_factory or self._default_result
        self.tasks: list[SimTask] = []
        self.called = False

    def map(self, tasks: Iterable[SimTask]) -> list[Future[SimResult]]:
        self.called = True
        self.tasks = list(tasks)
        return [completed_future(self.result_factory(task, index)) for index, task in enumerate(self.tasks)]

    @staticmethod
    def _default_result(task: SimTask, index: int) -> SimResult:
        return sim_result(
            waveforms={},
            sweep_var=None,
            corner=task.corner,
            metadata=task.metadata,
        )


class CapturingExecutor(RecordingExecutor):
    pass
