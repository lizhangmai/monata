"""Executor abstraction — dispatch SimTasks to backends."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from threading import Lock

from monata.sim.results import SimResult
from monata.sim.task import DEFAULT_SIMULATOR, SimTask


class Executor(ABC):
    @abstractmethod
    def submit(self, task: SimTask) -> Future:
        ...

    @abstractmethod
    def map(self, tasks: list[SimTask]) -> list[Future]:
        ...


class LocalExecutor(Executor):
    """Local thread executor.

    Args:
        max_workers: Number of parallel workers. Defaults to os.cpu_count().
        backend: Optional default simulator backend for tasks using the
            SimTask default.
    """

    def __init__(self, max_workers: int | None = None, backend: str | None = None):
        self._max_workers = max_workers
        self._backend = str(backend) if backend is not None else None
        self._executor: ThreadPoolExecutor | None = None
        self._pending = 0
        self._lock = Lock()

    def submit(self, task: SimTask) -> Future:
        executor = self._acquire_executor(1)
        future = executor.submit(self._execute, task)
        future.add_done_callback(self._release_executor)
        return future

    def map(self, tasks: list[SimTask]) -> list[Future]:
        if not tasks:
            return []
        executor = self._acquire_executor(len(tasks))
        futures = [executor.submit(self._execute, t) for t in tasks]
        for future in futures:
            future.add_done_callback(self._release_executor)
        return futures

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
            self._pending = 0
        if executor is not None:
            executor.shutdown(wait=wait)

    def _execute(self, task: SimTask) -> SimResult:
        from monata.sim._backend import execute_task

        return execute_task(_task_with_default_backend(task, self._backend))

    def run_monte_carlo_native(self, monte_carlo):
        """Execute Monte Carlo through the local native executor capability."""

        return monte_carlo.run_task_expanded(self, mode="native")

    def _acquire_executor(self, task_count: int) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self._worker_count())
            self._pending += task_count
            return self._executor

    def _release_executor(self, _future: Future) -> None:
        executor = None
        with self._lock:
            if self._pending <= 0:
                return
            self._pending -= 1
            if self._pending == 0:
                executor = self._executor
                self._executor = None
        if executor is not None:
            executor.shutdown(wait=False)

    def _worker_count(self) -> int:
        return self._max_workers or os.cpu_count() or 4


def _task_with_default_backend(task: SimTask, backend: str | None) -> SimTask:
    if backend is None or task.simulator != DEFAULT_SIMULATOR:
        return task
    return replace(task, simulator=backend)
