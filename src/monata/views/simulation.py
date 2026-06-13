from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from monata.sim.core import LocalExecutor, SimArtifactOptions, SimResult, SimTask
from monata.views.base import View

SimulationProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class SimulationViewResult:
    """Structured result returned by a first-class simulation view."""

    cell: str
    view_type: str
    mode: str
    results: tuple[SimResult, ...]
    metadata: dict[str, Any]

    @property
    def status(self) -> str:
        return "ok" if all(result.status == "ok" for result in self.results) else "failed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "view_type": self.view_type,
            "mode": self.mode,
            "status": self.status,
            "result_count": len(self.results),
            "metadata": dict(self.metadata),
        }


class SimulationView(View):
    """First-class view boundary for simulation task execution."""

    def __init__(
        self,
        cell,
        entry: str,
        function_name: str = "main",
        *,
        backend: str | None = None,
        max_workers: int | None = None,
        generated: bool = False,
    ):
        super().__init__(view_type="simulation", cell=cell, entry=entry, generated=generated)
        self._function_name = function_name
        self._backend = backend
        self._max_workers = max_workers

    def load(self):
        return self.load_python_attribute("simulation", self._function_name)

    def run(
        self,
        *,
        task: SimTask | None = None,
        tasks: Iterable[SimTask] | None = None,
        executor=None,
        mode: str = "simulation",
        metadata: dict[str, Any] | None = None,
        artifact_dir: str | Path | None = None,
        progress: SimulationProgressCallback | None = None,
        **kwargs: Any,
    ) -> SimulationViewResult | Any:
        if task is not None and tasks is not None:
            raise ValueError("provide only one of task or tasks")
        if task is not None:
            return self._result(
                (self.run_task(task, executor=executor, artifact_dir=artifact_dir, progress=progress),),
                mode=mode,
                metadata=metadata,
            )
        if tasks is not None:
            return self._result(
                tuple(self.run_tasks(tasks, executor=executor, artifact_dir=artifact_dir, progress=progress)),
                mode=mode,
                metadata=metadata,
            )

        func = self.load()
        produced = func(self.cell, **kwargs)
        if isinstance(produced, SimTask):
            return self._result(
                (self.run_task(produced, executor=executor, artifact_dir=artifact_dir, progress=progress),),
                mode=mode,
                metadata=metadata,
            )
        produced_tasks = _as_sim_tasks(produced)
        if produced_tasks is not None:
            return self._result(
                tuple(
                    self.run_tasks(
                        produced_tasks,
                        executor=executor,
                        artifact_dir=artifact_dir,
                        progress=progress,
                    )
                ),
                mode=mode,
                metadata=metadata,
            )
        return produced

    def run_task(
        self,
        task: SimTask,
        *,
        executor=None,
        artifact_dir: str | Path | None = None,
        progress: SimulationProgressCallback | None = None,
    ) -> SimResult:
        return self.run_tasks((task,), executor=executor, artifact_dir=artifact_dir, progress=progress)[0]

    def run_tasks(
        self,
        tasks: Iterable[SimTask],
        *,
        executor=None,
        artifact_dir: str | Path | None = None,
        progress: SimulationProgressCallback | None = None,
    ) -> tuple[SimResult, ...]:
        task_list = list(tasks)
        if not task_list:
            return ()
        task_list = _with_artifact_dirs(task_list, artifact_dir)
        resolved_executor = executor or LocalExecutor(max_workers=self._max_workers, backend=self._backend)
        _emit_progress(progress, event="tasks_start", completed=0, total=len(task_list))
        if len(task_list) == 1:
            future = resolved_executor.submit(task_list[0])
            try:
                result = future.result()
            except Exception as exc:
                _emit_progress(
                    progress,
                    event="task_error",
                    completed=0,
                    total=1,
                    task_index=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            _emit_progress(progress, event="task_done", completed=1, total=1, task_index=0, result=result)
            _emit_progress(progress, event="tasks_done", completed=1, total=1)
            return (result,)

        futures = list(resolved_executor.map(task_list))
        future_indices = {future: index for index, future in enumerate(futures)}
        results: list[SimResult | None] = [None] * len(futures)
        completed = 0
        for future in as_completed(futures):
            index = future_indices[future]
            try:
                result = future.result()
            except Exception as exc:
                _emit_progress(
                    progress,
                    event="task_error",
                    completed=completed,
                    total=len(futures),
                    task_index=index,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            results[index] = result
            completed += 1
            _emit_progress(
                progress,
                event="task_done",
                completed=completed,
                total=len(futures),
                task_index=index,
                result=result,
            )
        _emit_progress(progress, event="tasks_done", completed=completed, total=len(futures))
        return tuple(_require_result(result) for result in results)

    def _result(
        self,
        results: tuple[SimResult, ...],
        *,
        mode: str,
        metadata: dict[str, Any] | None,
    ) -> SimulationViewResult:
        qualified = getattr(self.cell, "qualified_name", getattr(self.cell, "name", "<unknown>"))
        return SimulationViewResult(
            cell=qualified,
            view_type=self.view_type,
            mode=mode,
            results=results,
            metadata=dict(metadata or {}),
        )


def _as_sim_tasks(value: object) -> tuple[SimTask, ...] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if not items or not all(isinstance(item, SimTask) for item in items):
        return None
    return tuple(items)


def _emit_progress(
    progress: SimulationProgressCallback | None,
    *,
    event: str,
    completed: int,
    total: int,
    task_index: int | None = None,
    result: SimResult | None = None,
    error: str | None = None,
) -> None:
    if progress is None:
        return
    payload: dict[str, Any] = {
        "event": event,
        "completed": completed,
        "total": total,
    }
    if task_index is not None:
        payload["task_index"] = task_index
    if result is not None:
        payload.update(_result_progress_payload(result))
    if error is not None:
        payload["error"] = error
    progress(payload)


def _result_progress_payload(result: SimResult) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": result.status}
    if result.error_message is not None:
        payload["error"] = result.error_message
    metadata = dict(result.metadata)
    task_metadata = metadata.get("task_metadata")
    if isinstance(task_metadata, Mapping):
        payload["task_metadata"] = dict(task_metadata)
    artifacts = metadata.get("artifacts")
    if isinstance(artifacts, Mapping):
        payload["artifacts"] = dict(artifacts)
    return payload


def _require_result(result: SimResult | None) -> SimResult:
    if result is None:
        raise RuntimeError("simulation task completed without a result")
    return result


def _with_artifact_dirs(tasks: list[SimTask], artifact_dir: str | Path | None) -> list[SimTask]:
    if artifact_dir is None:
        return tasks
    root = Path(artifact_dir)
    start_index = _next_artifact_index(root)
    seen: set[Path] = set()
    assigned = []
    for index, task in enumerate(tasks):
        artifact_index = start_index + index
        target = _dedupe_artifact_path(root / _artifact_relative_path(task, artifact_index), seen)
        target.mkdir(parents=True, exist_ok=True)
        metadata = dict(task.metadata)
        metadata["simulation_artifact_index"] = artifact_index
        assigned.append(
            replace(
                task,
                metadata=metadata,
                artifacts=SimArtifactOptions(target, overwrite=task.artifacts.overwrite),
            )
        )
    return assigned


def _artifact_relative_path(task: SimTask, index: int) -> Path:
    del task
    return Path("tasks") / f"task-{index:04d}"


def _next_artifact_index(root: Path) -> int:
    tasks_dir = root / "tasks"
    if not tasks_dir.is_dir():
        return 0
    observed = []
    for child in tasks_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("task-"):
            continue
        suffix = child.name.removeprefix("task-")
        if suffix.isdigit():
            observed.append(int(suffix))
    if not observed:
        return 0
    return max(observed) + 1


def _dedupe_artifact_path(path: Path, seen: set[Path]) -> Path:
    candidate = path
    suffix = 2
    while candidate in seen or candidate.exists():
        candidate = path.with_name(f"{path.name}-{suffix:02d}")
        suffix += 1
    seen.add(candidate)
    return candidate
