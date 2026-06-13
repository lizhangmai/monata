"""Content-addressed cache helpers for simulation results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from typing import Any

import numpy as np

from monata._json import json_safe
from monata.corner import corner_to_payload
from monata.netlist import render_ngspice
from monata.sim.export import sim_result_from_dict, sim_result_to_dict
from monata.sim.results import SimResult, SimStatus
from monata.sim.task import SimTask

CACHE_SCHEMA = "monata.sim.cache.v1"
TASK_FINGERPRINT_SCHEMA = "monata.sim.task-fingerprint.v1"

_CACHE_KEY_RE = re.compile(r"^[a-f0-9]{64}$")
_SPICE_ARTIFACT_COMMANDS = {".include": "include", ".lib": "lib"}


@dataclass(frozen=True)
class SourceArtifactDigest:
    """Content digest for a model/include/OSDI artifact referenced by a task."""

    kind: str
    path: str
    exists: bool
    sha256: str | None = None
    size: int | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
            "exists": self.exists,
            "sha256": self.sha256,
            "size": self.size,
            "source": self.source,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class SimTaskFingerprint:
    """Stable, backend-neutral identity for a simulation task input."""

    key: str
    payload: Mapping[str, Any]
    artifacts: tuple[SourceArtifactDigest, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "payload": dict(self.payload),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


class SimulationCache:
    """Small filesystem cache for ``SimResult`` objects keyed by ``SimTask`` identity."""

    def __init__(self, path: str | Path | None = None, *, base_path: str | Path | None = None) -> None:
        self._directory = Path(path) if path is not None else Path(tempfile.mkdtemp(prefix="monata-sim-cache-"))
        self._base_path = Path(base_path) if base_path is not None else None
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._directory

    def fingerprint(
        self,
        task: SimTask,
        *,
        extra_artifacts: Iterable[str | Path] = (),
    ) -> SimTaskFingerprint:
        return task_fingerprint(task, base_path=self._base_path, extra_artifacts=extra_artifacts)

    def key_for_task(self, task: SimTask, *, extra_artifacts: Iterable[str | Path] = ()) -> str:
        return self.fingerprint(task, extra_artifacts=extra_artifacts).key

    def contains(self, task: SimTask, *, extra_artifacts: Iterable[str | Path] = ()) -> bool:
        return self._entry_path(self.key_for_task(task, extra_artifacts=extra_artifacts)).is_dir()

    def load(self, task: SimTask, *, extra_artifacts: Iterable[str | Path] = ()) -> SimResult | None:
        return self.load_key(self.key_for_task(task, extra_artifacts=extra_artifacts))

    def load_key(self, key: str) -> SimResult | None:
        entry = self._entry_path(_cache_key(key))
        result_path = entry / "result.json"
        if not result_path.exists():
            return None
        payload = json.loads(result_path.read_text())
        arrays_path = entry / "arrays.npz"
        if arrays_path.exists():
            arrays_file = np.load(arrays_path, allow_pickle=False)
            try:
                array_store = {name: arrays_file[name] for name in arrays_file.files}
            finally:
                arrays_file.close()
            return sim_result_from_dict(payload, array_store=array_store)
        return sim_result_from_dict(payload)

    def store(
        self,
        task: SimTask,
        result: SimResult,
        *,
        extra_artifacts: Iterable[str | Path] = (),
    ) -> Path:
        fingerprint = self.fingerprint(task, extra_artifacts=extra_artifacts)
        return self.store_key(fingerprint.key, result, fingerprint=fingerprint)

    def store_key(
        self,
        key: str,
        result: SimResult,
        *,
        fingerprint: SimTaskFingerprint | None = None,
    ) -> Path:
        cache_key = _cache_key(key)
        entry = self._entry_path(cache_key)
        tmp = self._directory / f".{cache_key}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)

        array_store: dict[str, np.ndarray] = {}
        payload = sim_result_to_dict(result, array_store=array_store)
        (tmp / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        if array_store:
            savez_compressed: Any = np.savez_compressed
            savez_compressed(tmp / "arrays.npz", **array_store)

        manifest = {
            "schema": CACHE_SCHEMA,
            "key": cache_key,
            "fingerprint": fingerprint.to_dict() if fingerprint is not None else None,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

        shutil.rmtree(entry, ignore_errors=True)
        tmp.rename(entry)
        return entry

    def run(
        self,
        task: SimTask,
        runner: Callable[[SimTask], SimResult],
        *,
        store_failed: bool = False,
        extra_artifacts: Iterable[str | Path] = (),
    ) -> SimResult:
        cached = self.load(task, extra_artifacts=extra_artifacts)
        if cached is not None:
            return cached
        result = runner(task)
        if result.status is SimStatus.OK or store_failed:
            self.store(task, result, extra_artifacts=extra_artifacts)
        return result

    def delete(self, task: SimTask, *, extra_artifacts: Iterable[str | Path] = ()) -> bool:
        entry = self._entry_path(self.key_for_task(task, extra_artifacts=extra_artifacts))
        if not entry.exists():
            return False
        shutil.rmtree(entry)
        return True

    def clear(self) -> None:
        for entry in self._directory.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    def _entry_path(self, key: str) -> Path:
        return self._directory / key


def task_fingerprint(
    task: SimTask,
    *,
    base_path: str | Path | None = None,
    extra_artifacts: Iterable[str | Path] = (),
) -> SimTaskFingerprint:
    """Return a stable fingerprint for task inputs that affect simulation output."""

    rendered_circuit = _render_task_circuit(task.circuit)
    artifacts = _task_artifacts(task, rendered_circuit, base_path=base_path, extra_artifacts=extra_artifacts)
    payload = {
        "schema": TASK_FINGERPRINT_SCHEMA,
        "simulator": task.simulator,
        "circuit": rendered_circuit,
        "analysis_spec": _analysis_spec_payload(task.analysis_spec),
        "corner": corner_to_payload(task.corner),
        "param_overrides": _value_payload(task.param_overrides),
        "output_names": list(task.output_names),
        "osdi_paths": [str(path) for path in task.osdi_paths],
        "backend_options": _value_payload(task.backend_options),
        "metadata": _value_payload(task.metadata),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }
    return SimTaskFingerprint(_sha256_text(_canonical_json(payload)), payload, artifacts)


def _render_task_circuit(circuit: Any) -> str:
    try:
        return render_ngspice(circuit)
    except TypeError:
        return str(circuit)


def _analysis_spec_payload(spec: Any) -> dict[str, Any]:
    return {
        "type": f"{type(spec).__module__}.{type(spec).__qualname__}",
        "fields": _value_payload(asdict(spec)) if is_dataclass(spec) and not isinstance(spec, type) else _value_payload(spec),
    }


def _task_artifacts(
    task: SimTask,
    rendered_circuit: str,
    *,
    base_path: str | Path | None,
    extra_artifacts: Iterable[str | Path],
) -> tuple[SourceArtifactDigest, ...]:
    base = Path(base_path) if base_path is not None else None
    artifacts: list[SourceArtifactDigest] = []
    seen: set[Path] = set()
    for kind, path in _spice_artifact_refs(rendered_circuit):
        artifacts.extend(_collect_artifact(kind, path, base_path=base, source=None, seen=seen, recursive=True))
    if task.corner is not None and task.corner.model_file is not None:
        artifacts.extend(
            _collect_artifact("model_file", task.corner.model_file, base_path=base, source="corner", seen=seen)
        )
    for path in task.osdi_paths:
        artifacts.extend(_collect_artifact("osdi", path, base_path=base, source="task.osdi_paths", seen=seen))
    for path in extra_artifacts:
        artifacts.extend(_collect_artifact("extra", path, base_path=base, source="extra_artifacts", seen=seen))
    return tuple(artifacts)


def _collect_artifact(
    kind: str,
    path: str | Path,
    *,
    base_path: Path | None,
    source: str | None,
    seen: set[Path],
    recursive: bool = False,
) -> list[SourceArtifactDigest]:
    raw_path = Path(path)
    resolved = _resolve_artifact_path(raw_path, base_path)
    if resolved in seen:
        return []
    seen.add(resolved)
    if not resolved.exists() or not resolved.is_file():
        return [SourceArtifactDigest(kind=kind, path=str(raw_path), exists=False, source=source)]

    data = resolved.read_bytes()
    artifacts = [
        SourceArtifactDigest(
            kind=kind,
            path=str(raw_path),
            exists=True,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            source=source,
        )
    ]
    if recursive:
        try:
            text = resolved.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return artifacts
        for child_kind, child_path in _spice_artifact_refs(text):
            artifacts.extend(
                _collect_artifact(
                    child_kind,
                    child_path,
                    base_path=resolved.parent,
                    source=str(raw_path),
                    seen=seen,
                    recursive=True,
                )
            )
    return artifacts


def _spice_artifact_refs(text: str) -> tuple[tuple[str, str], ...]:
    refs: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            tokens = line.split()
        if len(tokens) < 2:
            continue
        kind = _SPICE_ARTIFACT_COMMANDS.get(tokens[0].lower())
        if kind is not None:
            refs.append((kind, tokens[1]))
    return tuple(refs)


def _resolve_artifact_path(path: Path, base_path: Path | None) -> Path:
    if path.is_absolute():
        return path
    if base_path is None:
        return path.resolve()
    base = base_path.parent if base_path.is_file() else base_path
    return (base / path).resolve()


def _value_payload(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _value_payload(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _value_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list | tuple):
        return [_value_payload(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_value_payload(item) for item in sorted(value, key=str)]
    safe = json_safe(value)
    try:
        json.dumps(safe, sort_keys=True)
    except TypeError:
        return str(value)
    return safe


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(key: str) -> str:
    text = str(key)
    if not _CACHE_KEY_RE.fullmatch(text):
        raise ValueError("simulation cache key must be a 64-character lowercase sha256 hex digest")
    return text


__all__ = [
    "CACHE_SCHEMA",
    "TASK_FINGERPRINT_SCHEMA",
    "SimTaskFingerprint",
    "SimulationCache",
    "SourceArtifactDigest",
    "task_fingerprint",
]
