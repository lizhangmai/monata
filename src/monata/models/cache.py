"""Content-hash-based compilation cache for Verilog-A models."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from uuid import uuid4
from typing import Any

from monata._home import monata_cache_dir
from monata.models.diagnostics import ModelDiagnostic, ModelDiagnosticError

_logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = None
CACHE_MARKER_FILENAME = ".monata-model-cache"


def resolve_model_cache_dir(
    *,
    explicit: str | Path | None = None,
    project_config: str | Path | None = None,
    home: str | Path | None = None,
) -> Path:
    """Resolve the writable model cache directory by Monata precedence rules."""

    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("MONATA_MODEL_CACHE")
    if env:
        return Path(env)
    if project_config is not None and str(project_config) != "auto":
        return Path(project_config)
    return monata_cache_dir(home=home) / "models"


def default_cache_dir() -> Path:
    """Return the default model cache directory under MONATA_HOME."""
    global _DEFAULT_CACHE_DIR
    if _DEFAULT_CACHE_DIR is None:
        _DEFAULT_CACHE_DIR = resolve_model_cache_dir()
    return _DEFAULT_CACHE_DIR


def reset_default_cache_dir_for_tests() -> None:
    """Reset process-level cache path memoization for precedence tests."""

    global _DEFAULT_CACHE_DIR
    _DEFAULT_CACHE_DIR = None


class ModelCache:
    """Hash-based cache that avoids recompiling unchanged .va sources.

    Cache layout:
        <cache_dir>/
            <sha256_hex>/
                model.osdi       (or other output)
                meta.json        (source path, timestamp, compiler version)
    """

    def __init__(self, cache_dir=None, *, namespace: str | None = None):
        self._dir = Path(cache_dir) if cache_dir else default_cache_dir()
        if namespace:
            self._dir = self._dir / namespace
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ensure_marker_for_new_cache()

    @property
    def path(self) -> Path:
        return self._dir

    @property
    def marker_path(self) -> Path:
        return self._dir / CACHE_MARKER_FILENAME

    def _ensure_marker_for_new_cache(self) -> None:
        if self.marker_path.exists():
            return
        if any(self._dir.iterdir()):
            return
        self.marker_path.write_text("monata model cache\n")

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _hash_files(paths: list[Path]) -> str:
        """Hash multiple files and their resolved identities in sorted order."""
        h = hashlib.sha256()
        for p in sorted(paths, key=lambda path: str(path)):
            path_bytes = str(p).encode()
            h.update(len(path_bytes).to_bytes(8, "big"))
            h.update(path_bytes)
            h.update(p.stat().st_size.to_bytes(8, "big"))
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _cache_hash(source_hash: str, context: dict[str, Any] | None = None) -> str:
        """Hash source identity plus optional toolchain/resolution context."""

        normalized = _normalize_context(context)
        if not normalized:
            return source_hash
        payload = json.dumps(
            {
                "schema": "monata-model-cache-key-v1",
                "source_hash": source_hash,
                "context": normalized,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _input_files(va_path: Path, include_paths=None) -> list[Path]:
        files = [va_path] + [Path(p) for p in (include_paths or [])]
        missing = [p for p in files if not p.exists()]
        if missing:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="model_source_missing",
                    message="model source or include file not found",
                    context={
                        "source": str(va_path),
                        "missing": [str(path) for path in missing],
                    },
                )
            )
        return [p.resolve() for p in files]

    def _entry_dir(self, content_hash: str) -> Path:
        return self._dir / content_hash[:2] / content_hash[2:]

    def lookup(self, va_path, include_paths=None, *, context: dict[str, Any] | None = None) -> Path | None:
        """Check if a compiled .osdi exists in cache for the given source.

        Args:
            va_path: Path to the .va source file.
            include_paths: Additional files that affect compilation (headers).
            context: Optional resolution/toolchain identity.

        Returns:
            Path to cached .osdi if hit, None if miss.
        """
        va_path = Path(va_path)
        files = self._input_files(va_path, include_paths=include_paths)
        source_hash = self._hash_files(files)
        content_hash = self._cache_hash(source_hash, context)
        entry = self._entry_dir(content_hash)
        osdi = self._cached_artifact_path(entry, va_path.stem + ".osdi", content_hash)
        if osdi is not None:
            _logger.debug("Cache hit: %s → %s", va_path.name, osdi)
            return osdi
        return None

    def lookup_compatible(
        self,
        va_path,
        include_paths=None,
        *,
        required_context: dict[str, Any] | None = None,
    ) -> Path | None:
        """Return a compatible cached OSDI without requiring current compiler identity."""

        va_path = Path(va_path)
        files = self._input_files(va_path, include_paths=include_paths)
        source_hash = self._hash_files(files)
        expected_artifact_name = va_path.stem + ".osdi"
        required = _normalize_context(required_context)
        for meta_path in sorted(self._dir.rglob("meta.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("source_hash", meta.get("hash")) != source_hash:
                continue
            context = _normalize_context(meta.get("context"))
            if not _context_contains(context, required):
                continue
            osdi = self._cached_artifact_path(
                meta_path.parent,
                str(meta.get("artifact_name") or expected_artifact_name),
                str(meta.get("hash") or ""),
            )
            if osdi is not None:
                _logger.debug("Compatible cache hit: %s → %s", va_path.name, osdi)
                return osdi
        return None

    def require_cached(self, va_path, include_paths=None, *, context: dict[str, Any] | None = None) -> Path:
        """Return a cached artifact or raise a stable diagnostic."""
        cached = self.lookup(va_path, include_paths=include_paths, context=context)
        if cached is None:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="model_cache_missing",
                    message="compiled model artifact is missing or stale",
                    context={
                        "source": str(Path(va_path)),
                        "includes": [str(p) for p in (include_paths or [])],
                    },
                )
            )
        return cached

    def store(self, va_path, osdi_path, include_paths=None, *, context: dict[str, Any] | None = None) -> Path:
        """Store a compiled .osdi in the cache.

        Args:
            va_path: Original .va source path.
            osdi_path: Path to the compiled .osdi to cache.
            include_paths: Additional source files used in compilation.
            context: Optional resolution/toolchain identity.

        Returns:
            Path to the cached copy.
        """
        va_path = Path(va_path)
        osdi_path = Path(osdi_path)
        files = self._input_files(va_path, include_paths=include_paths)
        source_hash = self._hash_files(files)
        normalized_context = _normalize_context(context)
        content_hash = self._cache_hash(source_hash, normalized_context)
        entry = self._entry_dir(content_hash)
        cached = entry / osdi_path.name
        if self._cached_artifact_path(entry, osdi_path.name, content_hash) is not None:
            return cached
        meta = {
            "source": str(va_path),
            "hash": content_hash,
            "source_hash": source_hash,
            "artifact_name": osdi_path.name,
            "includes": [str(path.resolve()) for path in (include_paths or [])],
            "inputs": [str(path) for path in files],
            "context": normalized_context,
        }
        entry.parent.mkdir(parents=True, exist_ok=True)
        temp = entry.parent / f".{entry.name}.tmp-{os.getpid()}-{uuid4().hex}"
        temp.mkdir()
        try:
            shutil.copy2(osdi_path, temp / osdi_path.name)
            (temp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
            try:
                temp.rename(entry)
            except FileExistsError:
                shutil.rmtree(temp)
            else:
                temp = None
        finally:
            if temp is not None and temp.exists():
                shutil.rmtree(temp)
        if self._cached_artifact_path(entry, osdi_path.name, content_hash) is None:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="model_cache_store_failed",
                    message="compiled model artifact was not committed to cache",
                    context={"source": str(va_path), "cache_entry": str(entry)},
                )
            )
        _logger.debug("Cached: %s → %s", va_path.name, cached)
        return cached

    @staticmethod
    def _cached_artifact_path(entry: Path, artifact_name: str, content_hash: str) -> Path | None:
        osdi = entry / artifact_name
        meta_path = entry / "meta.json"
        if not osdi.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if content_hash and meta.get("hash") != content_hash:
            return None
        return osdi

    def clear(self):
        """Remove all cached compilations."""
        if self._dir.exists():
            self._validate_clear_target()
            shutil.rmtree(self._dir)
            self._dir.mkdir(parents=True, exist_ok=True)
            self.marker_path.write_text("monata model cache\n")
            _logger.info("Cache cleared: %s", self._dir)

    def size(self) -> int:
        """Return total size of cache in bytes."""
        total = 0
        for f in self._dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def _validate_clear_target(self) -> None:
        resolved = self._dir.resolve()
        dangerous = {
            Path(resolved.anchor).resolve(),
            Path.home().resolve(),
            Path.cwd().resolve(),
        }
        if self._dir.is_symlink() or resolved in dangerous:
            raise _unsafe_cache_clear(self._dir, "cache path is not a safe clear target")
        if not self.marker_path.is_file():
            raise _unsafe_cache_clear(
                self._dir,
                f"cache marker {CACHE_MARKER_FILENAME!r} is missing",
            )


def _unsafe_cache_clear(path: Path, reason: str) -> ModelDiagnosticError:
    return ModelDiagnosticError(
        ModelDiagnostic(
            code="unsafe_cache_clear",
            message="refusing to clear model cache directory",
            context={
                "path": str(path),
                "reason": reason,
                "required_marker": CACHE_MARKER_FILENAME,
            },
        )
    )


def _normalize_context(context: dict[str, Any] | None) -> dict[str, Any]:
    return json.loads(json.dumps(context or {}, sort_keys=True, default=str))


def _context_contains(context: dict[str, Any], required: dict[str, Any]) -> bool:
    for key, value in required.items():
        if key not in context:
            return False
        observed = context[key]
        if isinstance(value, dict) and isinstance(observed, dict):
            if not _context_contains(observed, value):
                return False
        elif observed != value:
            return False
    return True
