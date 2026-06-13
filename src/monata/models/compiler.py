"""Verilog-A model compiler — wraps OpenVAF and ADMS toolchains."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from monata.models.diagnostics import ModelDiagnostic, ModelDiagnosticError

_logger = logging.getLogger(__name__)


class CompilationError(RuntimeError):
    def __init__(self, source: Path, backend: str, message: str):
        self.source = source
        self.backend = backend
        super().__init__(f"[{backend}] Failed to compile {source.name}: {message}")


class ModelCompiler:
    """Compile Verilog-A (.va) sources into loadable binaries.

    Supported backends:
        openvaf  — produces .osdi (ngspice, VACASK, Xyce)
        adms     — produces C code, then .so via Xyce plugin build system
    """

    def __init__(self, openvaf_bin=None, adms_bin=None, timeout: float | int | None = 300.0):
        found = openvaf_bin or os.environ.get("OPENVAF_BIN") or shutil.which("openvaf-r") or shutil.which("openvaf")
        self._openvaf = str(Path(found).resolve()) if found else None
        found = adms_bin or shutil.which("admsXml")
        self._adms = str(Path(found).resolve()) if found else None
        self.timeout = _coerce_timeout(timeout)

    @property
    def has_openvaf(self) -> bool:
        return self._openvaf is not None

    @property
    def openvaf_path(self) -> str | None:
        return self._openvaf

    @property
    def has_adms(self) -> bool:
        return self._adms is not None

    def openvaf_identity(self) -> dict[str, Any]:
        """Return stable identity fields for cache invalidation."""

        if not self._openvaf:
            return {"backend": "openvaf", "available": False}
        path = Path(self._openvaf)
        return {
            "backend": "openvaf",
            "available": True,
            "path": str(path),
            "executable_sha256": _file_sha256(path) if path.is_file() else None,
            "version": _compiler_version(path, self.timeout),
        }

    def compile_osdi(self, va_path, output_dir=None, extra_args=None, include_paths=None) -> Path:
        """Compile a .va file to .osdi using OpenVAF.

        Args:
            va_path: Path to the Verilog-A source file.
            output_dir: Directory for the .osdi output. Defaults to same dir as source.
            extra_args: Additional CLI arguments for openvaf.
            include_paths: Source/include dependencies that affect compilation.

        Returns:
            Path to the produced .osdi file.
        """
        va_path = Path(va_path).resolve()
        if not va_path.exists():
            raise FileNotFoundError(f"Source not found: {va_path}")
        if not self._openvaf:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="compiler_missing",
                    message="OpenVAF not found in PATH. Install openvaf-r or set openvaf_bin.",
                    context={"backend": "openvaf", "source": str(va_path)},
                )
            )

        output_dir = Path(output_dir) if output_dir else va_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        osdi_name = va_path.stem + ".osdi"
        osdi_path = output_dir / osdi_name

        cmd = [self._openvaf]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend([str(va_path), "-o", str(osdi_path)])
        for include_path in include_paths or ():
            cmd.extend(["-I", str(Path(include_path).resolve().parent)])

        _logger.info("Compiling %s → %s", va_path.name, osdi_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise _compiler_timeout("openvaf", va_path, self.timeout, exc) from exc
        if result.returncode != 0:
            raise CompilationError(va_path, "openvaf", result.stderr.strip())

        return osdi_path

    def compile_adms(self, va_path, output_dir=None) -> Path:
        """Run ADMS on a .va file to produce C code.

        This is the first step of the Xyce plugin build pipeline.
        The output is a directory containing generated .c/.h files.

        Args:
            va_path: Path to the Verilog-A source file.
            output_dir: Directory for generated C output. Defaults to <source_dir>/adms_out/.

        Returns:
            Path to the output directory containing generated files.
        """
        va_path = Path(va_path).resolve()
        if not va_path.exists():
            raise FileNotFoundError(f"Source not found: {va_path}")
        if not self._adms:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="compiler_missing",
                    message="admsXml not found in PATH. Install adms or set adms_bin.",
                    context={"backend": "adms", "source": str(va_path)},
                )
            )

        output_dir = Path(output_dir) if output_dir else va_path.parent / "adms_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [self._adms, str(va_path), "-I", str(va_path.parent)]
        _logger.info("Running ADMS on %s → %s", va_path.name, output_dir)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir), timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise _compiler_timeout("adms", va_path, self.timeout, exc) from exc
        if result.returncode != 0:
            raise CompilationError(va_path, "adms", result.stderr.strip())

        return output_dir

    def compile(self, va_path, output_dir=None, backend="openvaf", **kwargs) -> Path:
        """Unified compile entry point.

        Args:
            va_path: Path to the Verilog-A source.
            output_dir: Output directory.
            backend: "openvaf" (default) or "adms".

        Returns:
            Path to the compilation output (.osdi file or adms output dir).
        """
        if backend == "openvaf":
            return self.compile_osdi(
                va_path,
                output_dir,
                extra_args=kwargs.get("extra_args"),
                include_paths=kwargs.get("include_paths"),
            )
        elif backend == "adms":
            return self.compile_adms(va_path, output_dir)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'openvaf' or 'adms'.")


def _coerce_timeout(timeout: float | int | None) -> float | None:
    if timeout is None:
        return None
    seconds = float(timeout)
    if seconds <= 0:
        raise ValueError("compiler timeout must be positive or None")
    return seconds


def _compiler_timeout(
    backend: str,
    source: Path,
    timeout: float | None,
    exc: subprocess.TimeoutExpired,
) -> ModelDiagnosticError:
    timeout_text = f" after {timeout:g} seconds" if timeout is not None else ""
    return ModelDiagnosticError(
        ModelDiagnostic(
            code="compiler_timeout",
            message=f"{backend} timed out{timeout_text}",
            context={
                "backend": backend,
                "source": str(source),
                "timeout_seconds": timeout,
                "stdout": _process_text(getattr(exc, "stdout", None) or getattr(exc, "output", None)),
                "stderr": _process_text(getattr(exc, "stderr", None)),
            },
        )
    )


def _process_text(value: str | bytes | None, limit: int = 4000) -> str:
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = value or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compiler_version(path: Path, timeout: float | None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=min(timeout, 2.0) if timeout is not None else 2.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": _process_text(result.stdout, limit=400),
        "stderr": _process_text(result.stderr, limit=400),
    }
