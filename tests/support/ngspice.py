from __future__ import annotations

import os
from pathlib import Path
from shutil import which

import pytest

__all__ = [
    "ngspice_available",
    "ngspice_bin_dirs",
    "put_ngspice_on_path",
    "skip_if_no_ngspice",
]


def ngspice_bin_dirs(test_file: str) -> list[Path]:
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin")
    project_root = Path(test_file).resolve().parents[3]
    candidates.extend([
        project_root / "test" / ".pixi" / "envs" / "monata-dev" / "bin",
        project_root / "test" / ".pixi" / "envs" / "default" / "bin",
    ])
    return candidates


def put_ngspice_on_path(monkeypatch: pytest.MonkeyPatch, test_file: str) -> None:
    for bin_dir in ngspice_bin_dirs(test_file):
        if (bin_dir / "ngspice").is_file():
            monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
            return


def ngspice_available() -> bool:
    return which("ngspice") is not None


def skip_if_no_ngspice() -> None:
    if not ngspice_available():
        pytest.skip("ngspice executable is not available")
