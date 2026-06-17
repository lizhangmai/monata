from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import TextIO


_RECOMMENDED_PROMPT = (
    "Use the monata-sim-env skill to set up this complete Monata environment, "
    "bootstrap PTM techlibs, generate monata_readme_demo.py, and run it."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="monata")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="check the local Monata simulation environment")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    parser.print_help()
    return 2


def doctor(*, stdout: TextIO | None = None) -> int:
    out = stdout or sys.stdout
    status = _doctor_status()

    print("Monata environment doctor", file=out)
    print("", file=out)
    print("Python package: installed", file=out)
    _print_tool_status(out, "ngspice", status.ngspice)
    _print_tool_status(out, "openvaf-r", status.openvaf)
    if status.monata_home is None:
        print("MONATA_HOME: not set", file=out)
    else:
        print(f"MONATA_HOME: {status.monata_home}", file=out)
    if status.techlib_dir is None or not status.techlib_dir.is_dir():
        print("techlibs: missing", file=out)
    else:
        techlibs = _techlib_names(status.techlib_dir)
        summary = ", ".join(techlibs) if techlibs else "none found"
        print(f"techlibs: {status.techlib_dir} ({summary})", file=out)

    if status.ok:
        print("", file=out)
        print("Simulation runtime: ready", file=out)
        return 0

    print("", file=out)
    print("Simulation runtime: incomplete", file=out)
    print("", file=out)
    print("Recommended setup: install and use the monata-sim-env skill.", file=out)
    print("", file=out)
    print("Open Skills CLI:", file=out)
    print("  npx skills@latest add lizhangmai/skills --skill monata-sim-env", file=out)
    print("", file=out)
    print("Codex plugin:", file=out)
    print("  codex plugin marketplace add https://github.com/lizhangmai/skills --ref main", file=out)
    print("  codex plugin add monata-sim-env@lizhangmai", file=out)
    print("", file=out)
    print("Then ask your agent:", file=out)
    print(f"  {_RECOMMENDED_PROMPT}", file=out)
    print("  CONDA_BUILD_OUTPUT_DIR=<absolute-path-you-choose>", file=out)
    print("  MONATA_HOME=<optional-absolute-monata-home>", file=out)
    return 1


def _print_tool_status(out: TextIO, name: str, path: str | None) -> None:
    print(f"{name}: {path if path else 'missing'}", file=out)


class _DoctorStatus:
    def __init__(
        self,
        *,
        ngspice: str | None,
        openvaf: str | None,
        monata_home: Path | None,
        techlib_dir: Path | None,
    ) -> None:
        self.ngspice = ngspice
        self.openvaf = openvaf
        self.monata_home = monata_home
        self.techlib_dir = techlib_dir

    @property
    def ok(self) -> bool:
        return bool(
            self.ngspice
            and self.openvaf
            and self.techlib_dir is not None
            and self.techlib_dir.is_dir()
            and _techlib_names(self.techlib_dir)
        )


def _doctor_status() -> _DoctorStatus:
    monata_home_text = os.environ.get("MONATA_HOME")
    monata_home = Path(monata_home_text).expanduser() if monata_home_text else None
    techlib_dir = monata_home / "techlibs" if monata_home is not None else None
    return _DoctorStatus(
        ngspice=shutil.which("ngspice"),
        openvaf=shutil.which("openvaf-r"),
        monata_home=monata_home,
        techlib_dir=techlib_dir,
    )


def _techlib_names(path: Path) -> tuple[str, ...]:
    if not path.is_dir():
        return ()
    return tuple(sorted(child.name for child in path.iterdir() if child.is_dir()))


if __name__ == "__main__":
    raise SystemExit(main())
