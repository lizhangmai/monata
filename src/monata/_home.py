"""Monata-managed mutable state locations."""

from __future__ import annotations

import os
from pathlib import Path


def monata_home(explicit: str | Path | None = None) -> Path:
    """Return the configured or default Monata home root."""

    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("MONATA_HOME")
    return Path(env) if env else default_monata_home()


def default_monata_home() -> Path:
    """Return the fixed default used when MONATA_HOME is unset."""

    return Path(os.path.expanduser("~/.monata"))


def monata_cache_dir(*, home: str | Path | None = None) -> Path:
    """Return the generic Monata cache directory for a configured home/default."""

    return monata_home(home) / "cache"


def monata_registry_dir(*, home: str | Path | None = None) -> Path:
    """Return the Monata registry directory under the home root."""

    return monata_home(home) / "registry"


def monata_logs_dir(*, home: str | Path | None = None) -> Path:
    """Return the Monata logs directory under the home root."""

    return monata_home(home) / "logs"


def monata_techlib_dir(*, home: str | Path | None = None) -> Path:
    """Return the user-level Monata technology-library resource directory."""

    return monata_home(home) / "techlibs"
