"""Monata-managed mutable state locations."""

from __future__ import annotations

import os
from pathlib import Path


def monata_home(explicit: str | Path | None = None) -> Path | None:
    """Return the configured Monata home root, if one is configured."""

    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("MONATA_HOME")
    return Path(env) if env else None


def default_user_cache_root() -> Path:
    """Return the platform-neutral user cache root used when MONATA_HOME is unset."""

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg)
    return Path(os.path.expanduser("~/.cache"))


def monata_cache_dir(*, home: str | Path | None = None) -> Path:
    """Return the generic Monata cache directory for a configured home/default."""

    root = monata_home(home)
    if root is not None:
        return root / "cache"
    return default_user_cache_root() / "monata"


def monata_registry_dir(*, home: str | Path | None = None) -> Path | None:
    """Return the optional Monata registry directory when a home root exists."""

    root = monata_home(home)
    return root / "registry" if root is not None else None


def monata_logs_dir(*, home: str | Path | None = None) -> Path | None:
    """Return the optional Monata logs directory when a home root exists."""

    root = monata_home(home)
    return root / "logs" if root is not None else None
