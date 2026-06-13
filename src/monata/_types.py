"""Lightweight internal type aliases shared across public owner modules."""

from __future__ import annotations

from typing import Literal

ReferenceMode = Literal["concrete", "logical"]
NetlistProjectionMode = Literal["none", "concrete", "logical"]
