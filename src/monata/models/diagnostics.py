"""Stable diagnostics for model registry, cache, and compiler workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_MODEL_DIAGNOSTIC_FIELDS = frozenset({"code", "message", "context"})


@dataclass(frozen=True)
class ModelDiagnostic:
    """Structured model-workflow diagnostic suitable for tests and metadata."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelDiagnostic":
        unknown = sorted(key for key in data if key not in _MODEL_DIAGNOSTIC_FIELDS)
        if unknown:
            raise TypeError(f"unknown model diagnostic fields: {', '.join(unknown)}")
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            context=dict(data.get("context", {})),
        )


class ModelDiagnosticError(RuntimeError):
    """Error wrapper that exposes a stable model diagnostic payload."""

    def __init__(self, diagnostic: ModelDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic.code}: {diagnostic.message}")

    def to_dict(self) -> dict[str, Any]:
        return self.diagnostic.to_dict()
