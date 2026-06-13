"""Typed records for simulator model artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ModelArtifactKind = Literal[
    "spice_include",
    "spice_lib",
    "verilog_a_source",
    "osdi",
    "converted_model_card",
    "xyce_plugin",
]

_MODEL_ARTIFACT_FIELDS = frozenset({
    "kind",
    "path",
    "role",
    "requires",
    "dialect_constraints",
    "format_version",
    "provenance",
    "content_hash",
    "generated_from",
    "toolchain",
    "package_policy",
    "validation",
})


@dataclass(frozen=True)
class ModelArtifact:
    """Inspectable artifact metadata used by model-flow resolution."""

    kind: ModelArtifactKind
    path: str
    role: str
    requires: dict[str, Any] = field(default_factory=dict)
    dialect_constraints: dict[str, Any] = field(default_factory=dict)
    format_version: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None
    generated_from: list[str] = field(default_factory=list)
    toolchain: dict[str, Any] = field(default_factory=dict)
    package_policy: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
            "role": self.role,
        }
        optional = {
            "requires": self.requires,
            "dialect_constraints": self.dialect_constraints,
            "format_version": self.format_version,
            "provenance": self.provenance,
            "content_hash": self.content_hash,
            "generated_from": self.generated_from,
            "toolchain": self.toolchain,
            "package_policy": self.package_policy,
            "validation": self.validation,
        }
        for key, value in optional.items():
            if value not in (None, {}, []):
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelArtifact":
        unknown = sorted(key for key in data if key not in _MODEL_ARTIFACT_FIELDS)
        if unknown:
            raise TypeError(f"unknown model artifact fields: {', '.join(unknown)}")
        return cls(
            kind=data["kind"],
            path=str(data["path"]),
            role=str(data["role"]),
            requires=dict(data.get("requires", {})),
            dialect_constraints=dict(data.get("dialect_constraints", {})),
            format_version=data.get("format_version"),
            provenance=dict(data.get("provenance", {})),
            content_hash=data.get("content_hash"),
            generated_from=[str(path) for path in data.get("generated_from", [])],
            toolchain=dict(data.get("toolchain", {})),
            package_policy=data.get("package_policy"),
            validation=dict(data.get("validation", {})),
        )


def artifact_sha256(path: str | Path) -> str:
    """Return a stable sha256 digest for an artifact file."""

    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
