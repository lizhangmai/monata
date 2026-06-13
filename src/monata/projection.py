"""PDK projection services for source-level techlib instances."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from monata.corner import CornerLike, OperatingCorner
from monata._types import ReferenceMode


class PDKProjectionOwner(Protocol):
    @property
    def techlib_attachments(self) -> Iterable[Any]:
        ...


@dataclass(frozen=True)
class PDKProjectionContext:
    """Explicit owner context for techlib-backed PDK projection.

    Libraries can create one of these from their metadata, while projection
    callers can also construct it directly when they already have attachments.
    """

    attachments: tuple[Any, ...] = ()

    @classmethod
    def from_owner(cls, owner: Any) -> "PDKProjectionContext":
        return projection_context_for(owner)

    @property
    def techlib_attachments(self) -> list[Any]:
        return list(self.attachments)

    def validate_pdk_instance(
        self,
        instance: Any,
        registry: Any = None,
        corner: CornerLike = None,
        require_projectable: bool = False,
    ) -> Any:
        techlibs = registry if registry is not None else _default_techlib_registry()
        return techlibs.validate_instance(
            instance,
            attachments=self.techlib_attachments,
            corner=corner,
            require_projectable=require_projectable,
        )

    def project_pdk_instance(
        self,
        instance: Any,
        registry: Any = None,
        corner: CornerLike = None,
    ) -> Any:
        return project_pdk_instance(self, instance, registry=registry, corner=corner)

    def project_pdk_instances(
        self,
        netlist: Any,
        registry: Any = None,
        corner: CornerLike = None,
        reference_mode: ReferenceMode = "concrete",
        include_models: bool = True,
    ) -> Any:
        return project_pdk_instances(
            self,
            netlist,
            registry=registry,
            corner=corner,
            reference_mode=reference_mode,
            include_models=include_models,
        )

    def resolve_pdk_corner(
        self,
        corner: CornerLike = None,
        registry: Any = None,
    ) -> OperatingCorner | None:
        return resolve_pdk_corner(self, corner=corner, registry=registry)


def resolve_pdk_corner(
    owner: Any,
    corner: CornerLike = None,
    registry: Any = None,
) -> OperatingCorner | None:
    """Resolve a PDK corner against the owner's attached technology libraries."""

    context = projection_context_for(owner)
    attachments = context.techlib_attachments
    if not attachments:
        return None

    techlibs = registry if registry is not None else _default_techlib_registry()
    if isinstance(corner, OperatingCorner) and corner.techlib is not None:
        return techlibs.resolve(corner.techlib).corner(corner)

    if len(attachments) != 1:
        if corner is None:
            return None
        raise ValueError(
            "name-only PDK corner resolution is ambiguous with multiple attached techlibs; "
            "pass an OperatingCorner with techlib set"
        )

    attachment = attachments[0]
    selected = corner or attachment.default_corner
    return techlibs.resolve(attachment.name).corner(selected)


def project_pdk_instance(
    owner: Any,
    instance: Any,
    registry: Any = None,
    corner: CornerLike = None,
) -> Any:
    return projection_context_for(owner).validate_pdk_instance(
        instance,
        registry=registry,
        corner=corner,
        require_projectable=True,
    ).project()


def project_pdk_instances(
    owner: Any,
    netlist: Any,
    registry: Any = None,
    corner: CornerLike = None,
    reference_mode: ReferenceMode = "concrete",
    include_models: bool = True,
) -> Any:
    """Project source-level PDK instances in a Circuit/SubCircuit in place."""

    scopes = tuple(projection_scopes(netlist))
    if registry is None and any(scope.pdk_instances for scope in scopes):
        registry = _default_techlib_registry()
    for scope in scopes:
        project_scope_pdk_instances(
            owner,
            scope,
            registry=registry,
            corner=corner,
            reference_mode=reference_mode,
            include_models=include_models,
        )
    return netlist


def projection_scopes(netlist: Any) -> Iterator[Any]:
    from monata.netlist import Circuit, SubCircuit

    if isinstance(netlist, SubCircuit):
        yield netlist.ensure_built()
        return
    if isinstance(netlist, Circuit):
        for subcircuit in netlist.subcircuits:
            yield subcircuit.ensure_built()
        yield netlist
        return
    raise TypeError("netlist must be a monata.netlist Circuit or SubCircuit")


def project_scope_pdk_instances(
    owner: Any,
    scope: Any,
    registry: Any = None,
    corner: CornerLike = None,
    reference_mode: ReferenceMode = "concrete",
    include_models: bool = True,
) -> None:
    instances = tuple(scope.pdk_instances)
    if registry is None and instances:
        registry = _default_techlib_registry()
    seen_model_selections = set()
    for instance in instances:
        projection = project_pdk_instance(owner, instance, registry=registry, corner=corner)
        selection_key = model_selection_key(projection.model_selection)
        include_model_directives = include_models and selection_key not in seen_model_selections
        if selection_key is not None:
            seen_model_selections.add(selection_key)
        projection.apply_to(
            scope,
            include_models=include_model_directives,
            reference_mode=reference_mode,
        )
    scope.pdk_instances.clear()


def projection_context_for(owner: Any) -> PDKProjectionContext:
    """Return the PDK projection context owned by a library-like object."""

    if isinstance(owner, PDKProjectionContext):
        return owner
    context_factory = getattr(owner, "pdk_projection_context", None)
    if callable(context_factory):
        context = context_factory()
        if isinstance(context, PDKProjectionContext):
            return context
        attachments = getattr(context, "techlib_attachments", None)
        if attachments is not None:
            return PDKProjectionContext(tuple(attachments))
    attachments = getattr(owner, "techlib_attachments", None)
    if attachments is None:
        raise TypeError("PDK projection owner must expose techlib_attachments or pdk_projection_context()")
    return PDKProjectionContext(tuple(attachments))


def model_selection_key(model_selection: Any) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]] | None:
    if model_selection is None:
        return None
    return (
        tuple(model_selection.includes),
        tuple(model_selection.lib_sections),
        tuple(model_selection.osdi_paths),
    )


def _default_techlib_registry() -> Any:
    from monata.techlib.registry import TechlibRegistry

    return TechlibRegistry()


__all__ = [
    "PDKProjectionOwner",
    "PDKProjectionContext",
    "model_selection_key",
    "project_pdk_instance",
    "project_pdk_instances",
    "project_scope_pdk_instances",
    "projection_context_for",
    "projection_scopes",
    "resolve_pdk_corner",
]
