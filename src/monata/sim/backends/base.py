"""Backend protocol, capabilities, and registry for simulator integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol, cast

from monata.sim.analysis_spec import analysis_name
from monata.sim.results import SimResult
from monata.sim.task import SimPayload, SimTask

_BACKEND_CAPABILITIES_FIELDS = frozenset({
    "analyses",
    "result_modes",
    "source_mutations",
    "model_artifacts",
    "native_monte_carlo",
    "renderer_contract",
    "runner_contract",
    "parser_contract",
})


@dataclass(frozen=True)
class BackendCapabilities:
    """Descriptive capability contract for simulator backends."""

    analyses: tuple[str, ...] = ()
    result_modes: tuple[str, ...] = ()
    source_mutations: tuple[str, ...] = ()
    model_artifacts: tuple[str, ...] = ()
    native_monte_carlo: bool = False
    renderer_contract: str | None = None
    runner_contract: str | None = None
    parser_contract: str | None = None

    def supports_analysis(self, analysis: str) -> bool:
        return analysis in self.analyses

    def supports_task(self, task: SimTask) -> bool:
        return self.supports_analysis(analysis_name(task.analysis_spec))

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyses": list(self.analyses),
            "result_modes": list(self.result_modes),
            "source_mutations": list(self.source_mutations),
            "model_artifacts": list(self.model_artifacts),
            "native_monte_carlo": self.native_monte_carlo,
            "renderer_contract": self.renderer_contract,
            "runner_contract": self.runner_contract,
            "parser_contract": self.parser_contract,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackendCapabilities:
        unknown = sorted(key for key in data if key not in _BACKEND_CAPABILITIES_FIELDS)
        if unknown:
            raise TypeError(f"unknown backend capabilities fields: {', '.join(unknown)}")
        return cls(
            analyses=tuple(str(value) for value in data.get("analyses", ())),
            result_modes=tuple(str(value) for value in data.get("result_modes", ())),
            source_mutations=tuple(str(value) for value in data.get("source_mutations", ())),
            model_artifacts=tuple(str(value) for value in data.get("model_artifacts", ())),
            native_monte_carlo=bool(data.get("native_monte_carlo", False)),
            renderer_contract=data.get("renderer_contract"),
            runner_contract=data.get("runner_contract"),
            parser_contract=data.get("parser_contract"),
        )


@dataclass(frozen=True)
class BackendTaskPlan:
    """Backend-facing execution plan summary for a SimTask."""

    backend_name: str
    analysis_name: str
    output_names: tuple[str, ...] = ()
    output_vectors: tuple[str, ...] = ()
    metadata: SimPayload = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_name", str(self.backend_name))
        object.__setattr__(self, "analysis_name", str(self.analysis_name))
        object.__setattr__(self, "output_names", tuple(str(name) for name in self.output_names))
        object.__setattr__(self, "output_vectors", tuple(str(name) for name in self.output_vectors))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "analysis_name": self.analysis_name,
            "output_names": list(self.output_names),
            "output_vectors": list(self.output_vectors),
            "metadata": dict(self.metadata),
        }


class Backend(Protocol):
    """Simulator backend contract.

    Backends are process or API adapters that turn a SimTask into a SimResult.
    They must not raise for ordinary simulator failures; return
    SimResult(status="failed") with reason metadata instead.
    """

    name: str
    capabilities: BackendCapabilities

    def run(self, task: SimTask) -> SimResult:
        ...


class ValidatingBackend(Backend, Protocol):
    """Backend contract for adapters that can preflight a task before running."""

    def validate_task(self, task: SimTask) -> SimResult | None:
        ...


class PlanningBackend(Backend, Protocol):
    """Backend contract for adapters that expose a backend-facing task plan."""

    def plan_task(self, task: SimTask) -> BackendTaskPlan:
        ...


BackendFactory = Callable[[], Backend]


class BackendRegistry:
    """Registry object for simulator backend factories."""

    def __init__(self) -> None:
        self._backends: dict[str, BackendFactory] = {}
        self._capabilities: dict[str, BackendCapabilities] = {}

    def register(
        self,
        name: str,
        factory: BackendFactory,
        *,
        replace: bool = False,
        capabilities: BackendCapabilities | None = None,
    ) -> None:
        if not name:
            raise ValueError("backend name is required")
        if name in self._backends and not replace:
            raise ValueError(f"backend already registered: {name}")
        self._backends[name] = factory
        if capabilities is not None:
            self._capabilities[name] = capabilities
        else:
            backend_capabilities = getattr(factory, "capabilities", None)
            if isinstance(backend_capabilities, BackendCapabilities):
                self._capabilities[name] = backend_capabilities
            else:
                self._capabilities.pop(name, None)

    def get(self, name: str) -> Backend:
        try:
            factory = self._backends[name]
        except KeyError as exc:
            raise KeyError(f"unknown simulator backend: {name}") from exc
        return factory()

    def list(self) -> list[str]:
        return sorted(self._backends)

    def capabilities(self, name: str) -> BackendCapabilities:
        if name not in self._backends:
            raise KeyError(f"unknown simulator backend: {name}")
        return self._capabilities.get(name, BackendCapabilities())

    def list_capabilities(self) -> dict[str, BackendCapabilities]:
        return {name: self.capabilities(name) for name in self.list()}

    def unregister(self, name: str) -> None:
        self._backends.pop(name, None)
        self._capabilities.pop(name, None)


_DEFAULT_BACKEND_REGISTRY = BackendRegistry()
_BUILTIN_BACKEND_NAMES = frozenset({"ngspice-subprocess", "ngspice-shared"})
_BUILTIN_BACKENDS_REGISTERED = False
_BUILTIN_BACKENDS_LOCK = RLock()


def default_backend_registry(*, include_builtins: bool = False) -> BackendRegistry:
    """Return the default backend registry.

    By default this returns the raw registry without importing backend
    implementations. Use include_builtins=True when direct registry access should
    observe the same lazily bootstrapped builtins as get_backend/list_backends.
    """

    if include_builtins:
        register_builtin_backends()
    return _DEFAULT_BACKEND_REGISTRY


def _missing_builtin_backend_names() -> set[str]:
    return set(_BUILTIN_BACKEND_NAMES) - set(_DEFAULT_BACKEND_REGISTRY.list())


def register_builtin_backends() -> None:
    """Register built-in backends in the default registry on demand."""

    global _BUILTIN_BACKENDS_REGISTERED
    if _BUILTIN_BACKENDS_REGISTERED and not _missing_builtin_backend_names():
        return

    with _BUILTIN_BACKENDS_LOCK:
        missing = _missing_builtin_backend_names()
        if _BUILTIN_BACKENDS_REGISTERED and not missing:
            return

        if missing:
            from monata.sim.backends.ngspice import NgspiceRunner
            from monata.sim.backends.ngspice_shared import NgspiceSharedRunner

            if NgspiceRunner.name in missing:
                _DEFAULT_BACKEND_REGISTRY.register(NgspiceRunner.name, NgspiceRunner)
            if NgspiceSharedRunner.name in missing:
                _DEFAULT_BACKEND_REGISTRY.register(NgspiceSharedRunner.name, NgspiceSharedRunner)

        _BUILTIN_BACKENDS_REGISTERED = not _missing_builtin_backend_names()


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    replace: bool = False,
    capabilities: BackendCapabilities | None = None,
) -> None:
    default_backend_registry().register(
        name,
        factory,
        replace=replace,
        capabilities=capabilities,
    )


def get_backend(name: str) -> Backend:
    register_builtin_backends()
    return default_backend_registry().get(name)


def list_backends() -> list[str]:
    register_builtin_backends()
    return default_backend_registry().list()


def get_backend_capabilities(name: str) -> BackendCapabilities:
    register_builtin_backends()
    return default_backend_registry().capabilities(name)


def list_backend_capabilities() -> dict[str, BackendCapabilities]:
    register_builtin_backends()
    return default_backend_registry().list_capabilities()


def validate_backend_task(backend_or_name: Backend | str, task: SimTask) -> SimResult | None:
    backend = get_backend(backend_or_name) if isinstance(backend_or_name, str) else backend_or_name
    validator = getattr(backend, "validate_task", None)
    if callable(validator):
        return cast("SimResult | None", validator(task))
    capabilities = getattr(backend, "capabilities", None)
    if isinstance(capabilities, BackendCapabilities) and capabilities.analyses and not capabilities.supports_task(task):
        return unsupported_task_result(task, backend.name, capabilities)
    return None


def plan_backend_task(backend_or_name: Backend | str, task: SimTask) -> BackendTaskPlan:
    backend = get_backend(backend_or_name) if isinstance(backend_or_name, str) else backend_or_name
    planner = getattr(backend, "plan_task", None)
    if callable(planner):
        return cast(BackendTaskPlan, planner(task))
    failure = validate_backend_task(backend, task)
    if failure is not None:
        raise ValueError(failure.error_message or "backend task validation failed")
    return BackendTaskPlan(
        backend_name=backend.name,
        analysis_name=analysis_name(task.analysis_spec),
        output_names=task.output_names,
    )


def unregister_backend(name: str) -> None:
    global _BUILTIN_BACKENDS_REGISTERED

    default_backend_registry().unregister(name)
    if name in _BUILTIN_BACKEND_NAMES:
        _BUILTIN_BACKENDS_REGISTERED = False


def backend_failure_result(
    task: SimTask,
    error_message: str,
    *,
    metadata: SimPayload | None = None,
) -> SimResult:
    return SimResult(
        status="failed",
        waveforms={},
        sweep_var=None,
        corner=task.corner,
        metadata=metadata or {},
        error_message=error_message,
    )


def unsupported_task_result(
    task: SimTask,
    backend_name: str,
    capabilities: BackendCapabilities,
    *,
    reason: str = "unsupported_analysis",
) -> SimResult:
    analysis = analysis_name(task.analysis_spec)
    return backend_failure_result(
        task,
        f"{backend_name} does not support analysis: {analysis}",
        metadata={
            "simulator": backend_name,
            "reason": reason,
            "analysis": analysis,
            "supported_analyses": list(capabilities.analyses),
        },
    )
