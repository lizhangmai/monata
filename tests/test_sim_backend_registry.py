from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest

from monata.sim.analysis_spec import AnalysisSpec, TranSpec, analysis_name
from monata.sim.backends import (
    BackendCapabilities,
    BackendRegistry,
    BackendTaskPlan,
    default_backend_registry,
    get_backend,
    get_backend_capabilities,
    list_backends,
    list_backend_capabilities,
    PlanningBackend,
    plan_backend_task,
    register_backend,
    unsupported_task_result,
    unregister_backend,
    validate_backend_task,
    ValidatingBackend,
)
from monata.sim.backends.base import backend_failure_result
from monata.sim.executor import LocalExecutor
from monata.sim.task import SimTask
from support.backends import DummyBackend, MinimalBackend, PlannedBackend, cleanup_test_backends


class UnsupportedSpec(AnalysisSpec):
    pass


@pytest.fixture(autouse=True)
def _cleanup_dummy_backend():
    cleanup_test_backends()
    yield
    cleanup_test_backends()


def test_builtin_ngspice_backend_is_registered():
    assert "ngspice-subprocess" in list_backends()
    assert "ngspice-shared" in list_backends()
    assert get_backend("ngspice-subprocess").name == "ngspice-subprocess"
    assert get_backend("ngspice-shared").name == "ngspice-shared"


def test_builtin_backend_registry_has_only_ngspice_by_default():
    assert list_backends() == ["ngspice-shared", "ngspice-subprocess"]


def test_builtin_ngspice_backend_declares_capabilities():
    capabilities = get_backend_capabilities("ngspice-subprocess")

    assert capabilities.supports_analysis("tran")
    assert capabilities.supports_analysis("ac")
    assert "rawfile" in capabilities.result_modes
    assert "stdout-print" in capabilities.result_modes
    assert "wrdata" not in capabilities.result_modes
    assert "osdi" in capabilities.model_artifacts
    assert capabilities.native_monte_carlo is False
    assert capabilities.runner_contract == "subprocess"
    assert capabilities.parser_contract == "ngspice-rawfile/wrdata-fallback/stdout"
    assert list_backend_capabilities()["ngspice-subprocess"] == capabilities


def test_builtin_shared_ngspice_backend_declares_capabilities():
    capabilities = get_backend_capabilities("ngspice-shared")

    assert capabilities.supports_analysis("tran")
    assert capabilities.supports_analysis("ac")
    assert "rawfile" in capabilities.result_modes
    assert "callback-vectors" not in capabilities.result_modes
    assert "alter-device" not in capabilities.source_mutations
    assert "osdi" in capabilities.model_artifacts
    assert capabilities.runner_contract == "shared-library"
    assert capabilities.parser_contract == "ngspice-rawfile/wrdata-fallback/stdout"


def test_backend_registry_object_isolated_from_default_registry():
    registry = BackendRegistry()

    registry.register(DummyBackend.name, DummyBackend)

    assert registry.list() == [DummyBackend.name]
    assert registry.get(DummyBackend.name).name == DummyBackend.name
    assert registry.capabilities(DummyBackend.name) == DummyBackend.capabilities
    assert DummyBackend.name not in list_backends()


def test_default_backend_registry_can_explicitly_include_builtins():
    registry = default_backend_registry(include_builtins=True)

    assert "ngspice-subprocess" in registry.list()
    assert "ngspice-shared" in registry.list()
    assert registry.get("ngspice-subprocess").name == "ngspice-subprocess"
    assert registry.get("ngspice-shared").name == "ngspice-shared"


def test_builtin_registration_recovers_after_raw_default_registry_unregister():
    registry = default_backend_registry(include_builtins=True)
    registry.unregister("ngspice-subprocess")

    assert "ngspice-subprocess" not in registry.list()
    assert list_backends() == ["ngspice-shared", "ngspice-subprocess"]
    assert get_backend("ngspice-subprocess").name == "ngspice-subprocess"


def test_builtin_registration_is_safe_for_concurrent_first_use():
    unregister_backend("ngspice-subprocess")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: list_backends(), range(16)))

    assert results == [["ngspice-shared", "ngspice-subprocess"]] * 16
    assert get_backend("ngspice-subprocess").name == "ngspice-subprocess"


def test_register_backend_rejects_duplicate_without_replace():
    register_backend(DummyBackend.name, DummyBackend)

    with pytest.raises(ValueError, match="already registered"):
        register_backend(DummyBackend.name, DummyBackend)


def test_registered_backend_dispatches_by_simtask_simulator():
    register_backend(DummyBackend.name, DummyBackend)
    task = SimTask(
        circuit=object(),
        analysis_spec=TranSpec(stop=1e-6),
        simulator=DummyBackend.name,
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "ok"
    assert result.metadata["simulator"] == DummyBackend.name
    np.testing.assert_allclose(result.waveforms["out"], np.array([1.0, 2.0]))


def test_backend_task_plan_falls_back_to_capabilities_and_task_outputs():
    register_backend(DummyBackend.name, DummyBackend)
    task = SimTask(
        circuit=object(),
        analysis_spec=TranSpec(stop=1e-6),
        simulator=DummyBackend.name,
        output_names=["out"],
    )

    plan = plan_backend_task(DummyBackend.name, task)

    assert isinstance(plan, BackendTaskPlan)
    assert plan.to_dict() == {
        "backend_name": DummyBackend.name,
        "analysis_name": "tran",
        "output_names": ["out"],
        "output_vectors": [],
        "metadata": {},
    }
    assert validate_backend_task(DummyBackend.name, task) is None


def test_backend_optional_protocols_define_validation_and_planning_contracts():
    register_backend(PlannedBackend.name, PlannedBackend)
    backend = get_backend(PlannedBackend.name)
    validating = cast(ValidatingBackend, backend)
    planning = cast(PlanningBackend, backend)
    task = SimTask(
        circuit=object(),
        analysis_spec=TranSpec(stop=1e-6),
        simulator=PlannedBackend.name,
        output_names=["out"],
    )

    assert validating.validate_task(task) is None
    assert planning.plan_task(task).to_dict() == {
        "backend_name": PlannedBackend.name,
        "analysis_name": "tran",
        "output_names": ["out"],
        "output_vectors": ["v(out)"],
        "metadata": {"planned": True},
    }
    assert validate_backend_task(PlannedBackend.name, task) is None
    assert plan_backend_task(PlannedBackend.name, task).metadata == {"planned": True}


def test_backend_task_validation_uses_registered_capabilities():
    register_backend(DummyBackend.name, DummyBackend)
    task = SimTask(
        circuit=object(),
        analysis_spec=UnsupportedSpec(),
        simulator=DummyBackend.name,
    )

    result = validate_backend_task(DummyBackend.name, task)

    assert result is not None
    assert result.status == "failed"
    assert result.metadata["reason"] == "unsupported_analysis"
    assert result.metadata["supported_analyses"] == ["tran"]
    with pytest.raises(ValueError, match="does not support analysis"):
        plan_backend_task(DummyBackend.name, task)


def test_backend_without_declared_capabilities_is_not_preflight_rejected():
    register_backend(MinimalBackend.name, MinimalBackend)
    task = SimTask(
        circuit=object(),
        analysis_spec=UnsupportedSpec(),
        simulator=MinimalBackend.name,
    )

    assert validate_backend_task(MinimalBackend.name, task) is None
    assert LocalExecutor(max_workers=1).submit(task).result().status == "ok"


def test_registered_backend_capabilities_are_discoverable():
    register_backend(DummyBackend.name, DummyBackend)

    capabilities = get_backend_capabilities(DummyBackend.name)

    assert capabilities.supports_analysis("tran")
    assert not capabilities.supports_analysis("ac")
    assert capabilities.to_dict()["renderer_contract"] == "dummy-renderer"
    assert BackendCapabilities.from_dict(capabilities.to_dict()) == capabilities


def test_backend_capabilities_reject_unknown_serialized_fields():
    payload = DummyBackend.capabilities.to_dict()
    payload["unexpected"] = True

    with pytest.raises(TypeError, match="unknown backend capabilities fields: unexpected"):
        BackendCapabilities.from_dict(payload)


def test_register_backend_does_not_construct_factory_for_capabilities():
    constructed = []

    class LazyBackend(DummyBackend):
        name = "lazy-backend"

        def __init__(self):
            constructed.append(self.name)

    register_backend(LazyBackend.name, LazyBackend)

    assert constructed == []
    assert get_backend_capabilities(LazyBackend.name) == LazyBackend.capabilities


def test_capability_helper_builds_structured_unsupported_result():
    capabilities = BackendCapabilities(analyses=("dc",), result_modes=("table",))
    task = SimTask(
        circuit=object(),
        analysis_spec=TranSpec(stop=1e-6),
        simulator=DummyBackend.name,
    )

    result = unsupported_task_result(task, DummyBackend.name, capabilities)

    assert result.status == "failed"
    assert result.metadata["simulator"] == DummyBackend.name
    assert result.metadata["reason"] == "unsupported_analysis"
    assert result.metadata["analysis"] == "tran"
    assert result.metadata["supported_analyses"] == ["dc"]
    assert result.error_message is not None
    assert "tran" in result.error_message


def test_backend_failure_helper_builds_standard_failed_result():
    metadata = {"simulator": DummyBackend.name, "reason": "backend_error"}
    task = SimTask(
        circuit=object(),
        analysis_spec=TranSpec(stop=1e-6),
        simulator=DummyBackend.name,
        corner={"name": "nom", "temperature": 27},
    )

    result = backend_failure_result(
        task,
        "backend exploded",
        metadata=MappingProxyType(metadata),
    )
    metadata["reason"] = "changed"

    assert result.status == "failed"
    assert result.waveforms == {}
    assert result.sweep_var is None
    assert result.corner == task.corner
    assert result.metadata == {"simulator": DummyBackend.name, "reason": "backend_error"}
    assert result.error_message == "backend exploded"


def test_analysis_name_normalizes_common_specs():
    assert analysis_name(TranSpec(stop=1e-6)) == "tran"


def test_unknown_backend_returns_failed_result():
    task = SimTask(
        circuit=object(),
        analysis_spec=TranSpec(stop=1e-6),
        simulator="missing-backend",
    )

    result = LocalExecutor(max_workers=1).submit(task).result()

    assert result.status == "failed"
    assert result.metadata["simulator"] == "missing-backend"
    assert result.metadata["reason"] == "unknown_simulator"
    assert "missing-backend" in result.error_message
