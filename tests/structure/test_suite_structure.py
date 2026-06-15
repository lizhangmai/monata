from __future__ import annotations

import ast
import importlib
from pathlib import Path
import tomllib

TEST_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TEST_ROOT.parent

DELETED_LARGE_SUITES = {
    "test_models.py",
    "test_netlist.py",
    "test_sim_backend_native.py",
    "test_sim_digital.py",
    "test_sim_results.py",
}

SLOW_SPLIT_SUITES = {
    "models/test_models_cache.py",
    "models/test_models_compiler.py",
    "models/test_models_external_artifacts.py",
    "models/test_models_resolver.py",
    "models/test_models_schema.py",
    "netlist/test_netlist_construction.py",
    "netlist/test_netlist_mutation_surface.py",
    "netlist/test_netlist_rendering.py",
    "netlist/test_netlist_roundtrip_parser.py",
    "sim/test_sim_digital_result_extraction.py",
    "sim/test_sim_digital_task_construction.py",
    "sim/test_sim_digital_timing_claims.py",
    "sim/test_sim_digital_truth_table_spec.py",
    "sim/test_sim_results_analysis.py",
    "sim/test_sim_results_core.py",
    "sim/test_sim_results_export_plot.py",
    "sim/test_sim_results_measurement.py",
}

NATIVE_SPLIT_SUITES = {
    "integration/test_foundation_closed_loop.py",
    "integration/test_p4_workflow.py",
    "sim/test_sim_backend_native_artifact_parsing.py",
    "sim/test_sim_backend_native_failure_handling.py",
    "sim/test_sim_backend_native_planning.py",
    "sim/test_sim_backend_native_subprocess.py",
}

INTEGRATION_SUITES = {
    "integration/test_foundation_closed_loop.py",
    "integration/test_integration.py",
    "integration/test_p4_workflow.py",
}

STABLE_SUPPORT_EXPORTS = {
    "assertions": {"assert_failed_result", "assert_ok_result"},
    "backends": {
        "DummyBackend",
        "MinimalBackend",
        "PlannedBackend",
        "cleanup_backends",
        "cleanup_test_backends",
    },
    "executors": {
        "CapturingExecutor",
        "ImmediateExecutor",
        "RecordingExecutor",
        "ResultFactory",
        "completed_future",
    },
    "ngspice": {
        "ngspice_available",
        "ngspice_bin_dirs",
        "put_ngspice_on_path",
        "skip_if_no_ngspice",
    },
    "results": {"corner_results", "failed_result", "sim_result"},
    "workspaces": {"create_experiment", "create_project", "open_project"},
}


def _configured_pytest_options() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]


def _module_markers(path: Path) -> set[str]:
    module = ast.parse(path.read_text())
    markers: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets):
            continue
        markers.update(_marker_names(node.value))
    return markers


def _marker_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        if isinstance(node.value.value, ast.Name) and node.value.value.id == "pytest" and node.value.attr == "mark":
            return {node.attr}
    if isinstance(node, (ast.List, ast.Tuple)):
        return set().union(*(_marker_names(element) for element in node.elts))
    return set()


def test_pytest_marker_taxonomy_is_strict_and_registered():
    options = _configured_pytest_options()
    marker_names = {marker.split(":", 1)[0] for marker in options["markers"]}

    assert "--strict-markers" in options["addopts"]
    assert {"fast", "integration", "native", "slow"} <= marker_names


def test_large_root_suites_stay_split():
    deleted_suites = [
        path.relative_to(TEST_ROOT).as_posix()
        for name in DELETED_LARGE_SUITES
        for path in TEST_ROOT.rglob(name)
    ]

    assert deleted_suites == []


def test_split_suites_keep_layer_markers():
    for relative_path in SLOW_SPLIT_SUITES:
        assert "slow" in _module_markers(TEST_ROOT / relative_path)
    for relative_path in NATIVE_SPLIT_SUITES:
        assert "native" in _module_markers(TEST_ROOT / relative_path)
    for relative_path in INTEGRATION_SUITES:
        assert "integration" in _module_markers(TEST_ROOT / relative_path)


def test_test_modules_are_grouped_by_topic():
    root_test_modules = sorted(path.name for path in TEST_ROOT.glob("test_*.py"))

    assert root_test_modules == []


def test_stable_support_modules_expose_explicit_api():
    for module_name, expected_exports in STABLE_SUPPORT_EXPORTS.items():
        module = importlib.import_module(f"support.{module_name}")

        assert set(module.__all__) == expected_exports
        assert all(hasattr(module, name) for name in expected_exports)


def test_case_support_modules_stay_topic_private():
    for path in (TEST_ROOT / "support").glob("*_cases.py"):
        module = ast.parse(path.read_text())
        assigned_names = set()
        for node in module.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                assigned_names.add(node.targets[0].id)

        assert "__all__" not in assigned_names
