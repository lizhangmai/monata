from __future__ import annotations

import ast
import importlib
from pathlib import Path
import tomllib

TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent

DELETED_LARGE_SUITES = {
    "test_models.py",
    "test_netlist.py",
    "test_sim_backend_native.py",
    "test_sim_digital.py",
    "test_sim_results.py",
}

SLOW_SPLIT_SUITES = {
    "test_models_cache.py",
    "test_models_compiler.py",
    "test_models_external_artifacts.py",
    "test_models_resolver.py",
    "test_models_schema.py",
    "test_netlist_construction.py",
    "test_netlist_mutation_surface.py",
    "test_netlist_rendering.py",
    "test_netlist_roundtrip_parser.py",
    "test_sim_digital_result_extraction.py",
    "test_sim_digital_task_construction.py",
    "test_sim_digital_timing_claims.py",
    "test_sim_digital_truth_table_spec.py",
    "test_sim_results_analysis.py",
    "test_sim_results_core.py",
    "test_sim_results_export_plot.py",
    "test_sim_results_measurement.py",
}

NATIVE_SPLIT_SUITES = {
    "test_foundation_closed_loop.py",
    "test_p4_workflow.py",
    "test_sim_backend_native_artifact_parsing.py",
    "test_sim_backend_native_failure_handling.py",
    "test_sim_backend_native_planning.py",
    "test_sim_backend_native_subprocess.py",
}

INTEGRATION_SUITES = {
    "test_foundation_closed_loop.py",
    "test_integration.py",
    "test_p4_workflow.py",
    "test_sinomos_project_examples.py",
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


def test_large_legacy_suites_stay_split():
    assert not any((TEST_ROOT / name).exists() for name in DELETED_LARGE_SUITES)


def test_split_suites_keep_layer_markers():
    for name in SLOW_SPLIT_SUITES:
        assert "slow" in _module_markers(TEST_ROOT / name)
    for name in NATIVE_SPLIT_SUITES:
        assert "native" in _module_markers(TEST_ROOT / name)
    for name in INTEGRATION_SUITES:
        assert "integration" in _module_markers(TEST_ROOT / name)


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
