from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest


ROOT_FACADE_EXPORTS = {
    "Cell",
    "CellNotFoundError",
    "Category",
    "Library",
    "LibraryNotFoundError",
    "LibraryRegistry",
    "OperatingCorner",
    "Quantity",
    "Unit",
    "UnitArray",
    "View",
    "ViewAlreadyModifiedError",
    "ViewNotFoundError",
    "ViewNotGeneratedError",
}

PARSER_EXPORTS = {
    "ControlCommandAction",
    "ImportedAsset",
    "SourceSubcircuit",
    "SourceSubcircuitInstance",
    "SpiceAnalysisIssue",
    "SpiceAnalysisMeasurement",
    "SpiceAnalysisPlan",
    "SpiceAnalysisStep",
    "SpiceAnalysisSweep",
    "SpiceBinary",
    "SpiceBranch",
    "SpiceCall",
    "SpiceControlCommand",
    "SpiceExpression",
    "SpiceGroup",
    "SpiceIdentifier",
    "SpiceInternalParameter",
    "SpiceImportExpressionCheck",
    "SpiceImportIssue",
    "SpiceImportPlan",
    "SpiceImportStep",
    "SpiceNumber",
    "SpiceParseError",
    "SpiceSourceDependency",
    "SpiceTernary",
    "SpiceUnary",
    "SpiceVector",
    "UnsupportedConstructError",
    "import_spice_asset",
    "import_spice_deck",
    "inspect_spice_analysis",
    "inspect_spice_import",
    "parse_source_subcircuit",
    "parse_spice_expression",
    "parse_spice_to_circuit",
    "render_spice_expression",
    "spice_to_sim_tasks",
    "spice_to_python",
    "walk_spice_expression",
}

NETLIST_EXPORTS = {
    "Circuit",
    "DeviceSchemaError",
    "DeviceSchemaRegistry",
    "Directive",
    "Element",
    "ElementParameterSpec",
    "ElementSpec",
    "MutationError",
    "MutationProjection",
    "ModelCard",
    "Node",
    "Pin",
    "PinSpec",
    "SourceValue",
    "SubCircuit",
    "Topology",
    "TopologyElement",
    "TopologyError",
    "apply_mutation",
    "default_device_schema",
    "element_spec",
    "normalize_element_params",
    "project_param_overrides",
    "render_ngspice",
}

EDA_EXPORTS = {
    "KiCadComponent",
    "KiCadImportError",
    "KiCadImportIssue",
    "KiCadImportPlan",
    "KiCadImportPolicy",
    "KiCadImportStep",
    "KiCadNet",
    "KiCadNetlist",
    "KiCadNodeRef",
    "import_kicad_netlist",
    "inspect_kicad_netlist",
    "kicad_netlist_to_circuit",
    "kicad_netlist_to_python",
    "parse_kicad_netlist",
}

RAWFILE_EXPORTS = {
    "load_ngspice_rawfile",
}

ANALYSIS_SPEC_EXPORTS = {
    "ACSpec",
    "AnalysisSpec",
    "DCSweep",
    "DCSpec",
    "DistortionSpec",
    "FourierSpec",
    "NoiseSpec",
    "OPSpec",
    "PoleZeroSpec",
    "SensitivitySpec",
    "TranSpec",
    "TransferFunctionSpec",
    "analysis_name",
}

DIGITAL_RESULTS_EXPORTS = {
    "DigitalPropagationDelayRow",
    "DigitalTruthTableResult",
    "DigitalTruthTableRow",
}

VECTOR_NAMES_EXPORTS = {
    "VECTOR_KINDS",
    "VectorName",
    "branch_current_vector",
    "device_parameter_vector",
    "expression_vector",
    "internal_parameter_vector",
    "node_current_vector",
    "normalize_vector_name",
    "voltage_vector",
}

SIM_RESULTS_EXPORTS = {
    "AnalysisResult",
    "ResultPayload",
    "SimResult",
    "SimStatus",
    "Waveform",
    "WaveformMap",
    "WaveformNotFoundError",
    "analysis_result_from_arrays",
}

SIM_CACHE_EXPORTS = {
    "CACHE_SCHEMA",
    "TASK_FINGERPRINT_SCHEMA",
    "SimTaskFingerprint",
    "SimulationCache",
    "SourceArtifactDigest",
    "task_fingerprint",
}

SIMULATOR_CAPABILITIES_EXPORTS = {
    "CapabilityState",
    "SimulatorCapabilities",
    "SimulatorProfile",
    "native_level_profile",
    "ngspice_profile",
}

SIM_CORE_EXPORTS = {
    "ACSpec",
    "AnalysisResult",
    "AnalysisSpec",
    "CornerMatrix",
    "CornerResults",
    "DCSweep",
    "DCSpec",
    "DEFAULT_SIMULATOR",
    "DEFAULT_SIM_TIMEOUT_SECONDS",
    "DistortionSpec",
    "Executor",
    "FourierSpec",
    "LocalExecutor",
    "NoiseSpec",
    "OPSpec",
    "OperatingCorner",
    "PoleZeroSpec",
    "SensitivitySpec",
    "SimArtifactOptions",
    "SimResult",
    "SimStatus",
    "SimTask",
    "SimTaskFingerprint",
    "SimulationCache",
    "SimulationSession",
    "SourceArtifactDigest",
    "TranSpec",
    "TransferFunctionSpec",
    "Waveform",
    "branch_current_vector",
    "device_parameter_vector",
    "expression_vector",
    "internal_parameter_vector",
    "node_current_vector",
    "task_fingerprint",
    "voltage_vector",
}

BACKENDS_EXPORTS = {
    "Backend",
    "BackendCapabilities",
    "BackendRegistry",
    "BackendTaskPlan",
    "default_backend_registry",
    "get_backend",
    "get_backend_capabilities",
    "list_backends",
    "list_backend_capabilities",
    "PlanningBackend",
    "plan_backend_task",
    "register_builtin_backends",
    "register_backend",
    "unsupported_task_result",
    "unregister_backend",
    "validate_backend_task",
    "ValidatingBackend",
}

SPICE_LIBRARY_EXPORTS = {
    "SpiceLibrary",
    "SpiceLibraryAsset",
    "SpiceLibraryError",
    "SpiceLibraryItem",
    "SpiceLibraryReference",
}

VIEWS_EXPORTS = {
    "View",
}


@dataclass(frozen=True)
class ModuleContract:
    module: str
    exports: set[str] | frozenset[str] = frozenset()
    forbidden_attrs: frozenset[str] = frozenset()
    forbidden_exports: frozenset[str] = frozenset()
    exact_all: bool = True


PUBLIC_MODULE_CONTRACTS = (
    ModuleContract("monata", ROOT_FACADE_EXPORTS),
    ModuleContract("monata.views", VIEWS_EXPORTS, frozenset({"register_view_type", "SimulationView"})),
    ModuleContract("monata.parser", PARSER_EXPORTS, frozenset({"parse_spice", "SpiceDeck"})),
    ModuleContract("monata.netlist", NETLIST_EXPORTS),
    ModuleContract("monata.eda", EDA_EXPORTS),
    ModuleContract("monata.sim.rawfile", RAWFILE_EXPORTS),
    ModuleContract("monata.sim.analysis_spec", ANALYSIS_SPEC_EXPORTS),
    ModuleContract("monata.sim.digital_results", DIGITAL_RESULTS_EXPORTS),
    ModuleContract(
        "monata.sim.digital_table",
        forbidden_attrs=frozenset({
            "DigitalPropagationDelayRow",
            "DigitalTruthTableResult",
            "DigitalTruthTableRow",
        }),
        exact_all=False,
    ),
    ModuleContract("monata.sim.vector_names", VECTOR_NAMES_EXPORTS),
    ModuleContract(
        "monata.sim.results",
        SIM_RESULTS_EXPORTS,
        frozenset({"VectorName", "normalize_vector_name", "voltage_vector"}),
    ),
    ModuleContract("monata.sim.task", forbidden_attrs=frozenset({"SimResult", "SimStatus"}), exact_all=False),
    ModuleContract("monata.sim.cache", SIM_CACHE_EXPORTS),
    ModuleContract(
        "monata.sim.capabilities",
        SIMULATOR_CAPABILITIES_EXPORTS,
        frozenset({"BackendCapabilities"}),
    ),
    ModuleContract(
        "monata.sim.backends",
        BACKENDS_EXPORTS,
        frozenset({"NgspiceRunner", "NgspiceSharedRunner", "NgspiceSharedSession"}),
    ),
    ModuleContract("monata.spice_library", SPICE_LIBRARY_EXPORTS),
    ModuleContract("monata.sim", forbidden_attrs=frozenset({"LocalExecutor", "SimTask"}), exact_all=False),
    ModuleContract(
        "monata.sim.core",
        SIM_CORE_EXPORTS,
        forbidden_attrs=frozenset({
            "Corner",
            "MonteCarlo",
            "MonteCarloResults",
            "ParameterSweep",
            "SweepResults",
        }),
    ),
    ModuleContract("monata.sim.corner", forbidden_attrs=frozenset({"Corner"}), exact_all=False),
    ModuleContract("monata.techlib.schema", forbidden_attrs=frozenset({"Corner"}), exact_all=False),
    ModuleContract("monata.techlib", forbidden_attrs=frozenset({"Techlib", "TechlibRegistry"}), exact_all=False),
    ModuleContract("monata.techlib.registry", forbidden_exports=frozenset({"metadata"}), exact_all=False),
)


@pytest.mark.parametrize("contract", PUBLIC_MODULE_CONTRACTS, ids=lambda contract: contract.module)
def test_public_module_contracts(contract: ModuleContract):
    module = importlib.import_module(contract.module)
    module_all = getattr(module, "__all__", ())

    if contract.exact_all:
        assert set(module_all) == contract.exports
    else:
        assert contract.exports <= set(module_all)
    for name in contract.forbidden_attrs:
        assert not hasattr(module, name)
    for name in contract.forbidden_attrs | contract.forbidden_exports:
        assert name not in module_all


def test_ci_matrix_covers_advertised_python_classifiers():
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = pyproject["project"]["classifiers"]
    advertised_versions = {
        classifier.rsplit("::", maxsplit=1)[1].strip()
        for classifier in classifiers
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", classifier)
    }
    ci_text = (project_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert advertised_versions == {"3.11", "3.12"}
    for version in advertised_versions:
        assert f'"{version}"' in ci_text


def test_release_metadata_marks_0_2_as_stable():
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = set(pyproject["project"]["classifiers"])

    major, minor, *_ = pyproject["project"]["version"].split(".")
    assert (major, minor) == ("0", "2")
    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert "Development Status :: 3 - Alpha" not in classifiers


def test_core_package_metadata_keeps_external_tools_out_of_dependencies():
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])
    optional = pyproject["project"]["optional-dependencies"]

    assert dependencies == {"numpy", "matplotlib", "h5py", "cffi"}
    assert set(optional) == {"dev"}
    for dependency_group in (dependencies, *map(set, optional.values())):
        assert not {"ngspice", "openvaf", "xyce", "xdm"} & dependency_group


def test_specialized_sim_workflows_import_from_owner_modules():
    import monata.sim.core as core
    from monata.sim.montecarlo import MonteCarlo, MonteCarloResults
    from monata.sim.sweep import ParameterSweep, SweepResults

    assert ParameterSweep.__module__ == "monata.sim.sweep"
    assert SweepResults.__module__ == "monata.sim.sweep"
    assert MonteCarlo.__module__ == "monata.sim.montecarlo"
    assert MonteCarloResults.__module__ == "monata.sim.montecarlo"
    assert not hasattr(core, "ParameterSweep")
    assert not hasattr(core, "SweepResults")
    assert not hasattr(core, "MonteCarlo")
    assert not hasattr(core, "MonteCarloResults")


def test_import_monata_does_not_eagerly_load_sim_or_backends():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata\n"
                "print('monata.sim' in sys.modules)\n"
                "print('monata.sim.backends' in sys.modules)\n"
                "print('monata.sim.backends.ngspice' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False", "False"]


def test_import_monata_does_not_eagerly_load_model_registry_stack():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata\n"
                "print('monata.models.manifest' in sys.modules)\n"
                "print('monata.models.registry' in sys.modules)\n"
                "print('monata.models.cache' in sys.modules)\n"
                "print('monata.models.compiler' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False", "False", "False"]


def test_import_backend_api_does_not_eagerly_load_or_register_ngspice():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from monata.sim.backends import default_backend_registry\n"
                "print('monata.sim.backends.ngspice' in sys.modules)\n"
                "print(default_backend_registry().list())\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "[]"]


def test_top_level_sim_names_are_not_root_exports():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata\n"
                "print(hasattr(monata, 'SimTask'))\n"
                "print('monata.sim' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_vector_name_helpers_resolve_owned_request_names():
    import monata.sim.vector_names as vector_names

    assert vector_names.voltage_vector("out") == "v(out)"
    assert vector_names.normalize_vector_name("v(out)").normalized_name == "out"


def test_sim_capabilities_model_flow_profile_smoke():
    import monata.sim.capabilities as capabilities

    assert capabilities.ngspice_profile().backend_name == "ngspice-subprocess"


def test_docs_call_out_trusted_python_view_loading():
    project_root = Path(__file__).resolve().parents[1]

    readme = (project_root / "README.md").read_text()

    assert "schematic.monata.json" in readme
    assert "parsed and validated without executing project code" in readme
    assert "load_trusted()" in readme
    assert "trusted = true" in readme
    assert "metadata without an explicit format" in readme
    assert "view loading\nis not sandboxed" in readme
    assert "trusted libraries" in readme
    assert "https://github.com/lizhangmai/monata-docs" in readme
    assert "docs/reference/api-boundaries.md" in readme


def test_import_parser_api_does_not_eagerly_load_sim_or_backends():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata.parser\n"
                "print('monata.sim' in sys.modules)\n"
                "print('monata.sim.backends' in sys.modules)\n"
                "print('monata.sim.backends.ngspice' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False", "False"]


def test_import_netlist_api_does_not_eagerly_load_sim_or_backends():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata.netlist\n"
                "print('monata.sim' in sys.modules)\n"
                "print('monata.sim.backends' in sys.modules)\n"
                "print('monata.sim.backends.ngspice' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False", "False"]


def test_import_kicad_adapter_does_not_eagerly_load_sim_or_backends():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata.eda.kicad\n"
                "print('monata.sim' in sys.modules)\n"
                "print('monata.sim.backends' in sys.modules)\n"
                "print('monata.sim.backends.ngspice' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False", "False"]


def test_import_eda_api_does_not_eagerly_load_sim_or_backends():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata.eda\n"
                "print('monata.sim' in sys.modules)\n"
                "print('monata.sim.backends' in sys.modules)\n"
                "print('monata.sim.backends.ngspice' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False", "False"]


def test_import_rawfile_api_does_not_eagerly_load_ngspice_runner():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata.sim.rawfile\n"
                "print('monata.sim.backends.ngspice' in sys.modules)\n"
                "print('monata.sim.backends.ngspice_shared' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_import_sim_cache_api_does_not_eagerly_load_ngspice_runner():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import monata.sim.cache\n"
                "print('monata.sim.backends.ngspice' in sys.modules)\n"
                "print('monata.sim.backends.ngspice_shared' in sys.modules)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_sim_digital_facade_is_not_a_public_surface():
    project_root = Path(__file__).resolve().parents[1]

    assert not (project_root / "src" / "monata" / "sim" / "digital.py").exists()


def test_digital_task_and_extract_layers_use_explicit_plan_boundary():
    project_root = Path(__file__).resolve().parents[1]
    private_table_protocols = (
        "_sequence_arcs_for_result",
        "_scheduled_single_bit_sequence_arcs",
        "_scheduled_propagation_delay_arcs",
        "_logic_value",
        "_task_metadata",
        "_measurable_single_bit_arc_count",
        "_propagation_delay_coverage",
    )

    for relative_path in (
        "src/monata/sim/digital_tasks.py",
        "src/monata/sim/digital_extract.py",
    ):
        source = (project_root / relative_path).read_text()
        for name in private_table_protocols:
            assert f"table.{name}" not in source


def test_sim_core_convenience_bundle_exports_core_contracts():
    import monata.sim.core as core
    from monata.sim.core import (
        DEFAULT_SIMULATOR,
        SimArtifactOptions,
        SimStatus,
        SimTask,
        TranSpec,
        branch_current_vector,
        device_parameter_vector,
        expression_vector,
        internal_parameter_vector,
        node_current_vector,
        voltage_vector,
    )

    task = SimTask(circuit=None, analysis_spec=TranSpec(stop=1e-6))

    assert task.simulator == DEFAULT_SIMULATOR
    assert isinstance(task.artifacts, SimArtifactOptions)
    assert SimStatus.OK == "ok"
    assert voltage_vector("out") == "v(out)"
    assert branch_current_vector("V1") == "i(V1)"
    assert node_current_vector("M1", "id") == "@M1[id]"
    assert device_parameter_vector("M1", "gm") == "@M1[gm]"
    assert internal_parameter_vector("temp") == "@temp"
    assert expression_vector("v(out)-v(in)") == "v(out)-v(in)"
    assert {
        "branch_current_vector",
        "device_parameter_vector",
        "expression_vector",
        "internal_parameter_vector",
        "node_current_vector",
        "SimArtifactOptions",
        "voltage_vector",
    } <= set(core.__all__)


def test_sim_modules_do_not_import_public_vector_private_helpers():
    project_root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (project_root / "src" / "monata" / "sim").rglob("*.py"):
        if path.name == "vector_names.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "from monata.sim.vector_names import _" in source:
            offenders.append(path.relative_to(project_root).as_posix())

    assert offenders == []


def test_core_sdist_omits_local_docs_tree():
    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / "pyproject.toml", "rb") as file:
        pyproject = tomllib.load(file)

    sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert not (project_root / "docs").exists()
    assert all(not entry.startswith("docs/") for entry in sdist["include"])
