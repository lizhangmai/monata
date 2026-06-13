import numpy as np
import pytest
from monata.models.manifest import ModelManifest
from monata.models.registry import ModelEntry
from monata.corner import ModelCornerRef, OperatingCorner, OperatingPoint, coerce_operating_corner
from monata.sim.corner import CornerMatrix, CornerResults
from monata.sim.executor import LocalExecutor
from monata.sim.analysis_spec import TranSpec
from monata.sim.analysis_spec import DCSpec
from monata.netlist import Circuit, render_ngspice
from support.executors import CapturingExecutor
from support.results import failed_result, sim_result


def test_corner_creation():
    c = OperatingCorner(name="nom_27C_1V0", temperature=27, voltages={"vdd": 1.0})
    assert c.name == "nom_27C_1V0"
    assert c.temperature == 27
    assert c.voltages == {"vdd": 1.0}
    assert c.process is None
    assert c.model_file is None


def test_corner_legacy_attribute_aliases_are_not_public():
    point = OperatingPoint("nom", temperature=27, voltages={"vdd": 1.0})
    model = ModelCornerRef(process_node="65nm")
    corner = OperatingCorner("nom", voltages={"vdd": 1.0}, process_node="65nm")

    assert not hasattr(point, "voltage")
    assert not hasattr(model, "node")
    assert not hasattr(corner, "voltage")
    assert not hasattr(corner, "node")

    with pytest.raises(ValueError, match="unknown fields: node, voltage"):
        OperatingCorner.from_dict({"name": "legacy", "voltage": {"vdd": 0.9}, "node": "45nm"})
    with pytest.raises(ValueError, match="unknown fields: node"):
        ModelCornerRef.from_dict({"node": "45nm"})


def test_corner_constructor_uses_canonical_voltages_argument_only():
    with pytest.raises(TypeError):
        OperatingPoint("legacy", voltage={"vdd": 1.0})  # type: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        OperatingPoint("legacy", 27, {"vdd": 1.0})  # type: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        OperatingCorner("legacy", voltage={"vdd": 1.0})  # type: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        OperatingCorner("legacy", 27, {"vdd": 1.0})  # type: ignore[reportCallIssue]

    with pytest.raises(ValueError, match="unknown fields: voltage"):
        OperatingPoint.from_dict({"name": "legacy", "voltage": {"vdd": 0.9}})
    with pytest.raises(ValueError, match="unknown fields: voltage"):
        OperatingCorner.from_dict({"name": "legacy", "voltage": {"vdd": 0.9}})


def test_corner_object_coercion_requires_explicit_corner_payload_types():
    class CanonicalCorner:
        name = "canonical"
        temperature = 85
        voltages = {"vdd": 1.1}
        process_node = "22nm"

    class LegacyCornerShape:
        name = "legacy"
        temperature = 125
        voltage = {"vdd": 0.8}
        node = "45nm"

    with pytest.raises(TypeError, match="unsupported operating corner: CanonicalCorner"):
        coerce_operating_corner(CanonicalCorner())
    with pytest.raises(TypeError, match="unsupported operating corner: LegacyCornerShape"):
        coerce_operating_corner(LegacyCornerShape())


def test_corner_payloads_are_read_only_after_creation():
    c = OperatingCorner(
        name="nom_27C_1V0",
        temperature=27,
        voltages={"vdd": 1.0},
        device_defaults={"nmos": {"l": "65n"}},
        metadata={"source": "test"},
    )

    with pytest.raises(TypeError):
        c.voltages["vdd"] = 1.2  # type: ignore[index]
    with pytest.raises(TypeError):
        c.device_defaults["nmos"]["l"] = "70n"  # type: ignore[index]
    with pytest.raises(TypeError):
        c.metadata["source"] = "other"  # type: ignore[index]

    assert c.to_dict()["voltages"] == {"vdd": 1.0}
    assert c.defaults_for_device("nmos") == {"l": "65n"}


def test_corner_with_process():
    c = OperatingCorner(
        name="ff_m40C_1V1",
        temperature=-40,
        voltages={"vdd": 1.1},
        process="ff",
        model_file="/path/to/ff.mod",
    )
    assert c.process == "ff"
    assert c.model_file == "/path/to/ff.mod"


def test_operating_corner_payload_preserves_canonical_fields(tmp_path):
    model_file = tmp_path / "ptm.mod"
    corner = OperatingCorner(
        name="ptm65",
        temperature=85,
        voltages={"VDD": 1.1},
        techlib="PTM_BULK",
        model_deck="ptm_bulk_65nm",
        section="ptm65",
        model_file=model_file,
        nominal_vdd=1.1,
        process="ptm65",
        process_node="65nm",
        flavor="bulk",
        device_defaults={"nmos": {"l": "65n"}},
        metadata={"source": "test"},
    )

    payload = corner.to_dict()
    restored = OperatingCorner.from_dict(payload)

    assert payload["schema"] == "monata.operating_corner.v1"
    assert payload["voltages"] == {"VDD": 1.1}
    assert payload["model_deck"] == "ptm_bulk_65nm"
    assert payload["model_file"] == str(model_file)
    assert payload["device_defaults"] == {"nmos": {"l": "65n"}}
    assert restored == corner


def test_operating_corner_separates_operating_point_from_model_corner(tmp_path):
    model_file = tmp_path / "ptm.mod"
    point = OperatingPoint("ptm65_hot", temperature=85, voltages={"vdd": 1.1})
    model = ModelCornerRef(
        process="tt",
        model_file=model_file,
        techlib="PTM_BULK",
        model_deck="ptm_bulk_65nm",
        section="ptm65",
        nominal_vdd=1.0,
        process_node="65nm",
        flavor="bulk",
        device_defaults={"nmos": {"l": "65n"}},
        metadata={"source": "techlib"},
    )

    corner = OperatingCorner.from_parts(point, model, metadata={"run": "smoke"})

    assert corner.operating_point == point
    assert corner.model_corner == ModelCornerRef(
        process="tt",
        model_file=model_file,
        techlib="PTM_BULK",
        model_deck="ptm_bulk_65nm",
        section="ptm65",
        nominal_vdd=1.0,
        process_node="65nm",
        flavor="bulk",
        device_defaults={"nmos": {"l": "65n"}},
        metadata={"source": "techlib", "run": "smoke"},
    )
    assert corner.temperature == point.temperature
    assert corner.model_file == str(model_file)
    assert corner.defaults_for_device("nmos") == {"l": "65n"}
    assert corner.to_dict()["metadata"] == {"source": "techlib", "run": "smoke"}


@pytest.mark.parametrize("section", ["tt", "ss_125c", "ptm65:tt", "ptm.65-tt"])
def test_operating_corner_accepts_safe_model_section_tokens(section):
    corner = OperatingCorner("safe", section=section)

    assert corner.section == section


@pytest.mark.parametrize(
    "section",
    [
        "",
        "bad section",
        "tt\n.include evil.mod",
        "tt\r.temp 999",
        'tt" .include evil.mod',
        "tt; .include evil.mod",
        "tt # comment",
        ".tt",
        "1tt",
    ],
)
def test_operating_corner_rejects_unsafe_model_sections(section):
    with pytest.raises(ValueError, match="corner model section"):
        OperatingCorner("unsafe", section=section)


def test_corner_matrix_cartesian_product():
    spec = TranSpec(stop=1e-6)
    matrix = CornerMatrix(circuit=None, analysis_spec=spec)
    matrix.add_temperatures(-40, 27, 125)
    matrix.add_voltages("vdd", 0.9, 1.0, 1.1)
    corners = matrix.corners()
    assert len(corners) == 9  # 3 temps x 3 voltages


def test_corner_matrix_with_process():
    spec = TranSpec(stop=1e-6)
    matrix = CornerMatrix(circuit=None, analysis_spec=spec, model_manifest=ModelManifest([]))
    matrix.add_temperatures(27)
    matrix.add_voltages("vdd", 1.0)
    matrix.add_model_corners(tt="nmos_tt", ff="nmos_ff", ss="nmos_ss")
    corners = matrix.corners()
    assert len(corners) == 3  # 1 temp x 1 voltage x 3 process
    assert corners[0].model_file is None
    assert corners[0].metadata["model_selection"] == "nmos_tt"


def test_corner_matrix_full_pvt():
    spec = TranSpec(stop=1e-6)
    matrix = CornerMatrix(circuit=None, analysis_spec=spec, model_manifest=ModelManifest([]))
    matrix.add_temperatures(-40, 27, 125)
    matrix.add_voltages("vdd", 0.9, 1.0, 1.1)
    matrix.add_model_corners(tt="nmos_tt", ff="nmos_ff")
    corners = matrix.corners()
    assert len(corners) == 18  # 3 x 3 x 2


def test_corner_matrix_tasks_preserve_ingress_fields_and_metadata():
    spec = TranSpec(stop=1e-6)
    matrix = CornerMatrix(
        circuit=None,
        analysis_spec=spec,
        output_names=["out"],
        osdi_paths=["model.osdi"],
        metadata={"run": "corners"},
        simulator="custom-backend",
        timeout=None,
        backend_options={"rawfile_format": "binary"},
        artifacts="artifacts",
    )
    matrix.add_temperatures(27)
    matrix.add_voltages("vdd", 1.0)
    executor = CapturingExecutor()

    results = matrix.run(executor)
    task = executor.tasks[0]

    assert task.output_names == ("out",)
    assert [str(path) for path in task.osdi_paths] == ["model.osdi"]
    assert task.metadata["run"] == "corners"
    assert task.metadata["corner"]["name"] == "27C_1.0"
    assert task.simulator == "custom-backend"
    assert task.timeout is None
    assert task.backend_options == {"rawfile_format": "binary"}
    assert str(task.artifacts.directory) == "artifacts"
    assert results["27C_1.0"].metadata["corner"]["voltages"] == {"vdd": 1.0}


def test_corner_matrix_model_corners_require_manifest():
    matrix = CornerMatrix(circuit=None, analysis_spec=TranSpec(stop=1e-6))
    matrix.add_temperatures(27)

    with pytest.raises(ValueError, match="model_manifest"):
        matrix.add_model_corners(tt="nmos_tt")


@pytest.mark.parametrize("model_selection", ["tt.mod", "models/tt.lib", "models\\tt.lib", "nmos.osdi"])
def test_corner_matrix_rejects_model_file_path_selections(model_selection):
    matrix = CornerMatrix(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        model_manifest=ModelManifest([]),
    )

    with pytest.raises(ValueError, match="model selection"):
        matrix.add_model_corners(tt=model_selection)


def test_corner_matrix_resolves_manifest_model_corner_to_task_fields(tmp_path):
    model_file = tmp_path / "models.lib"
    osdi = tmp_path / "bsim4.osdi"
    model_file.write_text(".lib tt\n.endl\n")
    osdi.write_text("compiled")
    manifest = ModelManifest([
        ModelEntry(
            name="nmos_tt",
            family="mos",
            module_name="bsim4",
            osdi_path=osdi,
            model_file=model_file,
            lib_section="tt",
        )
    ])
    matrix = CornerMatrix(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        osdi_paths=["base.osdi"],
        model_manifest=manifest,
    )
    matrix.add_temperatures(27)
    matrix.add_model_corners(tt="nmos_tt")
    executor = CapturingExecutor()

    matrix.run(executor)
    task = executor.tasks[0]

    assert [str(path) for path in task.osdi_paths] == ["base.osdi", str(osdi)]
    assert task.corner.model_file == str(model_file)
    assert task.metadata["corner"]["model_file"] == str(model_file)
    assert task.metadata["corner"]["metadata"]["model_selection"] == "nmos_tt"
    assert task.metadata["model_selection"]["lib_sections"] == [
        {"path": str(model_file), "section": "tt"}
    ]
    assert task.circuit is None


def test_corner_matrix_projects_manifest_lib_to_task_circuit_without_mutating_source(tmp_path):
    model_file = tmp_path / "models.lib"
    model_file.write_text(".lib tt\n.endl\n")
    manifest = ModelManifest([
        ModelEntry(
            name="nmos_tt",
            family="mos",
            module_name="bsim4",
            model_file=model_file,
            lib_section="tt",
        )
    ])
    circuit = Circuit("manifest lib projection")
    circuit.resistor("1", "in", "0", "1k")
    matrix = CornerMatrix(
        circuit=circuit,
        analysis_spec=TranSpec(stop=1e-6),
        model_manifest=manifest,
    )
    matrix.add_model_corners(tt="nmos_tt")
    executor = CapturingExecutor()

    matrix.run(executor)
    task = executor.tasks[0]

    assert circuit.directives == []
    assert task.circuit is not circuit
    assert task.circuit.directives[0].name == "lib"
    assert task.circuit.directives[0].args == (str(model_file), "tt")
    assert f".lib {model_file} tt" in render_ngspice(task.circuit)


def test_corner_voltage_routes_through_structured_source_mutation():
    circuit = Circuit("corner voltage")
    circuit.voltage("DD", "vdd", "0", "0")
    circuit.resistor("1", "vdd", "0", "1k")
    task_matrix = CornerMatrix(
        circuit=circuit,
        analysis_spec=DCSpec(source="VDD", start=0, stop=1, step=1),
        output_names=["vdd"],
    )
    task_matrix.add_voltages("VDD", 1.2)

    result = task_matrix.run(LocalExecutor(max_workers=1))["27C_1.2"]

    assert result.status == "ok", result.error_message
    assert result.metadata["structured_mutations"] == [
        {"target": "VDD.V", "kind": "structured"}
    ]


def test_corner_results_iteration():
    results = [
        sim_result(waveforms={"out": np.zeros(10)}, sweep_var=np.zeros(10), corner=OperatingCorner("c1", 27, voltages={"vdd": 1.0})),
        failed_result(corner=OperatingCorner("c2", 125, voltages={"vdd": 0.9}), error_message="no convergence"),
    ]
    cr = CornerResults(results)
    assert len(cr) == 2
    assert list(cr) == results


def test_corner_results_passed_failed():
    c1 = OperatingCorner("c1", 27, voltages={"vdd": 1.0})
    c2 = OperatingCorner("c2", 125, voltages={"vdd": 0.9})
    results = [
        sim_result(waveforms={"out": np.ones(10)}, sweep_var=np.zeros(10), corner=c1),
        failed_result(corner=c2),
    ]
    cr = CornerResults(results)
    assert len(cr.passed()) == 1
    assert len(cr.failed()) == 1
    passed_corner = cr.passed()[0].corner
    failed_corner = cr.failed()[0].corner
    assert passed_corner is not None
    assert failed_corner is not None
    assert passed_corner.name == "c1"
    assert failed_corner.name == "c2"


def test_corner_results_getitem():
    c1 = OperatingCorner("c1", 27, voltages={"vdd": 1.0})
    results = [
        sim_result(waveforms={"out": np.ones(10)}, sweep_var=np.zeros(10), corner=c1),
    ]
    cr = CornerResults(results)
    assert cr["c1"].status == "ok"


def test_corner_results_filter():
    c1 = OperatingCorner("c1", 27, voltages={"vdd": 1.0})
    c2 = OperatingCorner("c2", 125, voltages={"vdd": 1.0})
    c3 = OperatingCorner("c3", 27, voltages={"vdd": 0.9})
    results = [
        sim_result(waveforms={"out": np.ones(10)}, sweep_var=np.zeros(10), corner=c1),
        sim_result(waveforms={"out": np.ones(10)}, sweep_var=np.zeros(10), corner=c2),
        sim_result(waveforms={"out": np.ones(10)}, sweep_var=np.zeros(10), corner=c3),
    ]
    cr = CornerResults(results)
    filtered = cr.filter(temperature=27)
    assert len(filtered) == 2
    voltage_filtered = cr.filter(voltages={"vdd": 1.0})
    assert voltage_filtered == [results[0], results[1]]
    with pytest.raises(TypeError):
        cr.filter(voltage={"vdd": 1.0})  # type: ignore[reportCallIssue]


def test_corner_results_extract_histogram_and_sigma_yield_use_passing_results():
    c1 = OperatingCorner("c1", 27, voltages={"vdd": 1.0})
    c2 = OperatingCorner("c2", 125, voltages={"vdd": 0.9})
    c3 = OperatingCorner("c3", -40, voltages={"vdd": 1.1})
    results = [
        sim_result(waveforms={"out": np.array([0.9])}, sweep_var=np.zeros(1), corner=c1),
        failed_result(corner=c2),
        sim_result(waveforms={"out": np.array([1.0])}, sweep_var=np.zeros(1), corner=c3),
    ]
    cr = CornerResults(results)

    values = cr.extract(lambda r: r.waveforms["out"][0])
    bin_edges, counts = cr.histogram(lambda r: r.waveforms["out"][0], bins=2)
    yield_value = cr.sigma_yield(lambda r: r.waveforms["out"][0], spec_min=0.95, spec_max=1.05)

    np.testing.assert_allclose(values, np.array([0.9, 1.0]))
    assert len(bin_edges) == 3
    assert counts.sum() == 2
    assert yield_value == pytest.approx(0.5)


def test_corner_results_statistics_require_passing_results():
    cr = CornerResults([
        failed_result(corner=OperatingCorner("c1")),
    ])

    with pytest.raises(ValueError, match="No passing results"):
        cr.histogram(lambda r: r.waveforms["out"][0])
    with pytest.raises(ValueError, match="No passing results"):
        cr.sigma_yield(lambda r: r.waveforms["out"][0], spec_min=0.0)
    with pytest.raises(ValueError, match="No passing results"):
        cr.worst_case(lambda r: r.waveforms["out"][0])


def test_corner_results_statistics_reject_invalid_metric_values():
    cr = CornerResults([
        sim_result(waveforms={"out": np.array([np.nan])}, sweep_var=np.zeros(1), corner=OperatingCorner("c1")),
    ])

    with pytest.raises(ValueError, match="finite"):
        cr.histogram(lambda r: r.waveforms["out"][0])
    with pytest.raises(ValueError, match="finite"):
        cr.worst_case(lambda r: r.waveforms["out"][0])


def test_corner_results_sigma_yield_rejects_invalid_spec_range():
    cr = CornerResults([
        sim_result(waveforms={"out": np.array([1.0])}, sweep_var=np.zeros(1), corner=OperatingCorner("c1")),
    ])

    with pytest.raises(ValueError, match="less than or equal"):
        cr.sigma_yield(lambda r: r.waveforms["out"][0], spec_min=2.0, spec_max=1.0)


def test_corner_results_worst_case():
    c1 = OperatingCorner("c1", 27, voltages={"vdd": 1.0})
    c2 = OperatingCorner("c2", 125, voltages={"vdd": 0.9})
    results = [
        sim_result(waveforms={"out": np.ones(10) * 5.0}, sweep_var=np.zeros(10), corner=c1),
        sim_result(waveforms={"out": np.ones(10) * 3.0}, sweep_var=np.zeros(10), corner=c2),
    ]
    cr = CornerResults(results)
    worst = cr.worst_case(lambda r: r.waveforms["out"][0])
    assert worst.corner is not None
    assert worst.corner.name == "c2"


def test_corner_results_arrays_preserve_process_and_model_file():
    corner = OperatingCorner("tt", 27, voltages={"vdd": 1.0}, process="tt", model_file="tt.mod")
    results = [
        sim_result(waveforms={"out": np.ones(10)}, sweep_var=np.zeros(10), corner=corner),
    ]
    columns = CornerResults(results).to_arrays()

    assert columns["process"][0] == "tt"
    assert columns["model_file"][0] == "tt.mod"
    assert columns["vdd"][0] == pytest.approx(1.0)


def test_corner_results_arrays_include_optional_metrics_and_failed_nan():
    c1 = OperatingCorner("c1", 27, voltages={"vdd": 1.0})
    c2 = OperatingCorner("c2", 125, voltages={"vdd": 0.9})
    results = [
        sim_result(waveforms={"out": np.array([1.2])}, sweep_var=np.zeros(1), corner=c1),
        failed_result(corner=c2),
    ]

    columns = CornerResults(results).to_arrays({"vout": lambda r: r.waveforms["out"][0]})

    assert columns["corner"].tolist() == ["c1", "c2"]
    np.testing.assert_allclose(columns["vdd"], np.array([1.0, 0.9]))
    assert columns["vout"][0] == pytest.approx(1.2)
    assert np.isnan(columns["vout"][1])


def test_corner_results_arrays_reject_metric_metadata_name_conflicts():
    results = [
        sim_result(waveforms={"out": np.array([1.2])}, sweep_var=np.zeros(1), corner=OperatingCorner("c1")),
    ]

    with pytest.raises(ValueError, match="status"):
        CornerResults(results).to_arrays({"status": lambda r: r.waveforms["out"][0]})


def test_corner_results_arrays_reject_metric_voltage_name_conflicts():
    results = [
        sim_result(
            waveforms={"out": np.array([1.2])},
            sweep_var=np.zeros(1),
            corner=OperatingCorner("c1", 27, voltages={"vdd": 1.0}),
        ),
    ]

    with pytest.raises(ValueError, match="vdd"):
        CornerResults(results).to_arrays({"vdd": lambda r: r.waveforms["out"][0]})
