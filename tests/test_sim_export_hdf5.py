import json
import sys

import numpy as np
import pytest

from monata.measure import MeasureResult
from monata.sim.export import export_sim_result_hdf5, load_sim_result_hdf5
from monata.sim.results import AnalysisResult, Waveform
from monata.sim.results import SimResult


def _result():
    return SimResult(
        status="ok",
        waveforms={"v/out": np.array([0.0, 1.0])},
        sweep_var=np.array([0.0, 1e-9]),
        corner=None,
        metadata={"simulator": "mock"},
    )


def test_export_sim_result_hdf5_requires_optional_dependency_when_missing(tmp_path, monkeypatch):
    if "h5py" in sys.modules:
        pytest.skip("h5py is installed; missing-dependency behavior is not active")

    monkeypatch.setitem(sys.modules, "h5py", None)

    with pytest.raises(RuntimeError, match="optional 'hdf5' extra"):
        export_sim_result_hdf5(_result(), tmp_path / "result.h5")


def test_load_sim_result_hdf5_requires_optional_dependency_when_missing(tmp_path, monkeypatch):
    if "h5py" in sys.modules:
        pytest.skip("h5py is installed; missing-dependency behavior is not active")

    monkeypatch.setitem(sys.modules, "h5py", None)

    with pytest.raises(RuntimeError, match="optional 'hdf5' extra"):
        load_sim_result_hdf5(tmp_path / "result.h5")


def test_export_sim_result_hdf5_writes_waveforms_when_h5py_available(tmp_path):
    h5py = pytest.importorskip("h5py")

    path = export_sim_result_hdf5(_result(), tmp_path / "result.h5")

    with h5py.File(path, "r") as file:
        assert file.attrs["format"] == "monata.sim.result"
        assert file["simulation"].attrs["status"] == "ok"
        assert json.loads(file["simulation"].attrs["metadata_json"]) == {"simulator": "mock"}
        assert "sweep" in file["abscissas"]
        assert "v__out" in file["waveforms"]
        assert file["waveforms"]["v__out"].compression == "gzip"
        assert file["waveforms"]["v__out"].compression_opts == 9
        np.testing.assert_allclose(file["waveforms"]["v__out"][:], np.array([0.0, 1.0]))


def test_export_sim_result_hdf5_can_disable_dataset_compression(tmp_path):
    h5py = pytest.importorskip("h5py")

    path = export_sim_result_hdf5(_result(), tmp_path / "uncompressed.h5", compression=None)

    with h5py.File(path, "r") as file:
        assert file["abscissas"]["sweep"].compression is None
        assert file["waveforms"]["v__out"].compression is None

    loaded = load_sim_result_hdf5(path)
    np.testing.assert_allclose(loaded.waveforms["v/out"], np.array([0.0, 1.0]))
    assert loaded.sweep_var is not None
    np.testing.assert_allclose(loaded.sweep_var, np.array([0.0, 1e-9]))


def test_load_sim_result_hdf5_round_trips_typed_result_payload(tmp_path):
    pytest.importorskip("h5py")
    sweep = np.array([0.0, 1e-9], dtype=np.float32)
    waveform = np.array([0.0 + 0.0j, 1.0 - 0.25j], dtype=np.complex64)
    analysis = AnalysisResult(
        analysis="tran",
        waveforms={
            "out": Waveform(
                name="out",
                data=waveform,
                unit="V",
                quantity="voltage",
                title="Output",
                metadata={"limits": [0.0, 1.0]},
                source_vector="v(out)",
                raw_vector_name="v(out)",
                vector_kind="node_voltage",
                abscissa="time",
                analysis="tran",
                source="rawfile",
                extraction="rawfile",
                plot_name="V(out)",
            ),
            "manual": Waveform.from_array(
                "manual",
                np.array([5.0, 6.0], dtype=np.float32),
                abscissa=np.array([2.0, 3.0], dtype=np.float32),
                unit="V",
                quantity="voltage",
            ),
        },
        abscissa=Waveform("time", sweep, unit="s", quantity="time"),
        metadata={"extraction": "rawfile"},
        source="rawfile",
    )
    result = SimResult(
        status="ok",
        waveforms={"v/out": waveform},
        sweep_var=sweep,
        corner={"name": "tt", "temperature": 27, "voltages": {"vdd": 1.0}},
        metadata={"simulator": "mock", "analysis": "tran"},
        analysis_result=analysis,
        measures={"final": MeasureResult("final", 1.0, unit="V", source="summary")},
        summaries={"plain": {"score": 2.0}},
    )

    path = export_sim_result_hdf5(result, tmp_path / "result.h5")
    loaded = load_sim_result_hdf5(path)

    assert loaded.status == "ok"
    assert loaded.metadata == {"analysis": "tran", "simulator": "mock"}
    assert loaded.corner is not None
    assert loaded.corner.name == "tt"
    assert loaded.corner.temperature == pytest.approx(27)
    assert loaded.measures["final"].value == pytest.approx(1.0)
    assert loaded.measures["final"].unit == "V"
    assert loaded.summaries == {"plain": {"score": 2.0}}
    assert set(loaded.waveforms) == {"v/out"}
    np.testing.assert_allclose(loaded.waveforms["v/out"], waveform)
    assert loaded.waveforms["v/out"].dtype == np.complex64
    assert loaded.sweep_var is not None
    np.testing.assert_allclose(loaded.sweep_var, sweep)
    assert loaded.sweep_var.dtype == np.float32

    assert loaded.analysis_result is not None
    assert loaded.analysis_result.abscissa is not None
    assert loaded.analysis_result.abscissa.name == "time"
    typed = loaded.analysis_result.waveform("v(out)")
    assert typed.name == "out"
    assert typed.unit == "V"
    assert typed.quantity == "voltage"
    assert typed.raw_vector_name == "v(out)"
    assert typed.source_vector == "v(out)"
    assert typed.abscissa_name == "time"
    assert typed.source == "rawfile"
    assert typed.plot_name == "V(out)"
    assert typed.metadata == {"limits": [0.0, 1.0]}
    np.testing.assert_allclose(typed.data, waveform)
    np.testing.assert_allclose(typed.abscissa_data, sweep)

    manual = loaded.analysis_result.waveform("manual")
    assert manual.abscissa_name == "abscissa"
    np.testing.assert_allclose(manual.data, np.array([5.0, 6.0]))
    np.testing.assert_allclose(manual.abscissa_data, np.array([2.0, 3.0]))


def test_hdf5_export_stores_shared_analysis_abscissa_once(tmp_path):
    h5py = pytest.importorskip("h5py")
    sweep = np.array([0.0, 1e-9], dtype=np.float32)
    custom_axis = np.array([2.0, 3.0], dtype=np.float32)
    analysis = AnalysisResult(
        analysis="tran",
        waveforms={
            "out": Waveform(
                name="out",
                data=np.array([0.0, 1.0], dtype=np.float32),
                unit="V",
                quantity="voltage",
                vector_kind="node_voltage",
                abscissa="time",
            ),
            "manual": Waveform.from_array(
                "manual",
                np.array([5.0, 6.0], dtype=np.float32),
                abscissa=custom_axis,
                unit="V",
                quantity="voltage",
            ),
        },
        abscissa=Waveform("time", sweep, unit="s", quantity="time"),
    )
    result = SimResult(status="ok", waveforms={}, sweep_var=None, corner=None, analysis_result=analysis)

    path = export_sim_result_hdf5(result, tmp_path / "shared-abscissa.h5")

    with h5py.File(path, "r") as file:
        analysis_group = file["analysis"]
        assert "time" in analysis_group["abscissas"]
        assert "out" not in analysis_group["waveform_abscissas"]
        assert "manual" in analysis_group["waveform_abscissas"]
        out = analysis_group["waveforms"]["out"]
        assert out.attrs["abscissa_source"] == "analysis"
        assert out.attrs["abscissa_dataset"] == "/analysis/abscissas/time"
        assert file[out.attrs["abscissa_ref"]].name == "/analysis/abscissas/time"
        manual = analysis_group["waveforms"]["manual"]
        assert manual.attrs["abscissa_source"] == "waveform"
        assert manual.attrs["abscissa_dataset"] == "/analysis/waveform_abscissas/manual"
        assert file[manual.attrs["abscissa_ref"]].name == "/analysis/waveform_abscissas/manual"
        np.testing.assert_allclose(analysis_group["abscissas"]["time"][:], sweep)
        np.testing.assert_allclose(analysis_group["waveform_abscissas"]["manual"][:], custom_axis)

    loaded = load_sim_result_hdf5(path)
    assert loaded.analysis_result is not None
    np.testing.assert_allclose(loaded.analysis_result.waveform("out").abscissa_data, sweep)
    np.testing.assert_allclose(loaded.analysis_result.waveform("manual").abscissa_data, custom_axis)


def test_load_sim_result_hdf5_resolves_waveform_abscissa_reference(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "referenced-abscissa.h5"
    custom_axis = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    with h5py.File(path, "w") as file:
        file.attrs["format"] = "monata.sim.result"
        file.attrs["version"] = 1
        simulation = file.create_group("simulation")
        simulation.attrs["status"] = "ok"
        file.create_group("waveforms")
        analysis = file.create_group("analysis")
        analysis.attrs["analysis"] = "tran"
        analysis.attrs["metadata_json"] = "{}"
        waveforms = analysis.create_group("waveforms")
        axis = analysis.create_group("referenced_abscissas").create_dataset("custom_axis", data=custom_axis)
        waveform = waveforms.create_dataset("out", data=np.array([0.0, 1.0, 4.0], dtype=np.float32))
        waveform.attrs["source_name"] = "out"
        waveform.attrs["name"] = "out"
        waveform.attrs["unit"] = "V"
        waveform.attrs["quantity"] = "voltage"
        waveform.attrs["vector_kind"] = "node_voltage"
        waveform.attrs["abscissa_name"] = "custom_axis"
        waveform.attrs["abscissa_ref"] = axis.ref

    loaded = load_sim_result_hdf5(path)

    assert loaded.analysis_result is not None
    assert loaded.analysis_result.abscissa is None
    loaded_waveform = loaded.analysis_result.waveform("out")
    assert loaded_waveform.abscissa_name == "custom_axis"
    np.testing.assert_allclose(loaded_waveform.data, np.array([0.0, 1.0, 4.0]))
    np.testing.assert_allclose(loaded_waveform.abscissa_data, custom_axis)


def test_hdf5_export_indexes_analysis_waveforms_by_entity(tmp_path):
    h5py = pytest.importorskip("h5py")
    analysis = AnalysisResult(
        analysis="op",
        waveforms={
            "v(out)": Waveform(
                "out",
                np.array([1.2]),
                quantity="voltage",
                source_vector="v(out)",
                raw_vector_name="v(out)",
                vector_kind="node_voltage",
            ),
            "i(VDD)": Waveform(
                "i(VDD)",
                np.array([0.01]),
                quantity="current",
                source_vector="i(VDD)",
                raw_vector_name="i(vdd)",
                vector_kind="branch_current",
            ),
            "@M1[gm]": Waveform(
                "@M1[gm]",
                np.array([1e-3]),
                source_vector="@M1[gm]",
                raw_vector_name="@m1[gm]",
                vector_kind="element_parameter",
            ),
            "@M1[id]": Waveform(
                "@M1[id]",
                np.array([0.02]),
                quantity="current",
                source_vector="@M1[id]",
                raw_vector_name="@m1[id]",
                vector_kind="node_current",
            ),
            "@temp": Waveform(
                "@temp",
                np.array([27.0]),
                source_vector="@temp",
                raw_vector_name="@temp",
                vector_kind="internal_parameter",
            ),
        },
    )
    result = SimResult(status="ok", waveforms={}, sweep_var=None, corner=None, analysis_result=analysis)

    path = export_sim_result_hdf5(result, tmp_path / "entities.h5")

    with h5py.File(path, "r") as file:
        entities = file["analysis"]["entities"]
        np.testing.assert_allclose(entities["node_voltages"]["out"][:], np.array([1.2]))
        np.testing.assert_allclose(entities["branch_currents"]["VDD"][:], np.array([0.01]))
        np.testing.assert_allclose(entities["device_parameters"]["M1"]["gm"][:], np.array([1e-3]))
        np.testing.assert_allclose(entities["device_parameters"]["M1"]["id"][:], np.array([0.02]))
        np.testing.assert_allclose(entities["internal_parameters"]["@temp"][:], np.array([27.0]))
        assert entities["node_voltages"]["out"].attrs["source_name"] == "v(out)"
        assert entities["device_parameters"]["M1"].attrs["source_name"] == "M1"

    loaded = load_sim_result_hdf5(path)
    assert loaded.analysis_result is not None
    assert set(loaded.analysis_result.node_voltages_by_node) == {"out"}
    assert set(loaded.analysis_result.branch_currents_by_element) == {"VDD"}
    assert set(loaded.analysis_result.device_parameters_by_element["M1"]) == {"gm", "id"}


def test_hdf5_export_preserves_colliding_waveform_storage_names(tmp_path):
    pytest.importorskip("h5py")
    result = SimResult(
        status="ok",
        waveforms={
            "v/out": np.array([1.0, 2.0]),
            "v__out": np.array([3.0, 4.0]),
        },
        sweep_var=np.array([0.0, 1e-9]),
        corner=None,
    )

    path = export_sim_result_hdf5(result, tmp_path / "colliding.h5")
    loaded = load_sim_result_hdf5(path)

    np.testing.assert_allclose(loaded.waveforms["v/out"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(loaded.waveforms["v__out"], np.array([3.0, 4.0]))


def test_load_sim_result_hdf5_rejects_unknown_file_format(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bad.h5"

    with h5py.File(path, "w") as file:
        file.attrs["format"] = "other"

    with pytest.raises(ValueError, match="unsupported HDF5 result format"):
        load_sim_result_hdf5(path)
