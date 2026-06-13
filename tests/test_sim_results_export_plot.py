import json
from types import MappingProxyType

import numpy as np
import pytest

from monata.measure import MeasureResult
from monata.corner import OperatingCorner
from monata.sim.export import (
    export_sim_result_json,
    load_sim_result_json,
    sim_result_from_dict,
    sim_result_to_dict,
)
from monata.sim.results import (
    AnalysisResult,
    SimResult,
    Waveform,
    analysis_result_from_arrays,
)
pytestmark = pytest.mark.slow

def test_complex_waveform_transforms_are_plotting_neutral():
    waveform = Waveform(
        name="out",
        data=np.array([1.0 + 1.0j, 1.0 - 1.0j]),
        unit="V",
        quantity="voltage",
        source_vector="v(out)",
        abscissa="frequency",
    )

    np.testing.assert_allclose(waveform.magnitude().data, np.array([np.sqrt(2), np.sqrt(2)]))
    np.testing.assert_allclose(waveform.real().data, np.array([1.0, 1.0]))
    np.testing.assert_allclose(waveform.imaginary().data, np.array([1.0, -1.0]))
    np.testing.assert_allclose(waveform.phase(degrees=True).data, np.array([45.0, -45.0]))
    np.testing.assert_allclose(waveform.db().data, 20 * np.log10(np.array([np.sqrt(2), np.sqrt(2)])))
    assert waveform.magnitude().raw_vector_name == "v(out)"
    assert waveform.db().unit == "dB"
    assert waveform.phase().unit == "rad"

    wrapped = Waveform("loop", np.exp(1j * np.deg2rad(np.array([170.0, -170.0]))))
    np.testing.assert_allclose(wrapped.phase(unwrap=True).data, np.deg2rad(np.array([170.0, 190.0])))
    np.testing.assert_allclose(wrapped.phase(degrees=True, unwrap=True).data, np.array([170.0, 190.0]))


def test_sim_result_dict_export_uses_external_array_store_contract():
    waveform = np.array([0.0, 1.2], dtype=np.float64)
    sweep = np.array([0.0, 1e-9], dtype=np.float64)
    result = SimResult(
        status="ok",
        waveforms={"out": waveform},
        sweep_var=sweep,
        corner=None,
        metadata={"analysis": "tran", "extraction": "rawfile"},
        analysis_result=AnalysisResult(
            analysis="tran",
            waveforms={
                "out": Waveform(
                    name="out",
                    data=waveform,
                    unit="V",
                    source_vector="v(out)",
                    abscissa="time",
                )
            },
            abscissa=Waveform("time", sweep, unit="s", quantity="time"),
            source="rawfile",
        ),
    )
    array_store = {}

    payload = sim_result_to_dict(result, array_store=array_store, array_prefix="case_")

    assert payload["waveforms"]["out"] == {
        "storage": "npz",
        "key": "case_waveforms__out",
        "dtype": "float64",
        "shape": [2],
        "kind": "real",
    }
    assert payload["sweep_var"]["key"] == "case_sweep_var"
    assert payload["analysis_result"]["abscissa"]["data"]["key"] == "case_analysis__abscissa"
    assert payload["analysis_result"]["waveforms"]["out"]["data"]["key"] == "case_analysis__waveforms__out"
    assert payload["analysis_result"]["waveforms"]["out"]["abscissa_data"]["key"] == (
        "case_analysis__waveforms__out__abscissa_data"
    )
    assert set(array_store) == {
        "case_waveforms__out",
        "case_sweep_var",
        "case_analysis__abscissa",
        "case_analysis__waveforms__out",
        "case_analysis__waveforms__out__abscissa_data",
    }

    with pytest.raises(ValueError, match="external storage"):
        sim_result_from_dict(payload)

    loaded = sim_result_from_dict(payload, array_store=MappingProxyType(array_store))

    np.testing.assert_allclose(loaded.waveforms["out"], waveform)
    assert loaded.sweep_var is not None
    np.testing.assert_allclose(loaded.sweep_var, sweep)
    assert loaded.analysis_result is not None
    np.testing.assert_allclose(loaded.analysis_result.waveform("out").data, waveform)
    np.testing.assert_allclose(loaded.analysis_result.waveform("out").abscissa_data, sweep)
    assert loaded.analysis_result.waveform("out").unit == "V"


def test_sim_result_from_dict_rejects_analysis_waveform_payload_without_name():
    payload = {
        "status": "ok",
        "waveforms": {},
        "sweep_var": None,
        "corner": None,
        "metadata": {},
        "error_message": None,
        "measures": {},
        "summaries": {},
        "analysis_result": {
            "analysis": "tran",
            "source": "rawfile",
            "metadata": {},
            "abscissa": None,
            "waveforms": {
                "out": {
                    "display_name": "out",
                    "raw_vector_name": "v(out)",
                    "data": {"dtype": "float64", "shape": [1], "kind": "real", "data": [1.0]},
                }
            },
        },
    }

    with pytest.raises(ValueError, match="waveform payload missing name"):
        sim_result_from_dict(payload)


def test_sim_result_from_dict_rejects_unknown_top_level_payload_fields():
    payload = {
        "status": "ok",
        "waveforms": {},
        "sweep_var": None,
        "corner": None,
        "metadata": {},
        "error_message": None,
        "measures": {},
        "summaries": {},
        "analysis_result": None,
        "unexpected": True,
    }

    with pytest.raises(ValueError, match="simulation result payload has unknown fields: unexpected"):
        sim_result_from_dict(payload)


def test_sim_result_json_export_can_use_npz_array_sidecar(tmp_path):
    waveform = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
    sweep = np.array([10.0, 20.0], dtype=np.float32)
    result = SimResult(
        status="ok",
        waveforms={"out": waveform},
        sweep_var=sweep,
        corner=None,
        metadata={"analysis": "ac", "extraction": "rawfile"},
        analysis_result=AnalysisResult(
            analysis="ac",
            waveforms={
                "out": Waveform(
                    name="out",
                    data=waveform,
                    unit="V",
                    source_vector="v(out)",
                    abscissa="frequency",
                    abscissa_data=sweep,
                )
            },
            abscissa=Waveform("frequency", sweep, unit="Hz", quantity="frequency"),
            source="rawfile",
        ),
    )
    manifest = tmp_path / "result.json"
    arrays = tmp_path / "result-arrays.npz"

    export_sim_result_json(result, manifest, array_store_path=arrays, array_prefix="run1_")
    payload = json.loads(manifest.read_text())

    assert payload["waveforms"]["out"] == {
        "storage": "npz",
        "key": "run1_waveforms__out",
        "dtype": "complex64",
        "shape": [2],
        "kind": "complex",
    }
    assert payload["analysis_result"]["waveforms"]["out"]["abscissa_data"]["key"] == (
        "run1_analysis__waveforms__out__abscissa_data"
    )
    with np.load(arrays, allow_pickle=False) as store:
        assert set(store.files) == {
            "run1_waveforms__out",
            "run1_sweep_var",
            "run1_analysis__abscissa",
            "run1_analysis__waveforms__out",
            "run1_analysis__waveforms__out__abscissa_data",
        }
        assert store["run1_waveforms__out"].dtype == np.complex64

    with pytest.raises(ValueError, match="external storage"):
        load_sim_result_json(manifest)

    loaded = load_sim_result_json(manifest, array_store_path=arrays)

    np.testing.assert_allclose(loaded.waveforms["out"], waveform)
    assert loaded.waveforms["out"].dtype == np.complex64
    assert loaded.sweep_var is not None
    np.testing.assert_allclose(loaded.sweep_var, sweep)
    assert loaded.sweep_var.dtype == np.float32
    assert loaded.analysis_result is not None
    np.testing.assert_allclose(loaded.analysis_result.waveform("out").data, waveform)
    np.testing.assert_allclose(loaded.analysis_result.waveform("out").abscissa_data, sweep)
    assert loaded.analysis_result.waveform("out").unit == "V"


def test_sim_result_json_export_preserves_arrays_and_typed_metadata(tmp_path):
    waveforms = {"out": np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex64)}
    sweep = np.array([1.0, 10.0], dtype=np.float32)
    typed = analysis_result_from_arrays(
        waveforms,
        sweep,
        {"analysis": "ac", "extraction": "rawfile"},
        source_vectors={"out": "v(out)"},
    )
    waveform = typed.waveform("out")
    analysis_result = AnalysisResult(
        analysis=typed.analysis,
        waveforms={
            "out": Waveform(
                name=waveform.name,
                data=waveform.data,
                unit=waveform.unit,
                quantity=waveform.quantity,
                title="Output voltage",
                source_vector=waveform.source_vector,
                abscissa=waveform.abscissa,
                metadata={"limits": np.array([0.0, 1.0], dtype=np.float32)},
                display_name=waveform.display_name,
                normalized_name=waveform.normalized_name,
                raw_vector_name=waveform.raw_vector_name,
                vector_kind=waveform.vector_kind,
                analysis=waveform.analysis,
                source=waveform.source,
                extraction=waveform.extraction,
                abscissa_name=waveform.abscissa_name,
                plot_name="V(out)",
            ),
            "derived": Waveform(
                name="derived",
                data=np.array([3, 4], dtype=np.int16),
                unit="count",
                quantity="sample",
                abscissa_data=np.array([2.0, 3.0], dtype=np.float32),
            ),
        },
        abscissa=typed.abscissa,
        metadata=typed.metadata,
        source=typed.source,
    )
    result = SimResult(
        status="ok",
        waveforms=waveforms,
        sweep_var=sweep,
        corner=OperatingCorner("tt", 27, voltages={"vdd": 1.0}, process="tt", model_file="tt.lib"),
        metadata={
            "analysis": "ac",
            "extraction": "rawfile",
            "vector_units": {"out": "V"},
            "vector_quantities": {"out": "voltage"},
            "vector_raw_names": {"out": "v(out)"},
            "numpy_scalar": np.float64(2.0),
            "numpy_array": np.array([1, 2], dtype=np.int16),
            "complex_scalar": np.complex64(1.0 + 2.0j),
            "complex_array": np.array([1.0 + 2.0j], dtype=np.complex64),
        },
        analysis_result=analysis_result,
        measures={
            "gain": MeasureResult(
                "gain",
                2.0,
                unit="V/V",
                source="summary",
                metadata={
                    "complex_scalar": np.complex64(1.0 + 2.0j),
                    "complex_array": np.array([1.0 + 2.0j], dtype=np.complex64),
                },
            )
        },
        summaries={"ac": {"peak_gain": np.float64(2.0)}},
    )
    path = tmp_path / "result.json"

    export_sim_result_json(result, path)
    payload = json.loads(path.read_text())
    loaded = load_sim_result_json(path)

    assert payload["corner"]["schema"] == "monata.operating_corner.v1"
    assert payload["corner"]["voltages"] == {"vdd": 1.0}
    assert payload["corner"]["model_file"] == "tt.lib"
    assert loaded.status == "ok"
    assert loaded.corner is not None
    assert loaded.corner.name == "tt"
    assert loaded.corner.voltages == {"vdd": 1.0}
    assert loaded.corner.process == "tt"
    assert loaded.corner.model_file == "tt.lib"
    assert loaded.metadata["analysis"] == "ac"
    assert loaded.metadata["numpy_scalar"] == 2.0
    assert loaded.metadata["numpy_array"] == [1, 2]
    assert loaded.metadata["complex_scalar"] == {"real": 1.0, "imag": 2.0}
    assert loaded.metadata["complex_array"] == {"real": [1.0], "imag": [2.0]}
    assert loaded.measures["gain"].value == 2.0
    assert loaded.measures["gain"].unit == "V/V"
    assert loaded.measures["gain"].metadata["complex_scalar"] == {"real": 1.0, "imag": 2.0}
    assert loaded.measures["gain"].metadata["complex_array"] == {"real": [1.0], "imag": [2.0]}
    assert loaded.summaries == {"ac": {"peak_gain": 2.0}}
    np.testing.assert_allclose(loaded.waveforms["out"], result.waveforms["out"])
    assert loaded.waveforms["out"].dtype == np.complex64
    assert loaded.sweep_var is not None
    assert result.sweep_var is not None
    np.testing.assert_allclose(loaded.sweep_var, result.sweep_var)
    assert loaded.sweep_var.dtype == np.float32
    assert loaded.analysis_result is not None
    assert loaded.analysis_result.abscissa is not None
    assert loaded.analysis_result.abscissa.name == "frequency"
    assert loaded.analysis_result.waveform("out").unit == "V"
    assert loaded.analysis_result.waveform("out").quantity == "voltage"
    assert loaded.analysis_result.waveform("out").raw_vector_name == "v(out)"
    assert loaded.analysis_result.waveform("out").title == "Output voltage"
    assert loaded.analysis_result.waveform("out").metadata["limits"] == [0.0, 1.0]
    assert loaded.analysis_result.waveform("out").plot_name == "V(out)"
    np.testing.assert_allclose(loaded.analysis_result.waveform("out").abscissa_data, sweep)
    np.testing.assert_array_equal(loaded.analysis_result.waveform("derived").data, np.array([3, 4], dtype=np.int16))
    np.testing.assert_allclose(loaded.analysis_result.waveform("derived").abscissa_data, np.array([2.0, 3.0]))
    assert loaded.analysis_result.waveform("derived").data.dtype == np.int16
    assert "derived" not in loaded.waveforms


def test_plotting_helpers_accept_typed_results():
    from monata.sim.plot import plot_analysis, plot_bode, plot_waveform

    result = analysis_result_from_arrays(
        {"out": np.array([1.0 + 0j, 0.0 - 1.0j])},
        np.array([1.0, 10.0]),
        {"analysis": "ac"},
        source_vectors={"out": "v(out)"},
    )
    real_result = analysis_result_from_arrays(
        {"out": np.array([1.0, 0.5]), "in": np.array([0.5, 0.25])},
        np.array([1.0, 10.0]),
        {"analysis": "ac"},
        source_vectors={"out": "v(out)", "in": "v(in)"},
    )
    waveform_abscissa_result = AnalysisResult(
        "ac",
        {
            "out": Waveform(
                "out",
                np.array([1.0, 0.5]),
                unit="V",
                quantity="voltage",
                abscissa="frequency",
                abscissa_data=np.array([100.0, 1000.0]),
            ),
            "in": Waveform(
                "in",
                np.array([0.5, 0.25]),
                unit="V",
                quantity="voltage",
                abscissa="frequency",
                abscissa_data=np.array([100.0, 1000.0]),
            ),
        },
    )
    waveform_bode_result = AnalysisResult(
        "ac",
        {
            "out": Waveform(
                "out",
                np.array([1.0 + 0j, 0.0 - 1.0j]),
                abscissa="frequency",
                abscissa_data=np.array([100.0, 1000.0]),
            )
        },
    )

    axis = plot_waveform(result.waveform("out").magnitude())
    bode_axes = plot_bode(result, "out")
    analysis_axis = plot_analysis(real_result, ("out", "in"))
    waveform_abscissa_axis = plot_analysis(waveform_abscissa_result, ("out", "in"))
    waveform_abscissa_bode_axes = plot_bode(waveform_bode_result, "out")
    styled_axis = plot_waveform(result.waveform("out").magnitude(), "--", color="red", label="gain")
    styled_bode_axes = plot_bode(result, "out", color="green", phase_kwargs={"linestyle": ":"})
    styled_analysis_axis = plot_analysis(
        real_result,
        ("out", "in"),
        "-.",
        marker="o",
        labels={"out": "Output", "in": "Input"},
    )

    assert axis.get_xlabel() == "frequency"
    np.testing.assert_allclose(axis.lines[0].get_xdata(), np.array([1.0, 10.0]))
    assert bode_axes[0].get_ylabel() == "dB"
    assert bode_axes[1].get_ylabel() == "deg"
    assert analysis_axis.get_xlabel() == "frequency"
    assert len(analysis_axis.lines) == 2
    np.testing.assert_allclose(analysis_axis.lines[0].get_xdata(), np.array([1.0, 10.0]))
    assert waveform_abscissa_axis.get_xlabel() == "frequency"
    np.testing.assert_allclose(waveform_abscissa_axis.lines[0].get_xdata(), np.array([100.0, 1000.0]))
    assert waveform_abscissa_bode_axes[1].get_xlabel() == "frequency"
    np.testing.assert_allclose(waveform_abscissa_bode_axes[0].lines[0].get_xdata(), np.array([100.0, 1000.0]))
    assert styled_axis.lines[0].get_linestyle() == "--"
    assert styled_axis.lines[0].get_color() == "red"
    assert styled_axis.lines[0].get_label() == "gain"
    assert styled_bode_axes[0].lines[0].get_color() == "green"
    assert styled_bode_axes[1].lines[0].get_color() == "green"
    assert styled_bode_axes[1].lines[0].get_linestyle() == ":"
    assert [line.get_label() for line in styled_analysis_axis.lines] == ["Output", "Input"]
    assert {line.get_marker() for line in styled_analysis_axis.lines} == {"o"}
    assert {line.get_linestyle() for line in styled_analysis_axis.lines} == {"-."}


def test_plotting_helpers_explain_broken_matplotlib_runtime_dependency(monkeypatch):
    import monata.sim.plot as plot

    def missing_matplotlib(name):
        assert name == "matplotlib.pyplot"
        raise ImportError("matplotlib is not installed")

    monkeypatch.setattr(plot, "import_module", missing_matplotlib)

    with pytest.raises(RuntimeError, match="default Monata runtime dependency"):
        plot.plot_waveform(Waveform(name="out", data=np.array([1.0])))


def test_plot_helpers_use_supplied_axes_without_importing_matplotlib(monkeypatch):
    import monata.sim.plot as plot

    class FalseyAxis:
        def __init__(self):
            self.plot_calls = []
            self.xlabel = ""
            self.ylabel = ""

        def __bool__(self):
            return False

        def plot(self, x_values, y_values, *args, **kwargs):
            self.plot_calls.append((np.asarray(x_values), np.asarray(y_values), args, kwargs))

        def set_xlabel(self, label):
            self.xlabel = label

        def set_ylabel(self, label):
            self.ylabel = label

        def legend(self):
            raise AssertionError("single-waveform plot should not request a legend")

    def fail_on_matplotlib_import(name):
        raise AssertionError(f"unexpected matplotlib import: {name}")

    monkeypatch.setattr(plot, "import_module", fail_on_matplotlib_import)
    waveform = Waveform(
        "out",
        np.array([1.0, 2.0]),
        unit="V",
        quantity="voltage",
        abscissa="time",
        abscissa_data=np.array([0.0, 1.0]),
    )
    waveform_axis = FalseyAxis()
    analysis_axis = FalseyAxis()
    result = AnalysisResult("tran", {"out": waveform})

    assert plot.plot_waveform(waveform, ax=waveform_axis) is waveform_axis
    assert plot.plot_analysis(result, ("out",), ax=analysis_axis) is analysis_axis
    np.testing.assert_allclose(waveform_axis.plot_calls[0][0], np.array([0.0, 1.0]))
    np.testing.assert_allclose(waveform_axis.plot_calls[0][1], np.array([1.0, 2.0]))
    np.testing.assert_allclose(analysis_axis.plot_calls[0][0], np.array([0.0, 1.0]))
    np.testing.assert_allclose(analysis_axis.plot_calls[0][1], np.array([1.0, 2.0]))
    assert waveform_axis.ylabel == "V"
    assert analysis_axis.ylabel == "V"


def test_plot_bode_uses_supplied_axes_without_importing_matplotlib(monkeypatch):
    import monata.sim.plot as plot

    class Axis:
        def __init__(self):
            self.semilogx_calls = []
            self.grid_calls = []
            self.xlabel = ""
            self.ylabel = ""
            self.ylim = None
            self.yticks = None
            self.yticklabels = None

        def semilogx(self, x_values, y_values, **kwargs):
            self.semilogx_calls.append((np.asarray(x_values), np.asarray(y_values), kwargs))

        def grid(self, visible, **kwargs):
            self.grid_calls.append((visible, kwargs))

        def set_xlabel(self, label):
            self.xlabel = label

        def set_ylabel(self, label):
            self.ylabel = label

        def set_ylim(self, low, high):
            self.ylim = (low, high)

        def set_yticks(self, ticks):
            self.yticks = np.asarray(ticks)

        def set_yticklabels(self, labels):
            self.yticklabels = tuple(labels)

    def fail_on_matplotlib_import(name):
        raise AssertionError(f"unexpected matplotlib import: {name}")

    monkeypatch.setattr(plot, "import_module", fail_on_matplotlib_import)
    result = analysis_result_from_arrays(
        {"out": np.array([1.0 + 0.0j, 0.0 - 1.0j])},
        np.array([1.0, 10.0]),
        {"analysis": "ac"},
    )
    magnitude_axis = Axis()
    phase_axis = Axis()

    returned = plot.plot_bode(
        result,
        "out",
        axes=(magnitude_axis, phase_axis),
        phase_unit="rad",
        color="green",
        phase_kwargs={"linestyle": ":"},
    )

    assert returned == (magnitude_axis, phase_axis)
    np.testing.assert_allclose(magnitude_axis.semilogx_calls[0][0], np.array([1.0, 10.0]))
    np.testing.assert_allclose(magnitude_axis.semilogx_calls[0][1], np.array([0.0, 0.0]))
    np.testing.assert_allclose(phase_axis.semilogx_calls[0][1], np.array([0.0, -np.pi / 2.0]))
    assert magnitude_axis.semilogx_calls[0][2] == {"color": "green"}
    assert phase_axis.semilogx_calls[0][2] == {"color": "green", "linestyle": ":"}
    assert magnitude_axis.grid_calls == [(True, {}), (True, {"which": "minor"})]
    assert phase_axis.grid_calls == [(True, {}), (True, {"which": "minor"})]
    assert magnitude_axis.ylabel == "dB"
    assert phase_axis.ylabel == "rad"
    assert phase_axis.xlabel == "frequency"
    assert phase_axis.ylim == (-np.pi, np.pi)
    assert phase_axis.yticks is not None
    np.testing.assert_allclose(phase_axis.yticks, np.array([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]))
    assert phase_axis.yticklabels == (r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$")
