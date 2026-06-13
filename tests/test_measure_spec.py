import numpy as np
import pytest

from monata.corner import OperatingCorner
from monata.measure.spec import Spec, SpecTable
from monata.measure import MeasureResult
from monata.sim.results import SimResult
from monata.sim.corner import CornerResults
from monata.units import quantity


def _make_result(out_value, corner_name="nom"):
    corner = OperatingCorner(corner_name, 27, voltages={"vdd": 1.0})
    return SimResult(
        status="ok",
        waveforms={"out": np.ones(10) * out_value},
        sweep_var=np.linspace(0, 1e-6, 10),
        corner=corner,
        metadata={},
    )


def _make_result_with_measure(out_value, measure_value, corner_name="nom"):
    corner = OperatingCorner(corner_name, 27, voltages={"vdd": 1.0})
    return SimResult(
        status="ok",
        waveforms={"out": np.ones(10) * out_value},
        sweep_var=np.linspace(0, 1e-6, 10),
        corner=corner,
        metadata={},
        measures={"tphl": MeasureResult("tphl", measure_value, unit="s")},
        summaries={"ac": {"bandwidth": 2e6}},
    )


def _make_failed_result(corner_name="bad"):
    corner = OperatingCorner(corner_name, 125, voltages={"vdd": 0.9})
    return SimResult(
        status="failed",
        waveforms={},
        sweep_var=None,
        corner=corner,
        error_message="no convergence",
        metadata={},
    )


class TestSpec:
    def test_pass_within_bounds(self):
        spec = Spec("gain", lambda r: r.waveforms["out"][0], min=50, max=80)
        result = spec.evaluate(_make_result(60.0))
        assert result.passed is True
        assert result.value == 60.0
        assert result.margin > 0

    def test_fail_below_min(self):
        spec = Spec("gain", lambda r: r.waveforms["out"][0], min=50)
        result = spec.evaluate(_make_result(40.0))
        assert result.passed is False
        assert result.margin < 0

    def test_fail_above_max(self):
        spec = Spec("gain", lambda r: r.waveforms["out"][0], max=50)
        result = spec.evaluate(_make_result(60.0))
        assert result.passed is False

    def test_min_only(self):
        spec = Spec("gain", lambda r: r.waveforms["out"][0], min=50)
        result = spec.evaluate(_make_result(60.0))
        assert result.passed is True
        assert result.margin == 10.0

    def test_max_only(self):
        spec = Spec("gain", lambda r: r.waveforms["out"][0], max=80)
        result = spec.evaluate(_make_result(60.0))
        assert result.passed is True
        assert result.margin == 20.0

    def test_measure_metric_helper(self):
        spec = Spec("tphl", Spec.measure("tphl"), max=2e-9, unit="s", source="simulator_measure")
        result = spec.evaluate(_make_result_with_measure(1.0, 1e-9))

        assert result.value == 1e-9
        assert result.passed is True
        assert result.unit == "s"
        assert result.source == "simulator_measure"

    def test_measure_metric_helper_converts_measure_units_to_spec_unit(self):
        spec = Spec(
            "tphl",
            Spec.measure("tphl"),
            max=quantity(2.0, "ns"),
            unit="ns",
            source="simulator_measure",
        )

        result = spec.evaluate(_make_result_with_measure(1.0, 1e-9))

        assert result.value == pytest.approx(1.0)
        assert result.passed is True
        assert result.margin == pytest.approx(1.0)
        assert result.unit == "ns"

    def test_converts_quantity_metric_to_spec_unit(self):
        spec = Spec(
            "vout",
            lambda _r: quantity(1200.0, "mV"),
            min=quantity(1.0, "V"),
            max=1.3,
            unit="V",
        )

        result = spec.evaluate(_make_result(0.0))

        assert result.value == pytest.approx(1.2)
        assert result.passed is True
        assert result.margin == pytest.approx(0.1)
        assert result.unit == "V"

    def test_quantity_metric_sets_result_unit_when_spec_unit_is_unspecified(self):
        spec = Spec("vout", lambda _r: quantity(1200.0, "mV"), min=quantity(1.0, "V"))

        result = spec.evaluate(_make_result(0.0))

        assert result.value == pytest.approx(1200.0)
        assert result.margin == pytest.approx(200.0)
        assert result.unit == "mV"

    def test_quantity_bounds_require_compatible_units(self):
        spec = Spec("vout", lambda _r: quantity(1.2, "V"), min=quantity(1.0, "s"))

        with pytest.raises(ValueError, match="value-compatible"):
            spec.evaluate(_make_result(0.0))

    def test_numeric_metric_allows_display_only_unit_labels(self):
        spec = Spec("gain", lambda _r: 42.0, min=20.0, unit="dB")

        result = spec.evaluate(_make_result(0.0))

        assert result.value == 42.0
        assert result.passed is True
        assert result.unit == "dB"

    def test_quantity_metric_requires_convertible_spec_unit(self):
        spec = Spec("vout", lambda _r: quantity(1.2, "V"), min=1.0, unit="dB")

        with pytest.raises(ValueError, match="unknown spec unit"):
            spec.evaluate(_make_result(0.0))

    def test_rejects_reversed_bounds(self):
        with pytest.raises(ValueError, match="less than or equal"):
            Spec("gain", lambda r: r.waveforms["out"][0], min=2.0, max=1.0)


class TestSpecTable:
    def test_evaluate_all(self):
        table = SpecTable()
        table.add("peak", lambda r: r.waveforms["out"].max(), min=0.8, max=1.2)
        table.add("mean", lambda r: r.waveforms["out"].mean(), min=0.5)

        results = [_make_result(1.0, "c1"), _make_result(0.9, "c2")]
        cr = CornerResults(results)
        df = table.evaluate_all(cr)
        assert len(df) == 2
        assert "peak" in df.columns
        assert "mean" in df.columns

    def test_add_rejects_reversed_bounds(self):
        table = SpecTable()
        with pytest.raises(ValueError, match="less than or equal"):
            table.add("peak", lambda r: r.waveforms["out"].max(), min=2.0, max=1.0)

    def test_failed_corner_marked_na(self):
        table = SpecTable()
        table.add("peak", lambda r: r.waveforms["out"].max(), min=0.8)

        results = [_make_result(1.0, "c1"), _make_failed_result("c2")]
        cr = CornerResults(results)
        df = table.evaluate_all(cr)
        assert len(df) == 2
        assert np.isnan(df.loc[df["corner"] == "c2", "peak"].values[0])

    def test_worst_corner(self):
        table = SpecTable()
        table.add("peak", lambda r: r.waveforms["out"][0], min=0.8)

        results = [_make_result(1.0, "c1"), _make_result(0.85, "c2")]
        cr = CornerResults(results)
        table.evaluate_all(cr)
        corner, value = table.worst_corner("peak", cr)
        assert corner.name == "c2"
        assert value == 0.85

    def test_worst_corner_uses_max_bound_margin(self):
        table = SpecTable()
        table.add("peak", lambda r: r.waveforms["out"][0], max=1.2)

        results = [_make_result(1.1, "safe"), _make_result(1.3, "over")]
        corner, value = table.worst_corner("peak", CornerResults(results))

        assert corner.name == "over"
        assert value == 1.3

    def test_evaluate_rows_mixes_python_measure_and_summary_sources(self):
        table = SpecTable()
        table.add("peak", lambda r: r.waveforms["out"].max(), min=0.8, unit="V")
        table.add_measure("tphl", max=2e-9, unit="s")
        table.add_summary("bw", summary="ac", key="bandwidth", min=1e6, unit="Hz")

        rows = table.evaluate_rows([_make_result_with_measure(1.0, 1e-9, "nom")])

        assert rows == [
            {
                "name": "peak",
                "corner": "nom",
                "value": 1.0,
                "passed": True,
                "margin": 0.19999999999999996,
                "unit": "V",
                "source": "python_metric",
                "reason": None,
            },
            {
                "name": "tphl",
                "corner": "nom",
                "value": 1e-9,
                "passed": True,
                "margin": 1e-9,
                "unit": "s",
                "source": "simulator_measure",
                "reason": None,
            },
            {
                "name": "bw",
                "corner": "nom",
                "value": 2e6,
                "passed": True,
                "margin": 1e6,
                "unit": "Hz",
                "source": "summary",
                "reason": None,
            },
        ]

    def test_evaluate_rows_reports_failed_missing_and_metric_errors(self):
        table = SpecTable()
        table.add_measure("missing", min=0)
        table.add("bad_metric", lambda _r: 1 / 0)

        rows = table.evaluate_rows([_make_failed_result("bad"), _make_result(1.0, "nom")])

        assert rows[0]["source"] == "failed"
        assert rows[0]["corner"] == "bad"
        assert rows[0]["value"] is None
        assert rows[0]["passed"] is None
        assert rows[0]["reason"] == "no convergence"
        assert rows[2]["name"] == "missing"
        assert rows[2]["source"] == "missing"
        assert "measure not found" in rows[2]["reason"]
        assert rows[3]["name"] == "bad_metric"
        assert rows[3]["source"] == "python_metric"
        assert "ZeroDivisionError" in rows[3]["reason"]

    def test_evaluate_rows_treats_nan_as_missing(self):
        table = SpecTable()
        table.add_measure("tphl", min=0)
        table.add("nan_metric", lambda _r: float("nan"), min=0)

        rows = table.evaluate_rows([_make_result_with_measure(1.0, np.nan, "nom")])

        assert rows[0]["name"] == "tphl"
        assert rows[0]["source"] == "missing"
        assert rows[0]["value"] is None
        assert "not finite" in rows[0]["reason"]
        assert rows[1]["name"] == "nan_metric"
        assert rows[1]["source"] == "python_metric"
        assert "not finite" in rows[1]["reason"]

    def test_evaluate_rows_keeps_python_value_error_as_metric_error(self):
        table = SpecTable()
        table.add("bad_value", lambda _r: (_ for _ in ()).throw(ValueError("bad metric")))

        rows = table.evaluate_rows([_make_result(1.0, "nom")])

        assert rows[0]["name"] == "bad_value"
        assert rows[0]["source"] == "python_metric"
        assert "bad metric" in rows[0]["reason"]

    def test_evaluate_all_remains_numeric_projection(self):
        table = SpecTable()
        table.add_measure("tphl", max=2e-9)

        df = table.evaluate_all([_make_result_with_measure(1.0, 1e-9, "nom")])

        assert df.columns == ["corner", "tphl"]
        assert df["tphl"][0] == 1e-9

    def test_evaluate_all_projects_quantity_metrics_in_spec_unit(self):
        table = SpecTable()
        table.add("vout", lambda _r: quantity(1200.0, "mV"), max=1.3, unit="V")

        df = table.evaluate_all([_make_result(0.0, "nom")])

        assert df["vout"][0] == pytest.approx(1.2)

    def test_evaluate_rows_uses_metric_quantity_unit_when_unspecified(self):
        table = SpecTable()
        table.add("vout", lambda _r: quantity(1200.0, "mV"), min=quantity(1.0, "V"))

        rows = table.evaluate_rows([_make_result(0.0, "nom")])

        assert rows[0]["value"] == pytest.approx(1200.0)
        assert rows[0]["margin"] == pytest.approx(200.0)
        assert rows[0]["unit"] == "mV"

    def test_evaluate_all_projects_missing_measure_and_summary_to_nan(self):
        table = SpecTable()
        table.add_measure("missing_measure")
        table.add_summary("missing_summary", "ac", "missing_key")

        df = table.evaluate_all([_make_result_with_measure(1.0, 1e-9, "nom")])

        assert np.isnan(df["missing_measure"][0])
        assert np.isnan(df["missing_summary"][0])

    def test_evaluate_all_does_not_swallow_metric_programming_errors(self):
        table = SpecTable()
        table.add("bad_metric", lambda _r: 1 / 0)

        with pytest.raises(ZeroDivisionError):
            table.evaluate_all([_make_result(1.0, "nom")])

    def test_worst_corner_ignores_failed_missing_and_returns_none_for_all_na(self):
        table = SpecTable()
        table.add_measure("tphl")

        corner, value = table.worst_corner("tphl", [_make_failed_result("bad"), _make_result(1.0, "missing")])

        assert corner is None
        assert value is None

    def test_worst_corner_ignores_nan_and_returns_none_for_all_na(self):
        table = SpecTable()
        table.add("peak", lambda _r: float("nan"))

        corner, value = table.worst_corner("peak", [_make_result(1.0, "nan")])

        assert corner is None
        assert value is None

    def test_worst_corner_does_not_swallow_metric_programming_errors(self):
        table = SpecTable()
        table.add("bad_metric", lambda _r: 1 / 0)

        with pytest.raises(ZeroDivisionError):
            table.worst_corner("bad_metric", [_make_result(1.0, "nom")])

    def test_worst_corner_uses_converted_quantity_margins(self):
        table = SpecTable()
        table.add(
            "vout",
            lambda r: quantity(900.0 if r.corner.name == "safe" else 1300.0, "mV"),
            max=quantity(1.2, "V"),
            unit="V",
        )

        corner, value = table.worst_corner("vout", [_make_result(0.0, "safe"), _make_result(0.0, "over")])

        assert corner.name == "over"
        assert value == pytest.approx(1.3)
