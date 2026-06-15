import json

import numpy as np
import pytest

from monata.measure import MeasureNotFoundError, MeasureResult, MeasureSet
from monata.sim.backends.ngspice_stdout import parse_measure_print
from monata.sim.results import SimResult
from monata.units import Quantity


def test_measure_result_serializes_json_safe_values():
    measure = MeasureResult(
        name="tphl",
        value=np.float64(1.2e-9),
        unit="s",
        analysis="tran",
        raw="tphl = 1.2e-09",
        metadata={"samples": np.array([1, 2]), "scale": np.float64(2.5)},
    )

    data = measure.to_dict()

    assert data["name"] == "tphl"
    assert data["value"] == 1.2e-9
    assert data["unit"] == "s"
    assert data["analysis"] == "tran"
    assert data["reason"] is None
    assert data["metadata"] == {"samples": [1, 2], "scale": 2.5}
    json.dumps(data)


def test_measure_set_lookup_value_and_missing_error():
    measures = MeasureSet({"tphl": MeasureResult("tphl", 1.5e-9, unit="s")})

    assert measures["tphl"].unit == "s"
    assert measures.value("tphl") == 1.5e-9

    with pytest.raises(MeasureNotFoundError) as exc_info:
        measures["tplh"]

    assert "tplh" in str(exc_info.value)
    assert "tphl" in str(exc_info.value)


def test_measure_set_value_with_unit_preserves_registered_units():
    measures = MeasureSet({"tphl": MeasureResult("tphl", 1.5, unit="ms")})

    typed = measures.value_with_unit("tphl")

    assert isinstance(typed, Quantity)
    assert typed.unit.symbol == "ms"
    assert typed.to("s").value == pytest.approx(1.5e-3)
    assert measures.value("tphl") == 1.5


def test_measure_set_value_with_unit_keeps_display_units_numeric():
    measures = MeasureSet({"gain": MeasureResult("gain", 42.0, unit="dB")})

    assert measures.value_with_unit("gain") == 42.0


def test_measure_set_accepts_serialized_measure_mapping():
    measures = MeasureSet({
        "gain": {
            "name": "gain",
            "value": 42.0,
            "unit": "dB",
            "source": "summary",
            "metadata": {"origin": "ac"},
        }
    })

    assert measures["gain"].source == "summary"
    assert measures.to_dict()["gain"]["metadata"] == {"origin": "ac"}


def test_failed_measure_can_be_represented_without_scalar_value():
    measures = MeasureSet({
        "missing_cross": MeasureResult(
            "missing_cross",
            None,
            source="simulator_measure",
            reason="measure_failed",
            raw="missing_cross failed",
        )
    })

    assert measures["missing_cross"].value is None
    assert measures["missing_cross"].reason == "measure_failed"
    with pytest.raises(ValueError, match="measure_failed"):
        measures.value("missing_cross")
    with pytest.raises(ValueError, match="measure_failed"):
        measures.value_with_unit("missing_cross")


def test_sim_result_array_constructor_has_empty_measures_and_summaries():
    result = SimResult(
        status="ok",
        waveforms={},
        sweep_var=None,
        corner=None,
    )

    assert len(result.measures) == 0
    assert result.summaries == {}


def test_sim_result_accepts_measures_and_summaries_without_breaking_waveforms():
    waveform = np.array([0.0, 1.0])
    result = SimResult(
        status="ok",
        waveforms={"out": waveform},
        sweep_var=None,
        corner=None,
        measures={"tphl": MeasureResult("tphl", 1e-9, unit="s")},
        summaries={"tran": {"delay": 1e-9}},
    )

    assert result.waveforms["out"] is not waveform
    assert not result.waveforms["out"].flags.writeable
    np.testing.assert_allclose(result.waveforms["out"], waveform)
    assert result.measures.value("tphl") == 1e-9
    assert result.summaries == {"tran": {"delay": 1e-9}}


def test_parse_measure_print_uses_expected_measure_names_only():
    measures = parse_measure_print(
        """
        unrelated = 3
        tphl = 1.25e-09 targ=1.25e-09 trig=0
        never = failed
        """,
        {"tphl": "tran", "never": "tran", "missing": "ac"},
    )

    assert measures.value("tphl") == 1.25e-9
    assert measures["tphl"].analysis == "tran"
    assert measures["tphl"].raw is not None
    assert measures["never"].value is None
    assert measures["never"].reason == "measure_failed"
    assert measures["missing"].value is None
    assert measures["missing"].reason == "measure_missing"
    assert "unrelated" not in measures
