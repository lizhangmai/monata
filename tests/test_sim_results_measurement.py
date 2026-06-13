
import numpy as np
import pytest

from monata.sim.results import (
    Waveform,
)
from monata.units import Quantity, UnitArray
pytestmark = pytest.mark.slow

def test_waveform_summary_methods_return_units_when_metadata_is_known():
    waveform = Waveform("out", np.array([1.0, 2.0, 3.0]), unit="V", quantity="voltage")

    total = waveform.sum()
    average = waveform.mean()
    minimum = waveform.min()
    maximum = waveform.max()
    standard_deviation = waveform.std()
    span = waveform.peak_to_peak()
    rms = waveform.rms()

    assert isinstance(total, Quantity)
    assert total.unit.symbol == "V"
    assert total.value == pytest.approx(6.0)
    assert isinstance(average, Quantity)
    assert average.unit.symbol == "V"
    assert average.value == pytest.approx(2.0)
    assert isinstance(minimum, Quantity)
    assert minimum.unit.symbol == "V"
    assert minimum.value == pytest.approx(1.0)
    assert isinstance(maximum, Quantity)
    assert maximum.unit.symbol == "V"
    assert maximum.value == pytest.approx(3.0)
    assert isinstance(standard_deviation, Quantity)
    assert standard_deviation.unit.symbol == "V"
    assert standard_deviation.value == pytest.approx(np.std(np.array([1.0, 2.0, 3.0])))
    assert isinstance(span, Quantity)
    assert span.unit.symbol == "V"
    assert span.value == pytest.approx(2.0)
    assert isinstance(rms, Quantity)
    assert rms.unit.symbol == "V"
    assert rms.value == pytest.approx(np.sqrt((1.0 + 4.0 + 9.0) / 3.0))

    matrix = Waveform("grid", np.array([[1.0, 2.0], [3.0, 4.0]]), unit="V")
    column_mean = matrix.mean(axis=0)
    column_std = matrix.std(axis=0)
    column_span = matrix.peak_to_peak(axis=0)
    assert isinstance(column_mean, UnitArray)
    assert column_mean.unit.symbol == "V"
    np.testing.assert_allclose(column_mean.values, np.array([2.0, 3.0]))
    assert isinstance(column_std, UnitArray)
    assert column_std.unit.symbol == "V"
    np.testing.assert_allclose(column_std.values, np.array([1.0, 1.0]))
    assert isinstance(column_span, UnitArray)
    assert column_span.unit.symbol == "V"
    np.testing.assert_allclose(column_span.values, np.array([2.0, 2.0]))


def test_waveform_summary_methods_fall_back_to_plain_values_without_known_scalar_units():
    raw = Waveform("raw", np.array([1.0, 2.0, 3.0]))
    derived = Waveform("derived", np.array([1.0, 4.0]), unit="V^2")
    complex_waveform = Waveform("phasor", np.array([1.0 + 1.0j, 1.0 - 1.0j]), unit="V")

    assert raw.mean() == pytest.approx(2.0)
    assert raw.std() == pytest.approx(np.std(np.array([1.0, 2.0, 3.0])))
    assert raw.peak_to_peak() == pytest.approx(2.0)
    assert derived.mean() == pytest.approx(2.5)
    assert derived.peak_to_peak() == pytest.approx(3.0)
    assert complex_waveform.mean() == pytest.approx(1.0 + 0.0j)
    with pytest.raises(ValueError, match="out"):
        raw.mean(out=np.empty(()))
    with pytest.raises(ValueError, match="out"):
        raw.std(out=np.empty(()))
