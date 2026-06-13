from dataclasses import FrozenInstanceError, is_dataclass
from typing import Any, Callable, cast

import pytest

from monata.sim.analysis_spec import (
    ACSpec,
    DCSweep,
    DCSpec,
    DistortionSpec,
    FourierSpec,
    AnalysisSpec,
    NoiseSpec,
    OPSpec,
    PoleZeroSpec,
    SensitivitySpec,
    TranSpec,
    TransferFunctionSpec,
    analysis_name,
)


def test_specs_are_frozen_dataclass_contracts():
    spec = TranSpec(stop=1e-6)

    assert is_dataclass(spec)
    with pytest.raises(FrozenInstanceError):
        setattr(spec, "stop", 2e-6)


def test_ac_spec_attributes():
    spec = ACSpec(start=1, stop=1e9, points=100, variation="dec")
    assert spec.start == 1
    assert spec.stop == 1e9
    assert spec.points == 100
    assert spec.variation == "dec"


def test_ac_spec_default_variation():
    spec = ACSpec(start=10, stop=1e6, points=50)
    assert spec.variation == "dec"


def test_ac_spec_normalizes_variation():
    spec = ACSpec(start=10, stop=1e6, points=50, variation="OCT")

    assert spec.variation == "oct"


def test_tran_spec_attributes():
    spec = TranSpec(stop=1e-6, step=1e-9, start=0, max_step=5e-10, uic=True)
    assert spec.stop == 1e-6
    assert spec.step == 1e-9
    assert spec.start == 0
    assert spec.max_step == 5e-10
    assert spec.uic is True


def test_tran_spec_defaults():
    spec = TranSpec(stop=1e-3)
    assert spec.start == 0
    assert spec.step is None
    assert spec.max_step is None
    assert spec.uic is False


def test_dc_spec_attributes():
    spec = DCSpec(source="Vgs", start=0, stop=1.2, step=0.01)
    assert spec.source == "Vgs"
    assert spec.start == 0
    assert spec.stop == 1.2
    assert spec.step == 0.01
    assert spec.secondary is None


def test_dc_spec_accepts_secondary_sweep():
    secondary = DCSweep(source="Vds", start=0, stop=5, step=0.5)
    spec = DCSpec(source="Vgs", start=0, stop=1.2, step=0.01, secondary=secondary)

    assert spec.secondary is secondary


def test_op_spec():
    spec = OPSpec()
    assert isinstance(spec, AnalysisSpec)


def test_noise_spec_attributes():
    spec = NoiseSpec(
        output_node="out",
        input_source="Vin",
        start=1,
        stop=1e9,
        points=100,
        reference_node="ref",
        variation="OCT",
        points_per_summary=5,
    )
    assert spec.output_node == "out"
    assert spec.input_source == "Vin"
    assert spec.start == 1
    assert spec.stop == 1e9
    assert spec.points == 100
    assert spec.reference_node == "ref"
    assert spec.variation == "oct"
    assert spec.points_per_summary == 5


def test_sensitivity_spec_attributes():
    spec = SensitivitySpec(output="v(out)", start=1, stop=1e6, points=10, variation="lin")
    assert spec.output == "v(out)"
    assert spec.start == 1
    assert spec.stop == 1e6
    assert spec.points == 10
    assert spec.variation == "lin"


def test_pole_zero_spec_attributes():
    spec = PoleZeroSpec(input_pos="in", input_neg="0", output_pos="out", output_neg="0", transfer="CUR", mode="ZER")
    assert spec.input_pos == "in"
    assert spec.input_neg == "0"
    assert spec.output_pos == "out"
    assert spec.output_neg == "0"
    assert spec.transfer == "cur"
    assert spec.mode == "zer"


def test_distortion_spec_attributes():
    spec = DistortionSpec(start=1, stop=1e6, points=10, variation="dec", f2overf1=0.9)
    assert spec.start == 1
    assert spec.stop == 1e6
    assert spec.points == 10
    assert spec.variation == "dec"
    assert spec.f2overf1 == 0.9


def test_transfer_function_spec_attributes():
    spec = TransferFunctionSpec(output="v(out)", input_source="V1")
    assert spec.output == "v(out)"
    assert spec.input_source == "V1"


def test_fourier_spec_attributes():
    spec = FourierSpec(frequency=1000, output="v(out)", stop=0.002, step=1e-5, start=0)
    assert spec.frequency == 1000
    assert spec.output == "v(out)"
    assert spec.stop == 0.002
    assert spec.step == 1e-5
    assert spec.start == 0


def test_all_specs_are_analysis_spec():
    specs = [
        ACSpec(start=1, stop=1e9, points=100),
        TranSpec(stop=1e-6),
        DCSpec(source="V1", start=0, stop=5, step=0.1),
        OPSpec(),
        NoiseSpec(output_node="out", input_source="V1", start=1, stop=1e6, points=50),
        SensitivitySpec(output="v(out)"),
        PoleZeroSpec(input_pos="in", input_neg="0", output_pos="out", output_neg="0"),
        DistortionSpec(start=1, stop=1e6, points=10),
        TransferFunctionSpec(output="v(out)", input_source="V1"),
        FourierSpec(frequency=1000, output="v(out)", stop=0.002),
    ]
    for spec in specs:
        assert isinstance(spec, AnalysisSpec)


def test_analysis_name_is_owned_by_analysis_specs():
    class CustomSpec(AnalysisSpec):
        pass

    cases = [
        (ACSpec(start=1, stop=10, points=3), "ac"),
        (DCSpec(source="V1", start=0, stop=1, step=0.1), "dc"),
        (TranSpec(stop=1e-6), "tran"),
        (OPSpec(), "op"),
        (NoiseSpec(output_node="out", input_source="V1", start=1, stop=10, points=3), "noise"),
        (SensitivitySpec(output="v(out)"), "sensitivity"),
        (PoleZeroSpec(input_pos="in", input_neg="0", output_pos="out", output_neg="0"), "pole-zero"),
        (DistortionSpec(start=1, stop=10, points=3), "distortion"),
        (TransferFunctionSpec(output="v(out)", input_source="V1"), "transfer-function"),
        (FourierSpec(frequency=1000, output="v(out)", stop=0.002), "fourier"),
        (CustomSpec(), "custom"),
    ]

    assert [analysis_name(spec) for spec, _ in cases] == [expected for _, expected in cases]


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: ACSpec(start=0, stop=1e9, points=100), "ac start"),
        (lambda: ACSpec(start=1, stop=1e9, points=0), "ac points"),
        (lambda: ACSpec(start=1, stop=1e9, points=100, variation="bad"), "AC variation"),
        (lambda: TranSpec(stop=0), "tran stop"),
        (lambda: TranSpec(stop=1e-6, step=0), "tran step"),
        (lambda: TranSpec(stop=1e-6, max_step=0), "tran max_step"),
        (lambda: TranSpec(stop=1e-6, start=-1e-9), "tran start"),
        (lambda: TranSpec(stop=1e-6, start=2e-6), "tran stop"),
        (lambda: DCSpec(source="", start=0, stop=1, step=0.1), "dc source"),
        (lambda: DCSpec(source="V1\nquit", start=0, stop=1, step=0.1), "dc source"),
        (lambda: DCSpec(source="V1", start=0, stop=1, step=0), "dc step"),
        (lambda: DCSpec(source="V1", start=0, stop=1, step=0.1, secondary=cast(Any, object())), "dc secondary"),
        (lambda: DCSweep(source="", start=0, stop=1, step=0.1), "dc source"),
        (lambda: DCSweep(source="V2", start=0, stop=1, step=0), "dc step"),
        (lambda: NoiseSpec(output_node="", input_source="V1", start=1, stop=10, points=3), "noise output"),
        (lambda: NoiseSpec(output_node="out;quit", input_source="V1", start=1, stop=10, points=3), "noise output"),
        (lambda: NoiseSpec(output_node="out", input_source="", start=1, stop=10, points=3), "noise input_source"),
        (lambda: NoiseSpec(output_node="out", reference_node="0\nquit", input_source="V1", start=1, stop=10, points=3), "noise reference"),
        (lambda: NoiseSpec(output_node="out", input_source="V1", start=0, stop=10, points=3), "noise start"),
        (lambda: NoiseSpec(output_node="out", input_source="V1", start=1, stop=10, points=3, variation="bad"), "noise variation"),
        (lambda: NoiseSpec(output_node="out", input_source="V1", start=1, stop=10, points=3, points_per_summary=0), "noise points_per_summary"),
        (lambda: SensitivitySpec(output=""), "sensitivity output"),
        (lambda: SensitivitySpec(output="v(out)", start=1, stop=10), "requires start, stop, and points"),
        (lambda: SensitivitySpec(output="v(out)", start=1, stop=10, points=0), "sensitivity points"),
        (lambda: SensitivitySpec(output="v(out)", start=1, stop=10, points=3, variation="bad"), "sensitivity variation"),
        (
            lambda: PoleZeroSpec(input_pos="", input_neg="0", output_pos="out", output_neg="0"),
            "pole-zero input_pos",
        ),
        (
            lambda: PoleZeroSpec(input_pos="in", input_neg="0", output_pos="out", output_neg="0", transfer="bad"),
            "pole-zero transfer",
        ),
        (
            lambda: PoleZeroSpec(input_pos="in", input_neg="0", output_pos="out", output_neg="0", mode="bad"),
            "pole-zero mode",
        ),
        (lambda: DistortionSpec(start=1, stop=10, points=0), "distortion points"),
        (lambda: DistortionSpec(start=1, stop=10, points=3, f2overf1=cast(Any, "bad")), "distortion f2overf1"),
        (lambda: TransferFunctionSpec(output="", input_source="V1"), "tf output"),
        (lambda: TransferFunctionSpec(output="v(out)", input_source="V1;quit"), "tf input_source"),
        (lambda: FourierSpec(frequency=0, output="v(out)", stop=0.002), "fourier frequency"),
        (lambda: FourierSpec(frequency=1000, output="", stop=0.002), "fourier output"),
        (lambda: FourierSpec(frequency=1000, output="v(out)", stop=0.002, step=0), "fourier step"),
        (lambda: FourierSpec(frequency=1000, output="v(out)\nquit", stop=0.002), "fourier output"),
    ],
)
def test_specs_reject_invalid_contract_fields(factory: Callable[[], AnalysisSpec], message: str):
    with pytest.raises(ValueError, match=message):
        factory()
