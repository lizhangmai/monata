from dataclasses import FrozenInstanceError, is_dataclass
from types import MappingProxyType

import numpy as np
import pytest
from monata.corner import OperatingCorner
from monata.sim.results import SimResult, SimStatus
from monata.sim.task import (
    DEFAULT_SIMULATOR,
    DEFAULT_SIM_TIMEOUT_SECONDS,
    SimArtifactOptions,
    SimTask,
)
from monata.sim.analysis_spec import TranSpec


def test_sim_task_creation():
    spec = TranSpec(stop=1e-6)
    task = SimTask(
        circuit=None,
        analysis_spec=spec,
        simulator="ngspice-subprocess",
    )
    assert task.circuit is None
    assert task.analysis_spec is spec
    assert task.simulator == "ngspice-subprocess"
    assert task.corner is None
    assert task.param_overrides == {}
    assert task.backend_options == {}
    assert task.artifacts == SimArtifactOptions()
    assert task.timeout == DEFAULT_SIM_TIMEOUT_SECONDS


def test_sim_task_default_simulator_is_subprocess():
    task = SimTask(circuit=None, analysis_spec=TranSpec(stop=1e-6))

    assert task.simulator == DEFAULT_SIMULATOR


def test_sim_task_timeout_can_be_overridden_or_disabled():
    timed = SimTask(circuit=None, analysis_spec=TranSpec(stop=1e-6), timeout=1)
    untimed = SimTask(circuit=None, analysis_spec=TranSpec(stop=1e-6), timeout=None)

    assert timed.timeout == 1.0
    assert untimed.timeout is None


def test_sim_task_timeout_must_be_positive_when_set():
    with pytest.raises(ValueError, match="timeout"):
        SimTask(circuit=None, analysis_spec=TranSpec(stop=1e-6), timeout=0)


def test_sim_task_contract_is_frozen_dataclass():
    task = SimTask(circuit=None, analysis_spec=TranSpec(stop=1e-6))

    assert is_dataclass(task)
    with pytest.raises(FrozenInstanceError):
        task.simulator = "other"  # type: ignore[misc]


def test_sim_task_with_corner():
    spec = TranSpec(stop=1e-6)
    corner = {"name": "nom", "temperature": 27}
    task = SimTask(
        circuit=None,
        analysis_spec=spec,
        corner=corner,
        param_overrides={"M1.W": 2e-6},
    )
    assert isinstance(task.corner, OperatingCorner)
    assert task.corner.name == "nom"
    assert task.corner.temperature == 27
    assert task.param_overrides == {"M1.W": 2e-6}


def test_sim_task_normalizes_ingress_fields():
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        output_names=["out", "in", "out"],
        osdi_paths=["model.osdi"],
        metadata={"sample_index": 3},
        backend_options={"rawfile_format": "binary"},
        artifacts="artifacts",
    )

    assert task.output_names == ("out", "in")
    assert [str(path) for path in task.osdi_paths] == ["model.osdi"]
    assert task.metadata == {"sample_index": 3}
    assert task.backend_options == {"rawfile_format": "binary"}
    assert task.artifacts == SimArtifactOptions("artifacts")


@pytest.mark.parametrize(
    "output_name",
    [
        "",
        "out node",
        "out\nquit",
        "out;quit",
        "out'bad",
        'out"bad',
        "v(out",
        "@m1[gm",
    ],
)
def test_sim_task_rejects_malformed_output_names(output_name):
    with pytest.raises(ValueError, match="output name"):
        SimTask(
            circuit=None,
            analysis_spec=TranSpec(stop=1e-6),
            output_names=[output_name],
        )


def test_sim_task_output_names_must_be_strings():
    with pytest.raises(TypeError, match="output_names"):
        SimTask(
            circuit=None,
            analysis_spec=TranSpec(stop=1e-6),
            output_names=[object()],  # type: ignore[list-item]
        )


def test_sim_task_output_names_must_be_iterable_not_single_string():
    with pytest.raises(TypeError, match="not a string"):
        SimTask(
            circuit=None,
            analysis_spec=TranSpec(stop=1e-6),
            output_names="out",  # type: ignore[arg-type]
        )


def test_sim_task_copies_mapping_payloads():
    param_overrides = {"M1.W": 2e-6}
    metadata = {"sample_index": 3}
    backend_options = {"rawfile_format": "binary"}
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        param_overrides=MappingProxyType(param_overrides),
        metadata=MappingProxyType(metadata),
        backend_options=MappingProxyType(backend_options),
    )

    param_overrides["M1.W"] = 4e-6
    metadata["sample_index"] = 7
    backend_options["rawfile_format"] = "ascii"

    assert task.param_overrides == {"M1.W": 2e-6}
    assert task.metadata == {"sample_index": 3}
    assert task.backend_options == {"rawfile_format": "binary"}


def test_sim_task_payloads_are_read_only_after_creation():
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        param_overrides={"M1.W": 2e-6},
        output_names=["out"],
        osdi_paths=["model.osdi"],
        metadata={"sample_index": 3},
        backend_options={"rawfile_format": "binary"},
    )

    with pytest.raises(TypeError):
        task.param_overrides["M1.W"] = 4e-6  # type: ignore[index]
    with pytest.raises(AttributeError):
        task.output_names.append("in")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        task.osdi_paths.append("other.osdi")  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        task.metadata["sample_index"] = 7  # type: ignore[index]
    with pytest.raises(TypeError):
        task.backend_options["rawfile_format"] = "ascii"  # type: ignore[index]


def test_sim_artifact_options_coerces_supported_inputs(tmp_path):
    target = tmp_path / "artifacts"

    assert SimArtifactOptions.coerce(None) == SimArtifactOptions()
    assert SimArtifactOptions.coerce(str(target)) == SimArtifactOptions(target)
    assert SimArtifactOptions.coerce({"directory": target, "overwrite": True}) == SimArtifactOptions(
        target,
        overwrite=True,
    )

    options = SimArtifactOptions(target)
    assert SimArtifactOptions.coerce(options) is options
    assert options.to_dict() == {"directory": str(target), "overwrite": False}


def test_sim_artifact_options_rejects_unknown_input():
    with pytest.raises(TypeError, match="artifacts"):
        SimArtifactOptions.coerce(object())  # type: ignore[arg-type]


def test_sim_result_ok():
    t = np.linspace(0, 1e-6, 100)
    v = np.sin(2 * np.pi * 1e6 * t)
    result = SimResult(
        status="ok",
        waveforms={"out": v},
        sweep_var=t,
        corner=None,
        metadata={"simulator": "ngspice", "elapsed_time": 0.5},
    )
    assert result.status is SimStatus.OK
    assert result.status == "ok"
    assert result.error_message is None
    assert len(result.waveforms["out"]) == 100
    assert result.metadata["simulator"] == "ngspice"


def test_sim_result_failed():
    result = SimResult(
        status="failed",
        waveforms={},
        sweep_var=None,
        corner=None,
        error_message="convergence failure at t=3.2ns",
        metadata={"simulator": "ngspice"},
    )
    assert result.status is SimStatus.FAILED
    assert result.status == "failed"
    assert result.error_message == "convergence failure at t=3.2ns"
    assert result.waveforms == {}
    assert result.sweep_var is None


def test_sim_result_contract_is_frozen_dataclass():
    result = SimResult(status="ok", waveforms={}, sweep_var=None, corner=None)

    assert is_dataclass(result)
    with pytest.raises(FrozenInstanceError):
        result.status = SimStatus.FAILED  # type: ignore[misc]


def test_sim_result_copies_mapping_payloads():
    waveforms = {"out": np.array([1.0])}
    metadata = {"simulator": "ngspice"}
    summaries = {"gain": 2.0}
    result = SimResult(
        status="ok",
        waveforms=MappingProxyType(waveforms),
        sweep_var=None,
        corner=None,
        metadata=MappingProxyType(metadata),
        summaries=MappingProxyType(summaries),
    )

    waveforms["extra"] = np.array([0.0])
    metadata["simulator"] = "other"
    summaries["gain"] = 3.0

    assert set(result.waveforms) == {"out"}
    assert result.metadata == {"simulator": "ngspice"}
    assert result.summaries == {"gain": 2.0}


def test_sim_result_payloads_are_read_only_after_creation():
    result = SimResult(
        status="ok",
        waveforms={"out": np.array([1.0])},
        sweep_var=None,
        corner=None,
        metadata={"simulator": "ngspice"},
        summaries={"gain": 2.0},
    )

    with pytest.raises(TypeError):
        result.waveforms["extra"] = np.array([0.0])  # type: ignore[index]
    with pytest.raises(TypeError):
        result.metadata["simulator"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.summaries["gain"] = 3.0  # type: ignore[index]

    updated = result.with_summary("delay", 1e-9)

    assert "delay" not in result.summaries
    assert updated.summaries["gain"] == 2.0
    assert updated.summaries["delay"] == 1e-9
