import json

import pytest

from monata.sim.analysis_spec import TranSpec
from monata.sim.artifacts import persist_simulation_artifacts, simulation_artifact_dir
from monata.sim.task import SimTask


def test_simulation_artifact_dir_uses_task_artifact_options(tmp_path):
    target = tmp_path / "current"
    task = SimTask(circuit=None, analysis_spec=TranSpec(stop=1e-6), artifacts=target)

    assert simulation_artifact_dir(task) == target


def test_simulation_artifact_dir_ignores_metadata_control_keys(tmp_path):
    legacy_target = tmp_path / "legacy"
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        metadata={
            "artifact_dir": str(legacy_target),
            "simulation_artifact_dir": str(legacy_target),
        },
    )

    assert simulation_artifact_dir(task) is None
    assert persist_simulation_artifacts(task, simulator="unit", text_files={"stdout": "ignored"}) == {}
    assert not legacy_target.exists()


def test_persist_simulation_artifacts_uses_schema_file_names(tmp_path):
    target = tmp_path / "artifacts"
    rawfile = tmp_path / "custom-name.raw"
    rawfile.write_text("raw")
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        artifacts=target,
    )

    result = persist_simulation_artifacts(
        task,
        simulator="unit",
        files={"rawfile": rawfile},
        text_files={"stdout": "hello", "stderr": None},
        metadata={"status": "ok"},
    )

    assert (target / "result.raw").read_text() == "raw"
    assert (target / "stdout.txt").read_text() == "hello"
    assert (target / "stderr.txt").read_text() == ""
    payload = json.loads((target / "metadata.json").read_text())
    assert payload["files"]["rawfile"] == str(target / "result.raw")
    assert payload["files"]["stdout"] == str(target / "stdout.txt")
    assert payload["status"] == "ok"
    assert result["artifacts"]["files"]["metadata"] == str(target / "metadata.json")


def test_persist_simulation_artifacts_refuses_to_overwrite_existing_schema_files(tmp_path):
    target = tmp_path / "artifacts"
    first_rawfile = tmp_path / "first.raw"
    second_rawfile = tmp_path / "second.raw"
    first_rawfile.write_text("first")
    second_rawfile.write_text("second")
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        artifacts=target,
    )

    persist_simulation_artifacts(
        task,
        simulator="unit",
        files={"rawfile": first_rawfile},
        text_files={"stdout": "first stdout"},
    )

    with pytest.raises(FileExistsError, match="simulation artifact destination already exists"):
        persist_simulation_artifacts(
            task,
            simulator="unit",
            files={"rawfile": second_rawfile},
            text_files={"stdout": "second stdout"},
        )

    assert (target / "result.raw").read_text() == "first"
    assert (target / "stdout.txt").read_text() == "first stdout"


def test_persist_simulation_artifacts_can_overwrite_when_explicit(tmp_path):
    target = tmp_path / "artifacts"
    first_rawfile = tmp_path / "first.raw"
    second_rawfile = tmp_path / "second.raw"
    first_rawfile.write_text("first")
    second_rawfile.write_text("second")
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        artifacts=target,
    )

    persist_simulation_artifacts(task, simulator="unit", files={"rawfile": first_rawfile})
    persist_simulation_artifacts(task, simulator="unit", files={"rawfile": second_rawfile}, overwrite=True)

    assert (target / "result.raw").read_text() == "second"


def test_persist_simulation_artifacts_can_overwrite_from_task_options(tmp_path):
    target = tmp_path / "artifacts"
    first_rawfile = tmp_path / "first.raw"
    second_rawfile = tmp_path / "second.raw"
    first_rawfile.write_text("first")
    second_rawfile.write_text("second")
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        artifacts={"directory": target, "overwrite": True},
    )

    persist_simulation_artifacts(task, simulator="unit", files={"rawfile": first_rawfile})
    persist_simulation_artifacts(task, simulator="unit", files={"rawfile": second_rawfile})

    assert (target / "result.raw").read_text() == "second"


def test_persist_simulation_artifacts_rejects_unknown_file_key(tmp_path):
    target = tmp_path / "artifacts"
    source = tmp_path / "extra.dat"
    source.write_text("extra")
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        artifacts=target,
    )

    with pytest.raises(ValueError, match="unknown simulation artifact key: custom"):
        persist_simulation_artifacts(task, simulator="unit", files={"custom": source})

    assert not target.exists()


def test_persist_simulation_artifacts_rejects_unknown_text_key(tmp_path):
    target = tmp_path / "artifacts"
    task = SimTask(
        circuit=None,
        analysis_spec=TranSpec(stop=1e-6),
        artifacts=target,
    )

    with pytest.raises(ValueError, match=r"unknown simulation artifact key: \.\./escape"):
        persist_simulation_artifacts(task, simulator="unit", text_files={"../escape": "nope"})

    assert not target.exists()
