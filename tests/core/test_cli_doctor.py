from __future__ import annotations

def test_doctor_reports_missing_runtime_tools_and_recommended_skill(monkeypatch, capsys, tmp_path):
    from monata.cli import main

    monkeypatch.delenv("MONATA_HOME", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "Monata environment doctor" in out
    assert "ngspice: missing" in out
    assert "openvaf-r: missing" in out
    assert "MONATA_HOME: not set" in out
    assert "monata-sim-env" in out
    assert "npx skills@latest add lizhangmai/skills --skill monata-sim-env" in out
    assert "Use the monata-sim-env skill to set up this complete Monata environment" in out


def test_doctor_passes_when_runtime_tools_and_techlibs_are_available(monkeypatch, capsys, tmp_path):
    from monata.cli import main

    monata_home = tmp_path / "monata-home"
    techlibs = monata_home / "techlibs"
    (techlibs / "PTM_BULK").mkdir(parents=True)
    (techlibs / "PTM_MG").mkdir()

    def fake_which(name: str) -> str | None:
        return f"/tools/bin/{name}" if name in {"ngspice", "openvaf-r"} else None

    monkeypatch.setenv("MONATA_HOME", str(monata_home))
    monkeypatch.setattr("shutil.which", fake_which)

    rc = main(["doctor"])

    out = capsys.readouterr().out
    assert rc == 0
    assert f"MONATA_HOME: {monata_home}" in out
    assert "ngspice: /tools/bin/ngspice" in out
    assert "openvaf-r: /tools/bin/openvaf-r" in out
    assert f"techlibs: {techlibs}" in out
    assert "PTM_BULK" in out
    assert "PTM_MG" in out


def test_python_module_entrypoint_dispatches_doctor(monkeypatch, capsys, tmp_path):
    from monata.__main__ import main

    monkeypatch.setenv("MONATA_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")

    rc = main(["doctor"])

    assert rc == 1
    assert "techlibs: missing" in capsys.readouterr().out
