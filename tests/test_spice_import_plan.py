from pathlib import Path

import pytest

from monata.netlist import render_ngspice
from monata.parser import (
    SpiceBinary,
    SpiceCall,
    SpiceControlCommand,
    SpiceGroup,
    SpiceImportExpressionCheck,
    SpiceSourceDependency,
    UnsupportedConstructError,
    inspect_spice_import,
)


def test_inspect_spice_import_classifies_projection_preservation_and_flow():
    plan = inspect_spice_import(
        """
import plan
.param rload=1k
.subckt gain in out
R1 in out {rload}
.ends gain
X1 a b gain
.control
run
.endc
.tran 1n 10n
.end
"""
    )

    assert plan.title == "import plan"
    assert plan.supported is True
    assert plan.projected_count == 3
    assert plan.preserved_count == 2
    assert plan.unsupported_count == 0
    assert [step.action for step in plan.steps if step.name in {"subckt", "ends", "end"}] == [
        "flow",
        "flow",
        "flow",
    ]

    circuit = plan.to_circuit()
    rendered = render_ngspice(circuit)
    assert ".subckt gain in out" in rendered
    assert ".tran 1n 10n" in rendered


def test_inspect_spice_import_tracks_title_directive_as_metadata():
    plan = inspect_spice_import(
        """
placeholder title
.title final import plan title
R1 in out 1k
.end
"""
    )

    title_steps = [step for step in plan.steps if step.kind == "title"]
    assert plan.title == "final import plan title"
    assert [step.action for step in title_steps] == ["metadata", "metadata"]
    assert title_steps[-1].detail == "title directive"
    assert render_ngspice(plan.to_circuit()).splitlines()[0] == "final import plan title"


def test_inspect_spice_import_classifies_standalone_comments_as_preserved():
    plan = inspect_spice_import(
        """
comment plan
* top-level note
R1 in out 1k
.end
"""
    )

    comment_steps = [step for step in plan.steps if step.kind == "comment"]

    assert [(step.action, step.name, step.raw, step.detail) for step in comment_steps] == [
        ("preserve", "comment", "* top-level note", "standalone comment")
    ]
    assert "* top-level note" in render_ngspice(plan.to_circuit())


def test_inspect_spice_import_classifies_control_block_commands_without_projecting_them():
    plan = inspect_spice_import(
        """
control commands
R1 in out 1k
.control
set noaskquit
tran 1n 10n
meas tran vout FIND v(out) AT=1n
plot v(out)
wrdata out.dat v(out)
shell echo unsafe
.endc
.end
"""
    )

    assert plan.supported is True
    assert plan.control_command_count == 6
    assert plan.migratable_control_count == 2

    by_name = {command.name: command for command in plan.control_commands}
    assert isinstance(by_name["tran"], SpiceControlCommand)
    assert by_name["tran"].action == "analysis"
    assert by_name["tran"].migratable is True
    assert by_name["meas"].action == "measurement"
    assert by_name["set"].action == "state"
    assert by_name["plot"].action == "output"
    assert by_name["wrdata"].action == "side_effect"
    assert by_name["shell"].action == "side_effect"
    assert any("wrdata" in issue.message for issue in plan.issues)

    rendered = render_ngspice(plan.to_circuit())
    assert ".control" in rendered
    assert "tran 1n 10n" in rendered
    assert "wrdata out.dat v(out)" in rendered


def test_inspect_spice_import_reports_source_dependencies(tmp_path):
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    (deck_dir / "local.inc").write_text("* local models\n")
    (vendor_dir / "models.lib").write_text(".lib tt\n.model n nmos level=1\n.endl tt\n")
    deck_path = deck_dir / "main.cir"
    deck_path.write_text(
        """
dependency plan
.include "local.inc"
.lib models.lib tt
.include missing.mod
R1 in out 1k
.end
"""
    )

    plan = inspect_spice_import(deck_path, include_paths=[vendor_dir])

    assert plan.supported is True
    assert plan.source_dependency_count == 3
    assert plan.missing_source_dependency_count == 1

    by_target = {dependency.target: dependency for dependency in plan.source_dependencies}
    assert isinstance(by_target["local.inc"], SpiceSourceDependency)
    assert by_target["local.inc"].kind == "include"
    assert by_target["local.inc"].exists is True
    assert by_target["local.inc"].resolved_path == deck_dir / "local.inc"
    assert by_target["models.lib"].kind == "lib"
    assert by_target["models.lib"].section == "tt"
    assert by_target["models.lib"].resolved_path == vendor_dir / "models.lib"
    assert by_target["missing.mod"].exists is False
    assert by_target["missing.mod"].status == "missing"
    assert any("missing.mod" in issue.message for issue in plan.issues)

    rendered = render_ngspice(plan.to_circuit())
    assert "local.inc" in rendered
    assert ".lib models.lib tt" in rendered


def test_inspect_spice_import_leaves_inline_relative_dependencies_unchecked():
    plan = inspect_spice_import(
        """
inline dependencies
.include rel.lib
R1 in out 1k
.end
"""
    )

    assert plan.supported is True
    assert plan.source_dependency_count == 1
    dependency = plan.source_dependencies[0]
    assert dependency.target == "rel.lib"
    assert dependency.exists is None
    assert dependency.status == "unchecked"
    assert dependency.search_paths == ()
    assert any("not checked" in issue.message for issue in plan.issues)


def test_inspect_spice_import_reports_unsupported_constructs_without_throwing():
    plan = inspect_spice_import(
        """
unsupported plan
.foo bar
R1 in out 1k
.end
"""
    )

    assert plan.supported is False
    assert plan.unsupported_count == 1
    assert plan.issues[0].message == "unsupported dot command: .foo"
    assert plan.issues[0].line == 3
    assert plan.steps[1].action == "unsupported"
    assert plan.steps[2].action == "project"
    with pytest.raises(UnsupportedConstructError, match="unsupported dot command: .foo"):
        plan.to_circuit()


def test_inspect_spice_import_checks_expression_fields_without_blocking_projection():
    plan = inspect_spice_import(
        """
expression plan
.param rload={sqrt(4)} bad=2k3
.model nmos nmos level=1 vto={-0.4}
.func double(x) {x*2}
.if rload > 1k
R1 in out {rload}
.endif
B1 out 0 v={limit(v(in), -1, 1)}
.end
"""
    )

    assert plan.supported is True
    assert plan.expression_count == 8
    assert plan.parsed_expression_count == 7
    assert plan.failed_expression_count == 1
    assert any(issue.severity == "warning" and "bad" in issue.message for issue in plan.issues)

    rload = next(check for check in plan.expression_checks if check.owner_name == "param" and check.field == "param.rload")
    condition = next(check for check in plan.expression_checks if check.owner_name == "if")
    bsource = next(check for check in plan.expression_checks if check.owner_name == "B1")

    assert isinstance(rload, SpiceImportExpressionCheck)
    assert isinstance(rload.expression, SpiceGroup)
    assert isinstance(rload.expression.expression, SpiceCall)
    assert isinstance(condition.expression, SpiceBinary)
    assert isinstance(bsource.expression, SpiceGroup)
    assert isinstance(bsource.expression.expression, SpiceCall)
    rendered = render_ngspice(plan.to_circuit())
    assert ".param bad=2k3" in rendered
    assert "B1 out 0 v={limit(v(in), -1, 1)}" in rendered

    lt_plan = inspect_spice_import(
        """
lt expression plan
.param rload=2k3
R1 in out 4R7
.end
""",
        expression_dialect="ltspice",
    )

    assert lt_plan.supported is True
    assert lt_plan.expression_count == 2
    assert lt_plan.parsed_expression_count == 2
    assert lt_plan.failed_expression_count == 0


def test_inspect_spice_import_reads_only_explicit_path_objects(tmp_path):
    deck_path = tmp_path / "from_file.cir"
    deck_path.write_text("file plan\nR1 in out 1k\n.end\n")

    from_path = inspect_spice_import(deck_path)
    inline = inspect_spice_import(str(deck_path))

    assert from_path.path == str(deck_path)
    assert from_path.title == "file plan"
    assert inline.path is None
    assert inline.title == str(deck_path)


def test_import_plan_path_accepts_path_subclasses(tmp_path):
    class CustomPath(Path):
        _flavour = type(tmp_path)._flavour

    deck_path = CustomPath(tmp_path / "custom.cir")
    deck_path.write_text("custom plan\nR1 in out 1k\n.end\n")

    assert inspect_spice_import(deck_path).title == "custom plan"
