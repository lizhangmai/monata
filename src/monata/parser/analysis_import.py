"""Import SPICE analysis commands into Monata simulation tasks.

Analysis import is intentionally separate from netlist import. The base circuit
remains native `Circuit` IR, while `.tran`, `.ac`, and related commands become
backend-neutral `AnalysisSpec`/`SimTask` records.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from monata.netlist.ir import Directive
from monata.parser.deck import ControlBlock, DotCommand, SpiceDeck, UnsupportedStatement, parse_spice
from monata.parser.errors import UnsupportedConstructError
from monata.units import UnitError, parse_spice_number

if TYPE_CHECKING:
    from monata.netlist import Circuit
    from monata.sim.task import SimTask


AnalysisAction = Literal["task", "unsupported"]
AnalysisIssueSeverity = Literal["error", "warning"]
AnalysisOrigin = Literal["deck", "control"]
AnalysisSweepKind = Literal["param", "temperature"]
AnalysisSweepMode = Literal["list", "linear", "dec", "oct"]

_DEFAULT_SIMULATOR = "ngspice-subprocess"
_DEFAULT_SIM_TIMEOUT_SECONDS = 300.0
_ANALYSIS_COMMANDS = frozenset(
    {
        "ac",
        "dc",
        "disto",
        "distortion",
        "four",
        "fourier",
        "noise",
        "op",
        "pz",
        "sens",
        "tf",
        "tran",
    }
)
_MEASUREMENT_COMMANDS = frozenset({"meas", "measure"})
_OUTPUT_DIRECTIVES = frozenset({"plot", "print", "probe", "save"})
_STEP_COMMANDS = frozenset({"step"})
_CONTROL_IGNORED_COMMANDS = frozenset({"exit", "quit", "run"})
_CONTROL_TASK_COMMANDS = _ANALYSIS_COMMANDS | _MEASUREMENT_COMMANDS | _OUTPUT_DIRECTIVES | _CONTROL_IGNORED_COMMANDS


@dataclass(frozen=True)
class SpiceAnalysisIssue:
    """A diagnostic found while converting SPICE analysis commands."""

    severity: AnalysisIssueSeverity
    message: str
    line: int | None = None
    source_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class SpiceAnalysisStep:
    """One SPICE analysis command import decision."""

    action: AnalysisAction
    name: str
    raw: str
    line: int
    source_lines: tuple[int, ...]
    analysis_spec: Any = None
    output_names: tuple[str, ...] = ()
    detail: str = ""
    origin: AnalysisOrigin = "deck"


@dataclass(frozen=True)
class SpiceAnalysisMeasurement:
    """A SPICE `.measure` statement associated with an imported analysis."""

    analysis: str
    name: str
    expression: str
    raw: str
    line: int
    source_lines: tuple[int, ...]
    origin: AnalysisOrigin = "deck"

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-compatible record for SimTask metadata."""

        return {
            "analysis": self.analysis,
            "name": self.name,
            "expression": self.expression,
            "raw": self.raw,
            "line": self.line,
            "source_lines": self.source_lines,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class _TaskControlBlock:
    line: int
    commands: tuple[DotCommand, ...]
    issues: tuple[SpiceAnalysisIssue, ...] = ()


@dataclass(frozen=True)
class SpiceAnalysisSweep:
    """A `.step param ...` sweep that expands imported analysis tasks."""

    target: str
    values: tuple[Any, ...]
    raw: str
    line: int
    source_lines: tuple[int, ...]
    mode: AnalysisSweepMode = "list"
    kind: AnalysisSweepKind = "param"

    def to_metadata(self, value: Any) -> dict[str, Any]:
        """Return the active sweep point as JSON-compatible task metadata."""

        return {
            "kind": self.kind,
            "target": self.target,
            "value": value,
            "mode": self.mode,
            "raw": self.raw,
            "line": self.line,
            "source_lines": self.source_lines,
        }


@dataclass(frozen=True)
class SpiceAnalysisPlan:
    """Inspectable plan for turning deck analysis commands into SimTasks."""

    title: str
    path: str | None
    steps: tuple[SpiceAnalysisStep, ...]
    issues: tuple[SpiceAnalysisIssue, ...] = ()
    measurements: tuple[SpiceAnalysisMeasurement, ...] = ()
    sweeps: tuple[SpiceAnalysisSweep, ...] = ()
    _source_text: str = field(repr=False, compare=False, default="")

    @property
    def supported(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def task_count(self) -> int:
        return sum(1 for step in self.steps if step.action == "task") * _sweep_combination_count(self.sweeps)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for step in self.steps if step.action == "unsupported")

    @property
    def measurement_count(self) -> int:
        return len(self.measurements)

    @property
    def sweep_count(self) -> int:
        return len(self.sweeps)

    def measurements_for(self, analysis: str) -> tuple[SpiceAnalysisMeasurement, ...]:
        """Return `.measure` statements bound to a canonical analysis name."""

        canonical = _canonical_analysis_name(analysis)
        return tuple(measurement for measurement in self.measurements if measurement.analysis == canonical)

    def require_supported(self) -> None:
        for issue in self.issues:
            if issue.severity == "error":
                raise UnsupportedConstructError(issue.message, path=self.path, line=issue.line)

    def base_circuit(self) -> Circuit:
        """Return native Circuit IR with deck-level analysis commands removed."""

        self.require_supported()
        return _base_circuit(self._source_text, path=self.path)

    def to_tasks(
        self,
        *,
        simulator: str = _DEFAULT_SIMULATOR,
        corner: Any = None,
        param_overrides: dict[str, Any] | None = None,
        osdi_paths: tuple[str | Path, ...] = (),
        metadata: dict[str, Any] | None = None,
        timeout: float | int | None = _DEFAULT_SIM_TIMEOUT_SECONDS,
    ) -> tuple[SimTask, ...]:
        """Project supported analysis commands into SimTask records."""

        self.require_supported()
        from monata.sim.task import SimTask

        circuit = self.base_circuit()
        shared_metadata = dict(metadata or {})
        tasks = []
        sweep_combinations = _sweep_combinations(self.sweeps)
        for step in self.steps:
            if step.action != "task":
                continue
            for sweep_points in sweep_combinations:
                sweep_overrides = _sweep_overrides(sweep_points)
                task_measurements = self.measurements_for(step.name)
                task_metadata = {
                    **shared_metadata,
                    "import_source": self.path or "<memory>",
                    "import_analysis": step.name,
                    "import_line": step.line,
                    "import_raw": step.raw,
                    "import_origin": step.origin,
                }
                if task_measurements:
                    task_metadata["import_measurements"] = tuple(
                        measurement.to_metadata() for measurement in task_measurements
                    )
                if sweep_points:
                    task_metadata["import_sweeps"] = tuple(
                        sweep.to_metadata(value) for sweep, value in sweep_points
                    )
                    task_metadata["import_sweep_overrides"] = dict(sweep_overrides)
                task_circuit = _task_circuit(circuit, analysis_name=step.name)
                _ensure_measure_directives(task_circuit, task_measurements)
                tasks.append(
                    SimTask(
                        circuit=task_circuit,
                        analysis_spec=step.analysis_spec,
                        simulator=simulator,
                        corner=_sweep_corner(corner, sweep_points),
                        param_overrides=_merged_param_overrides(param_overrides, sweep_overrides),
                        output_names=step.output_names,
                        osdi_paths=osdi_paths,
                        metadata=task_metadata,
                        timeout=timeout,
                    )
                )
        return tuple(tasks)


def inspect_spice_analysis(text_or_path: str | Path) -> SpiceAnalysisPlan:
    """Inspect SPICE analysis commands in text or an explicit Path."""

    text, source_path = _read_text_or_path(text_or_path)
    deck = parse_spice(text, path=source_path, strict=False)
    control_blocks = _task_control_blocks(deck)
    control_commands = tuple(command for block in control_blocks for command in block.commands)
    control_issues = tuple(issue for block in control_blocks for issue in block.issues)
    control_commands_by_line = {block.line: block.commands for block in control_blocks}
    output_commands = _output_commands(deck, control_commands=control_commands)
    measurements, measurement_issues = _measurements(deck, control_commands=control_commands)
    sweeps, sweep_issues = _sweeps(deck)
    steps: list[SpiceAnalysisStep] = []
    issues: list[SpiceAnalysisIssue] = [*control_issues, *measurement_issues, *sweep_issues]
    last_tran = None

    for statement in deck.statements:
        if isinstance(statement, UnsupportedStatement):
            step = SpiceAnalysisStep(
                "unsupported",
                "unsupported",
                statement.text,
                statement.line,
                statement.source_lines,
                detail=statement.message,
            )
            steps.append(step)
            issues.append(SpiceAnalysisIssue("error", statement.message, statement.line, statement.source_lines))
            continue
        if not isinstance(statement, DotCommand) or statement.name not in _ANALYSIS_COMMANDS:
            if isinstance(statement, ControlBlock):
                for command in control_commands_by_line.get(statement.line, ()):
                    if command.name not in _ANALYSIS_COMMANDS:
                        continue
                    step, step_issues = _analysis_step(
                        command,
                        output_commands=output_commands,
                        last_tran=last_tran,
                        origin="control",
                    )
                    steps.append(step)
                    issues.extend(step_issues)
                    if step.action == "task" and step.name == "tran":
                        last_tran = step.analysis_spec
            continue
        step, step_issues = _analysis_step(statement, output_commands=output_commands, last_tran=last_tran)
        steps.append(step)
        issues.extend(step_issues)
        if step.action == "task" and step.name == "tran":
            last_tran = step.analysis_spec

    return SpiceAnalysisPlan(
        title=deck.title,
        path=deck.path,
        steps=tuple(steps),
        measurements=measurements,
        sweeps=sweeps,
        issues=tuple(issues),
        _source_text=text,
    )


def spice_to_sim_tasks(
    text_or_path: str | Path,
    *,
    simulator: str = _DEFAULT_SIMULATOR,
    corner: Any = None,
    param_overrides: dict[str, Any] | None = None,
    osdi_paths: tuple[str | Path, ...] = (),
    metadata: dict[str, Any] | None = None,
    timeout: float | int | None = _DEFAULT_SIM_TIMEOUT_SECONDS,
) -> tuple[SimTask, ...]:
    """Convert SPICE deck analysis commands into Monata SimTasks."""

    return inspect_spice_analysis(text_or_path).to_tasks(
        simulator=simulator,
        corner=corner,
        param_overrides=param_overrides,
        osdi_paths=osdi_paths,
        metadata=metadata,
        timeout=timeout,
    )


def _analysis_step(
    command: DotCommand,
    *,
    output_commands: tuple[DotCommand, ...],
    last_tran: Any,
    origin: AnalysisOrigin = "deck",
) -> tuple[SpiceAnalysisStep, tuple[SpiceAnalysisIssue, ...]]:
    try:
        spec = _analysis_spec(command, last_tran=last_tran)
    except ValueError as exc:
        return (
            SpiceAnalysisStep(
                "unsupported",
                command.name,
                command.raw,
                command.line,
                command.source_lines,
                detail=str(exc),
                origin=origin,
            ),
            (SpiceAnalysisIssue("error", str(exc), command.line, command.source_lines),),
        )
    output_names, output_issues = _output_names(command, output_commands)
    issues = list(output_issues)
    canonical_name = _canonical_analysis_name(command.name)
    action: AnalysisAction = "unsupported" if any(issue.severity == "error" for issue in issues) else "task"
    return (
        SpiceAnalysisStep(
            action,
            canonical_name,
            command.raw,
            command.line,
            command.source_lines,
            analysis_spec=spec if action == "task" else None,
            output_names=output_names if action == "task" else (),
            detail=f".{command.name} analysis",
            origin=origin,
        ),
        tuple(issues),
    )


def _analysis_spec(command: DotCommand, *, last_tran: Any) -> Any:
    name = command.name
    args = tuple(command.args)
    from monata.sim.analysis_spec import (
        ACSpec,
        DCSweep,
        DCSpec,
        DistortionSpec,
        FourierSpec,
        NoiseSpec,
        OPSpec,
        PoleZeroSpec,
        SensitivitySpec,
        TranSpec,
        TransferFunctionSpec,
    )

    if name == "op":
        if args:
            raise ValueError(".op import does not accept arguments")
        return OPSpec()
    if name == "tran":
        return _tran_spec(args, TranSpec)
    if name == "ac":
        if len(args) != 4:
            raise ValueError(".ac import expects: variation points start stop")
        return ACSpec(start=_number(args[2], ".ac start"), stop=_number(args[3], ".ac stop"), points=_points(args[1], ".ac points"), variation=args[0])
    if name == "dc":
        if len(args) not in {4, 8}:
            raise ValueError(".dc import expects: source start stop step [source2 start2 stop2 step2]")
        secondary = None
        if len(args) == 8:
            secondary = DCSweep(
                source=args[4],
                start=_number(args[5], ".dc secondary start"),
                stop=_number(args[6], ".dc secondary stop"),
                step=_number(args[7], ".dc secondary step"),
            )
        return DCSpec(
            source=args[0],
            start=_number(args[1], ".dc start"),
            stop=_number(args[2], ".dc stop"),
            step=_number(args[3], ".dc step"),
            secondary=secondary,
        )
    if name == "noise":
        if len(args) not in {6, 7}:
            raise ValueError(".noise import expects: v(output[,ref]) input_source variation points start stop [points_per_summary]")
        output_node, reference_node = _nodes_from_voltage(args[0], ".noise output")
        return NoiseSpec(
            output_node=output_node,
            input_source=args[1],
            points=_points(args[3], ".noise points"),
            start=_number(args[4], ".noise start"),
            stop=_number(args[5], ".noise stop"),
            reference_node=reference_node,
            variation=args[2],
            points_per_summary=_positive_points(args[6], ".noise points_per_summary") if len(args) == 7 else None,
        )
    if name == "sens":
        if len(args) == 1:
            return SensitivitySpec(output=args[0])
        if len(args) == 6 and args[1].lower() == "ac":
            return SensitivitySpec(
                output=args[0],
                variation=args[2],
                points=_points(args[3], ".sens points"),
                start=_number(args[4], ".sens start"),
                stop=_number(args[5], ".sens stop"),
            )
        raise ValueError(".sens import expects output or: output ac variation points start stop")
    if name == "pz":
        if len(args) != 6:
            raise ValueError(".pz import expects: in+ in- out+ out- transfer mode")
        return PoleZeroSpec(args[0], args[1], args[2], args[3], transfer=args[4], mode=args[5])
    if name in {"disto", "distortion"}:
        if len(args) not in {4, 5}:
            raise ValueError(".disto import expects: variation points start stop [f2overf1]")
        return DistortionSpec(
            start=_number(args[2], ".disto start"),
            stop=_number(args[3], ".disto stop"),
            points=_points(args[1], ".disto points"),
            variation=args[0],
            f2overf1=_number(args[4], ".disto f2overf1") if len(args) == 5 else None,
        )
    if name == "tf":
        if len(args) != 2:
            raise ValueError(".tf import expects: output input_source")
        return TransferFunctionSpec(output=args[0], input_source=args[1])
    if name in {"four", "fourier"}:
        if last_tran is None:
            raise ValueError(f".{name} import requires an earlier supported .tran command")
        if len(args) != 2:
            raise ValueError(f".{name} import expects: frequency output")
        return FourierSpec(
            frequency=_number(args[0], f".{name} frequency"),
            output=args[1],
            stop=last_tran.stop,
            step=last_tran.step,
            start=last_tran.start,
        )
    raise ValueError(f"unsupported analysis command: .{name}")


def _tran_spec(args: tuple[str, ...], tran_spec_type: Any) -> Any:
    uic = any(arg.lower() == "uic" for arg in args)
    values = tuple(arg for arg in args if arg.lower() != "uic")
    if len(values) not in {2, 3, 4}:
        raise ValueError(".tran import expects: step stop [start [max_step]] [uic]")
    return tran_spec_type(
        stop=_number(values[1], ".tran stop"),
        step=_number(values[0], ".tran step"),
        start=_number(values[2], ".tran start") if len(values) >= 3 else 0,
        max_step=_number(values[3], ".tran max_step") if len(values) == 4 else None,
        uic=uic,
    )


def _output_names(command: DotCommand, output_commands: tuple[DotCommand, ...]) -> tuple[tuple[str, ...], tuple[SpiceAnalysisIssue, ...]]:
    analysis_name = _canonical_analysis_name(command.name)
    if analysis_name == "noise":
        return (), ()
    raw_outputs = _raw_outputs_for(analysis_name, output_commands)
    output_names: list[str] = []
    issues: list[SpiceAnalysisIssue] = []
    seen: set[str] = set()
    for output_command, raw_output in raw_outputs:
        try:
            output_name = _output_name_for_analysis(raw_output, analysis_name)
        except ValueError as exc:
            issues.append(SpiceAnalysisIssue("error", str(exc), output_command.line, output_command.source_lines))
            continue
        if output_name not in seen:
            seen.add(output_name)
            output_names.append(output_name)
    return tuple(output_names), tuple(issues)


def _raw_outputs_for(analysis_name: str, output_commands: tuple[DotCommand, ...]) -> tuple[tuple[DotCommand, str], ...]:
    outputs: list[tuple[DotCommand, str]] = []
    for command in output_commands:
        if command.name in {"plot", "save", "probe"}:
            outputs.extend((command, arg) for arg in command.args)
        elif command.name == "print" and command.args and _canonical_analysis_name(command.args[0]) == analysis_name:
            outputs.extend((command, arg) for arg in command.args[1:])
    return tuple(outputs)


def _output_name_for_analysis(raw_output: str, analysis_name: str) -> str:
    text = raw_output.strip()
    if not text:
        raise ValueError("empty output vector in imported analysis")
    lowered = text.lower()
    if lowered == "all" or lowered.startswith("@"):
        raise ValueError(f"unsupported imported output vector for SimTask: {raw_output}")
    if analysis_name == "ac":
        return text
    if lowered.startswith("v(") and text.endswith(")"):
        node = text[2:-1]
        if "," in node:
            raise ValueError(f"differential voltage output is not representable as a node output: {raw_output}")
        return node
    if lowered.startswith(("i(", "db(", "mag(", "phase(", "real(", "imag(")):
        raise ValueError(f"unsupported imported output vector for .{analysis_name}: {raw_output}")
    return text


def _output_commands(deck: SpiceDeck, *, control_commands: tuple[DotCommand, ...] = ()) -> tuple[DotCommand, ...]:
    deck_commands = tuple(
        statement
        for statement in deck.statements
        if isinstance(statement, DotCommand) and statement.name in _OUTPUT_DIRECTIVES
    )
    return (*deck_commands, *(command for command in control_commands if command.name in _OUTPUT_DIRECTIVES))


def _measurements(
    deck: SpiceDeck,
    *,
    control_commands: tuple[DotCommand, ...] = (),
) -> tuple[tuple[SpiceAnalysisMeasurement, ...], tuple[SpiceAnalysisIssue, ...]]:
    measurements: list[SpiceAnalysisMeasurement] = []
    issues: list[SpiceAnalysisIssue] = []
    commands: list[tuple[DotCommand, AnalysisOrigin]] = [
        (statement, "deck")
        for statement in deck.statements
        if isinstance(statement, DotCommand) and statement.name in _MEASUREMENT_COMMANDS
    ]
    commands.extend((command, "control") for command in control_commands if command.name in _MEASUREMENT_COMMANDS)
    for statement, origin in commands:
        if len(statement.args) < 3:
            message = f".{statement.name} import requires analysis, name, and expression"
            issues.append(SpiceAnalysisIssue("error", message, statement.line, statement.source_lines))
            continue
        measurements.append(
            SpiceAnalysisMeasurement(
                analysis=_canonical_analysis_name(statement.args[0]),
                name=statement.args[1],
                expression=" ".join(statement.args[2:]),
                raw=statement.raw,
                line=statement.line,
                source_lines=statement.source_lines,
                origin=origin,
            )
        )
    return tuple(measurements), tuple(issues)


def _sweeps(deck: SpiceDeck) -> tuple[tuple[SpiceAnalysisSweep, ...], tuple[SpiceAnalysisIssue, ...]]:
    sweeps: list[SpiceAnalysisSweep] = []
    issues: list[SpiceAnalysisIssue] = []
    for statement in deck.statements:
        if not isinstance(statement, DotCommand) or statement.name not in _STEP_COMMANDS:
            continue
        try:
            sweeps.append(_step_sweep(statement))
        except ValueError as exc:
            issues.append(SpiceAnalysisIssue("error", str(exc), statement.line, statement.source_lines))
    temperature_sweeps = [sweep for sweep in sweeps if sweep.kind == "temperature"]
    if len(temperature_sweeps) > 1:
        issues.append(
            SpiceAnalysisIssue(
                "error",
                ".step import supports at most one temperature sweep; combine temperatures into one .step temp command",
                temperature_sweeps[1].line,
                temperature_sweeps[1].source_lines,
            )
        )
    return tuple(sweeps), tuple(issues)


def _step_sweep(command: DotCommand) -> SpiceAnalysisSweep:
    args = tuple(command.args)
    if not args:
        raise ValueError(".step import expects: param name list values... or param name start stop step")
    first = args[0].lower()
    if first == "dec":
        return _mode_prefixed_step_sweep(command, "dec")
    if first == "oct":
        return _mode_prefixed_step_sweep(command, "oct")
    if first in {"temp", "temperature"}:
        if len(args) < 3:
            raise ValueError(".step temperature import expects: temp list values... or temp start stop step")
        values, mode = _step_values(args[1:], label=".step temperature")
        return SpiceAnalysisSweep(
            "temperature",
            values,
            command.raw,
            command.line,
            command.source_lines,
            mode=mode,
            kind="temperature",
        )
    if first == "param":
        if len(args) < 4:
            raise ValueError(".step param import expects: param name list values... or param name start stop step")
        target = args[1]
        spec = args[2:]
    else:
        if len(args) < 3:
            raise ValueError(".step import expects: name list values... or name start stop step")
        target = args[0]
        spec = args[1:]
    if not target:
        raise ValueError(".step import requires a parameter name")
    values, mode = _step_values(spec, label=".step")
    return SpiceAnalysisSweep(target, values, command.raw, command.line, command.source_lines, mode=mode)


def _mode_prefixed_step_sweep(command: DotCommand, mode: Literal["dec", "oct"]) -> SpiceAnalysisSweep:
    args = tuple(command.args[1:])
    label = f".step {mode}"
    if not args:
        raise ValueError(f"{label} import expects: {mode} param name start stop points")
    first = args[0].lower()
    if first in {"temp", "temperature"}:
        if len(args) < 4:
            raise ValueError(f"{label} temperature import expects: {mode} temp start stop points")
        values, sweep_mode = _step_values((mode, *args[1:]), label=f"{label} temperature")
        return SpiceAnalysisSweep(
            "temperature",
            values,
            command.raw,
            command.line,
            command.source_lines,
            mode=sweep_mode,
            kind="temperature",
        )
    if first == "param":
        if len(args) < 5:
            raise ValueError(f"{label} param import expects: {mode} param name start stop points")
        target = args[1]
        spec = args[2:]
    else:
        if len(args) < 4:
            raise ValueError(f"{label} import expects: {mode} name start stop points")
        target = args[0]
        spec = args[1:]
    if not target:
        raise ValueError(f"{label} import requires a parameter name")
    values, sweep_mode = _step_values((mode, *spec), label=label)
    return SpiceAnalysisSweep(target, values, command.raw, command.line, command.source_lines, mode=sweep_mode)


def _step_values(spec: tuple[str, ...], *, label: str) -> tuple[tuple[Any, ...], AnalysisSweepMode]:
    if spec[0].lower() == "list":
        if len(spec) < 2:
            raise ValueError(f"{label} list import requires at least one value")
        return tuple(spec[1:]), "list"
    if spec[0].lower() == "dec":
        if len(spec) != 4:
            raise ValueError(f"{label} dec import expects start stop points")
        start = _number(spec[1], f"{label} start")
        stop = _number(spec[2], f"{label} stop")
        points = _positive_points(spec[3], f"{label} points")
        return _log_step_values(start, stop, points, base=10.0, label=label), "dec"
    if spec[0].lower() == "oct":
        if len(spec) != 4:
            raise ValueError(f"{label} oct import expects start stop points")
        start = _number(spec[1], f"{label} start")
        stop = _number(spec[2], f"{label} stop")
        points = _positive_points(spec[3], f"{label} points")
        return _log_step_values(start, stop, points, base=2.0, label=label), "oct"
    if len(spec) != 3:
        raise ValueError(f"{label} linear import expects start stop step")
    start = _number(spec[0], f"{label} start")
    stop = _number(spec[1], f"{label} stop")
    step = _number(spec[2], f"{label} step")
    return _linear_step_values(start, stop, step), "linear"


def _linear_step_values(start: float, stop: float, step: float) -> tuple[float, ...]:
    if step == 0:
        raise ValueError(".step step must be non-zero")
    if (stop - start) * step < 0:
        raise ValueError(".step step sign does not move from start toward stop")
    values: list[float] = []
    value = start
    epsilon = abs(step) * 1e-9 + 1e-15
    if step > 0:
        while value <= stop + epsilon:
            values.append(value)
            value += step
    else:
        while value >= stop - epsilon:
            values.append(value)
            value += step
    return tuple(values)


def _log_step_values(start: float, stop: float, points: int, *, base: float, label: str) -> tuple[float, ...]:
    if start <= 0 or stop <= 0:
        raise ValueError(f"{label} logarithmic import requires positive start and stop")
    if stop < start:
        raise ValueError(f"{label} logarithmic import expects stop to be greater than or equal to start")
    values: list[float] = []
    multiplier = base ** (1.0 / points)
    value = start
    epsilon = max(stop, start) * 1e-9 + 1e-15
    while value <= stop + epsilon:
        values.append(value)
        value *= multiplier
    if not values:
        values.append(start)
    return tuple(values)


def _sweep_combination_count(sweeps: tuple[SpiceAnalysisSweep, ...]) -> int:
    count = 1
    for sweep in sweeps:
        count *= len(sweep.values)
    return count


def _sweep_combinations(sweeps: tuple[SpiceAnalysisSweep, ...]) -> tuple[tuple[tuple[SpiceAnalysisSweep, Any], ...], ...]:
    if not sweeps:
        return ((),)
    return tuple(tuple(zip(sweeps, values, strict=True)) for values in product(*(sweep.values for sweep in sweeps)))


def _sweep_overrides(sweep_points: tuple[tuple[SpiceAnalysisSweep, Any], ...]) -> dict[str, Any]:
    return {sweep.target: value for sweep, value in sweep_points if sweep.kind == "param"}


def _merged_param_overrides(param_overrides: dict[str, Any] | None, sweep_overrides: dict[str, Any]) -> dict[str, Any]:
    return {**dict(param_overrides or {}), **sweep_overrides}


def _sweep_corner(corner: Any, sweep_points: tuple[tuple[SpiceAnalysisSweep, Any], ...]) -> Any:
    temperature_values = [float(value) for sweep, value in sweep_points if sweep.kind == "temperature"]
    if not temperature_values:
        return corner
    if len(temperature_values) > 1:
        raise ValueError("multiple temperature sweep values resolved for one task")
    from monata.corner import OperatingCorner, coerce_operating_corner

    temperature = temperature_values[0]
    base = coerce_operating_corner(corner)
    if base is None:
        return OperatingCorner(_temperature_corner_name(temperature), temperature=temperature)
    return base.with_updates(
        name=f"{base.name}_{_temperature_corner_name(temperature)}",
        temperature=temperature,
    )


def _temperature_corner_name(temperature: float) -> str:
    return f"{temperature:g}C".replace("-", "m")


def _task_control_blocks(deck: SpiceDeck) -> tuple[_TaskControlBlock, ...]:
    blocks: list[_TaskControlBlock] = []
    for statement in deck.statements:
        if not isinstance(statement, ControlBlock):
            continue
        block_commands = tuple(
            command
            for line in statement.lines[1:-1]
            if (command := _control_dot_command(line.text, line=line.line, source_lines=line.source_lines)) is not None
        )
        if not any(command.name in _ANALYSIS_COMMANDS for command in block_commands):
            continue
        unsupported = tuple(command for command in block_commands if command.name not in _CONTROL_TASK_COMMANDS)
        if unsupported:
            names = ", ".join(command.name for command in unsupported)
            blocks.append(
                _TaskControlBlock(
                    statement.line,
                    (),
                    (
                        SpiceAnalysisIssue(
                            "error",
                            f".control block analysis import cannot migrate non-task commands: {names}",
                            statement.line,
                            statement.source_lines,
                        ),
                    ),
                )
            )
            continue
        blocks.append(
            _TaskControlBlock(
                statement.line,
                tuple(command for command in block_commands if command.name not in _CONTROL_IGNORED_COMMANDS),
            )
        )
    return tuple(blocks)


def _control_dot_command(text: str, *, line: int, source_lines: tuple[int, ...]) -> DotCommand | None:
    tokens = text.split()
    if not tokens:
        return None
    name = tokens[0].lower().lstrip(".")
    return DotCommand(name, tuple(tokens[1:]), {}, text, line, source_lines)


def _base_circuit(text: str, *, path: str | None) -> Circuit:
    from monata.parser.importer import parse_spice_to_circuit

    circuit = parse_spice_to_circuit(text, path=path)
    circuit.directives[:] = _base_directives(circuit.directives)
    return circuit


def _base_directives(directives: list[Directive]) -> list[Directive]:
    result: list[Directive] = []
    control_block: list[Directive] = []
    in_control = False
    for directive in directives:
        if _is_control_start(directive):
            in_control = True
            control_block = [directive]
            continue
        if in_control:
            control_block.append(directive)
            if _is_control_end(directive):
                if not _is_task_control_block(control_block):
                    result.extend(control_block)
                control_block = []
                in_control = False
            continue
        if not _is_analysis_raw_directive(directive) and not _is_step_raw_directive(directive):
            result.append(directive)
    if control_block:
        result.extend(control_block)
    return result


def _task_circuit(circuit: Circuit, *, analysis_name: str) -> Circuit:
    task_circuit = deepcopy(circuit)
    task_circuit.directives[:] = [
        directive
        for directive in task_circuit.directives
        if not _is_measure_directive(directive) or _measure_analysis(directive) == analysis_name
    ]
    return task_circuit


def _ensure_measure_directives(circuit: Circuit, measurements: tuple[SpiceAnalysisMeasurement, ...]) -> None:
    for measurement in measurements:
        if _has_measure_directive(circuit, measurement):
            continue
        circuit.measure(measurement.analysis, measurement.name, measurement.expression)


def _has_measure_directive(circuit: Circuit, measurement: SpiceAnalysisMeasurement) -> bool:
    expected = (measurement.analysis, measurement.name, measurement.expression)
    return any(
        _is_measure_directive(directive) and tuple(str(arg) for arg in directive.args[:3]) == expected
        for directive in circuit.directives
    )


def _is_analysis_raw_directive(directive: Directive) -> bool:
    if not directive.raw:
        return False
    text = directive.name.strip()
    if not text.startswith("."):
        return False
    return _canonical_analysis_name(text.split(maxsplit=1)[0].lower().lstrip(".")) in _ANALYSIS_COMMANDS


def _is_step_raw_directive(directive: Directive) -> bool:
    return _raw_directive_name(directive) in _STEP_COMMANDS


def _is_control_start(directive: Directive) -> bool:
    return _raw_directive_name(directive) == "control"


def _is_control_end(directive: Directive) -> bool:
    return _raw_directive_name(directive) == "endc"


def _is_task_control_block(directives: list[Directive]) -> bool:
    commands = tuple(
        command
        for directive in directives[1:-1]
        if (command := _control_dot_command(directive.name, line=0, source_lines=())) is not None
    )
    return any(command.name in _ANALYSIS_COMMANDS for command in commands) and all(
        command.name in _CONTROL_TASK_COMMANDS for command in commands
    )


def _raw_directive_name(directive: Directive) -> str | None:
    if not directive.raw:
        return None
    text = directive.name.strip()
    if not text.startswith("."):
        return None
    return text.split(maxsplit=1)[0].lower().lstrip(".")


def _is_measure_directive(directive: Directive) -> bool:
    return not directive.raw and directive.name == "measure" and len(directive.args) >= 3


def _measure_analysis(directive: Directive) -> str:
    return _canonical_analysis_name(str(directive.args[0]))


def _canonical_analysis_name(name: str) -> str:
    lowered = str(name).lower().lstrip(".")
    if lowered == "distortion":
        return "disto"
    if lowered == "fourier":
        return "four"
    return lowered


def _node_from_voltage(value: str, label: str) -> str:
    node, reference = _nodes_from_voltage(value, label)
    if reference == "0":
        return node
    raise ValueError(f"{label} must be a single-node voltage expression")


def _nodes_from_voltage(value: str, label: str) -> tuple[str, str]:
    text = value.strip()
    if not text.lower().startswith("v(") or not text.endswith(")"):
        raise ValueError(f"{label} must be a voltage expression")
    inner = text[2:-1]
    if not inner:
        raise ValueError(f"{label} must be a voltage expression")
    nodes = tuple(part.strip() for part in inner.split(","))
    if len(nodes) == 1 and nodes[0]:
        return nodes[0], "0"
    if len(nodes) == 2 and nodes[0] and nodes[1]:
        return nodes
    raise ValueError(f"{label} must be a one- or two-node voltage expression")


def _points(value: str, label: str) -> int:
    number = _number(value, label)
    integer = int(number)
    if integer != number:
        raise ValueError(f"{label} must be an integer")
    return integer


def _positive_points(value: str, label: str) -> int:
    points = _points(value, label)
    if points <= 0:
        raise ValueError(f"{label} must be positive")
    return points


def _number(value: str, label: str) -> float:
    try:
        return parse_spice_number(value)
    except UnitError as exc:
        raise ValueError(f"{label} must be a numeric SPICE value") from exc


def _read_text_or_path(text_or_path: str | Path) -> tuple[str, str | None]:
    if isinstance(text_or_path, Path):
        return text_or_path.read_text(), str(text_or_path)
    return text_or_path, None


__all__ = [
    "AnalysisAction",
    "AnalysisIssueSeverity",
    "AnalysisOrigin",
    "SpiceAnalysisIssue",
    "SpiceAnalysisMeasurement",
    "SpiceAnalysisPlan",
    "SpiceAnalysisStep",
    "SpiceAnalysisSweep",
    "inspect_spice_analysis",
    "spice_to_sim_tasks",
]
