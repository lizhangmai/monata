"""Stateful simulation session builder over backend-neutral SimTasks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from monata.netlist import Circuit, Directive
from monata.sim.analysis_spec import (
    ACSpec,
    AnalysisSpec,
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
from monata.sim.task import (
    DEFAULT_SIMULATOR,
    DEFAULT_SIM_TIMEOUT_SECONDS,
    SimArtifactOptions,
    SimTask,
    normalize_output_names,
)

_UNSET = object()
TAnalysisSpec = TypeVar("TAnalysisSpec", bound=AnalysisSpec)


class SimulationSession:
    """Convenience facade for circuit directives plus task construction."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        simulator: str = DEFAULT_SIMULATOR,
        output_names: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        backend_options: Mapping[str, Any] | None = None,
        artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None,
        snapshot_tasks: bool = True,
        timeout: float | int | None = DEFAULT_SIM_TIMEOUT_SECONDS,
        temperature: Any = _UNSET,
        nominal_temperature: Any = _UNSET,
    ) -> None:
        self.circuit = circuit
        self.simulator = str(simulator)
        self.output_names = normalize_output_names(output_names)
        self.metadata = dict(metadata or {})
        self.backend_options = dict(backend_options or {})
        self.artifacts = SimArtifactOptions.coerce(artifacts)
        self.snapshot_tasks = bool(snapshot_tasks)
        self.timeout = timeout
        self._queued_tasks: list[SimTask] = []
        initial_options = {}
        if temperature is not _UNSET:
            initial_options["temp"] = temperature
        if nominal_temperature is not _UNSET:
            initial_options["tnom"] = nominal_temperature
        if initial_options:
            self.circuit.options(**initial_options)

    def options(self, *flags: str, **options: Any) -> SimulationSession:
        self.circuit.options(*flags, **options)
        return self

    def temperature(self, value: Any, *, nominal: Any | None = None) -> SimulationSession:
        options = {"temp": value}
        if nominal is not None:
            options["tnom"] = nominal
        self.circuit.options(**options)
        return self

    def initial_condition(self, **nodes: Any) -> SimulationSession:
        self.circuit.ic(**nodes)
        return self

    def node_set(self, **nodes: Any) -> SimulationSession:
        self.circuit.nodeset(**nodes)
        return self

    def save(self, *vectors: str) -> SimulationSession:
        self.circuit.save(*vectors)
        return self

    def save_internal_parameters(self, *vectors: str) -> SimulationSession:
        self.circuit.save(*vectors, "all")
        return self

    def save_currents(self, enabled: bool = True) -> SimulationSession:
        if enabled:
            self.circuit.options(savecurrents=True)
        else:
            _remove_option(self.circuit, "savecurrents")
        return self

    def measure(self, analysis: str, name: str, *expressions: str) -> SimulationSession:
        self.circuit.measure(analysis, name, *expressions)
        return self

    def raw_directive(self, line: str) -> SimulationSession:
        self.circuit.raw_directive(line)
        return self

    def raw_block(self, lines: str | Iterable[Any]) -> SimulationSession:
        self.circuit.raw_block(lines)
        return self

    def str_netlist(self) -> str:
        return self.circuit.to_spice()

    def str_options(self) -> str:
        """Return this session's rendered .options directives."""

        return "".join(
            f"{directive.to_spice()}\n"
            for directive in self.circuit.directives
            if not directive.raw and directive.name == "options"
        )

    def to_spice(self) -> str:
        return self.str_netlist()

    def __str__(self) -> str:
        return self.to_spice()

    def task(
        self,
        analysis_spec: TAnalysisSpec,
        *,
        output_names: Iterable[str] | None = None,
        corner: Any = None,
        param_overrides: dict[str, Any] | None = None,
        osdi_paths: Iterable[str | Path] = (),
        metadata: dict[str, Any] | None = None,
        backend_options: Mapping[str, Any] | None = None,
        artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None,
        snapshot: bool | None = None,
        simulator: str | None = None,
        timeout: float | int | None = None,
    ) -> SimTask[TAnalysisSpec]:
        snapshot_task = self.snapshot_tasks if snapshot is None else bool(snapshot)
        return SimTask(
            circuit=_task_circuit(self.circuit, snapshot=snapshot_task),
            analysis_spec=analysis_spec,
            simulator=simulator or self.simulator,
            corner=corner,
            param_overrides=param_overrides,
            output_names=tuple(output_names) if output_names is not None else self.output_names,
            osdi_paths=tuple(osdi_paths),
            metadata={**self.metadata, **dict(metadata or {})},
            backend_options={**self.backend_options, **dict(backend_options or {})},
            artifacts=self.artifacts if artifacts is None else artifacts,
            timeout=self.timeout if timeout is None else timeout,
        )

    def run(self, analysis_spec: Any, executor: Any = None, **task_kwargs: Any) -> Any:
        from monata.sim.executor import LocalExecutor

        runner = executor or LocalExecutor()
        return runner.submit(self.task(analysis_spec, **task_kwargs)).result()

    def queue(self, analysis_spec: SimTask[Any] | AnalysisSpec, **task_kwargs: Any) -> SimTask[Any]:
        """Append an analysis task to this session's explicit task queue."""

        task = analysis_spec if isinstance(analysis_spec, SimTask) else self.task(analysis_spec, **task_kwargs)
        if task_kwargs and isinstance(analysis_spec, SimTask):
            raise ValueError("queued SimTask instances cannot be combined with task keyword arguments")
        self._queued_tasks.append(task)
        return task

    @property
    def queued_tasks(self) -> tuple[SimTask[Any], ...]:
        """Return queued analysis tasks in insertion order."""

        return tuple(self._queued_tasks)

    def iter_queued_tasks(self) -> Iterator[SimTask[Any]]:
        """Iterate over a stable snapshot of queued analysis tasks."""

        return iter(self.queued_tasks)

    def clear_queue(self) -> SimulationSession:
        """Remove all queued analysis tasks."""

        self._queued_tasks.clear()
        return self

    def run_queue(self, executor: Any = None, *, clear: bool = False) -> list[Any]:
        """Execute queued tasks through an executor and return results in queue order."""

        tasks = list(self.queued_tasks)
        if not tasks:
            if clear:
                self.clear_queue()
            return []

        from monata.sim.executor import LocalExecutor

        runner = executor or LocalExecutor()
        futures = runner.map(tasks)
        results = [future.result() for future in futures]
        if clear:
            self.clear_queue()
        return results

    def op(self, **kwargs: Any) -> SimTask[OPSpec]:
        return self.task(OPSpec(), **kwargs)

    def tran(
        self,
        stop: float,
        step: float | None = None,
        *,
        start: float = 0,
        max_step: float | None = None,
        uic: bool = False,
        **kwargs: Any,
    ) -> SimTask[TranSpec]:
        return self.task(TranSpec(stop=stop, step=step, start=start, max_step=max_step, uic=uic), **kwargs)

    def ac(
        self,
        *,
        start: float | None = None,
        stop: float | None = None,
        points: int | None = None,
        variation: str = "dec",
        **kwargs: Any,
    ) -> SimTask[ACSpec]:
        start, stop, points = _resolve_frequency_sweep("ac", start, stop, points)
        return self.task(ACSpec(start=start, stop=stop, points=points, variation=variation), **kwargs)

    def dc(
        self,
        source: str,
        start: float,
        stop: float,
        step: float,
        *,
        secondary: DCSweep | None = None,
        **kwargs: Any,
    ) -> SimTask[DCSpec]:
        return self.task(DCSpec(source=source, start=start, stop=stop, step=step, secondary=secondary), **kwargs)

    def noise(
        self,
        output_node: str,
        *,
        input_source: str | None = None,
        start: float | None = None,
        stop: float | None = None,
        points: int | None = None,
        reference_node: str = "0",
        variation: str = "dec",
        points_per_summary: int | None = None,
        **kwargs: Any,
    ) -> SimTask[NoiseSpec]:
        if input_source is None:
            raise TypeError("noise requires input_source")
        start, stop, points = _resolve_frequency_sweep(
            "noise",
            start,
            stop,
            points,
        )
        return self.task(
            NoiseSpec(
                output_node,
                input_source,
                start,
                stop,
                points,
                reference_node=reference_node,
                variation=variation,
                points_per_summary=points_per_summary,
            ),
            **kwargs,
        )

    def sensitivity(self, output: str, **kwargs: Any) -> SimTask[SensitivitySpec]:
        return self.task(SensitivitySpec(output), **kwargs)

    def ac_sensitivity(
        self,
        output: str,
        *,
        start: float | None = None,
        stop: float | None = None,
        points: int | None = None,
        variation: str = "dec",
        **kwargs: Any,
    ) -> SimTask[SensitivitySpec]:
        start, stop, points = _resolve_frequency_sweep("ac sensitivity", start, stop, points)
        return self.task(
            SensitivitySpec(output, start=start, stop=stop, points=points, variation=variation),
            **kwargs,
        )

    def pole_zero(
        self,
        input_pos: str,
        input_neg: str,
        output_pos: str,
        output_neg: str,
        *,
        transfer: str = "vol",
        mode: str = "pz",
        **kwargs: Any,
    ) -> SimTask[PoleZeroSpec]:
        return self.task(PoleZeroSpec(input_pos, input_neg, output_pos, output_neg, transfer=transfer, mode=mode), **kwargs)

    def distortion(
        self,
        *,
        start: float | None = None,
        stop: float | None = None,
        points: int | None = None,
        variation: str = "dec",
        f2overf1: float | None = None,
        **kwargs: Any,
    ) -> SimTask[DistortionSpec]:
        start, stop, points = _resolve_frequency_sweep("distortion", start, stop, points)
        return self.task(DistortionSpec(start, stop, points, variation=variation, f2overf1=f2overf1), **kwargs)

    def transfer_function(self, output: str, input_source: str, **kwargs: Any) -> SimTask[TransferFunctionSpec]:
        return self.task(TransferFunctionSpec(output, input_source), **kwargs)

    def fourier(
        self,
        frequency: float,
        output: str,
        stop: float,
        *,
        step: float | None = None,
        start: float = 0,
        **kwargs: Any,
    ) -> SimTask[FourierSpec]:
        return self.task(FourierSpec(frequency, output, stop, step=step, start=start), **kwargs)


def _remove_option(circuit: Circuit, option_name: str) -> None:
    key = option_name.lower()
    for index in range(len(circuit.directives) - 1, -1, -1):
        directive = circuit.directives[index]
        if directive.raw or directive.name != "options":
            continue
        params = OrderedDict((name, value) for name, value in directive.params.items() if name.lower() != key)
        if len(params) == len(directive.params):
            continue
        if params:
            circuit.directives[index] = Directive("options", directive.args, params)
        else:
            del circuit.directives[index]


def _task_circuit(circuit: Circuit, *, snapshot: bool) -> Circuit:
    if not snapshot:
        return circuit
    return circuit.clone()


def _resolve_frequency_sweep(
    label: str,
    start: float | None,
    stop: float | None,
    points: int | None,
) -> tuple[float, float, int]:
    if start is None or stop is None or points is None:
        raise TypeError(f"{label} requires start, stop, and points")
    return start, stop, points


__all__ = ["SimulationSession"]
