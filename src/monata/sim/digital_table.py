"""Digital truth-table simulation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, cast

from monata.corner import CornerLike
from monata.models.flow import SimulationModelConfig
from monata.sim._digital_bits import bit_combinations, bits_to_text
from monata.sim.digital_claims import (
    DigitalComparator,
    DigitalOutputTolerance,
    DigitalTransientObservation,
    DigitalVerificationClaim,
)
from monata.sim.digital_circuits import _subckt_name
from monata.sim.digital_extract import (
    DigitalTruthTableResultExtractor as _DigitalTruthTableResultExtractor,
)
from monata.sim.digital_plan import DigitalTruthTablePlan as _DigitalTruthTablePlan
from monata.sim.digital_projection import PdkProjectionOwner
from monata.sim.digital_results import (
    DigitalPropagationDelayRow as _DigitalPropagationDelayRow,
    DigitalTruthTableResult as _DigitalTruthTableResult,
)
from monata.sim.digital_spec import DigitalTruthTableSpec, ExpectedLike, ExpectedTable
from monata.sim.digital_table_config import (
    SetupFn,
    SubCircuitInput,
    _DigitalTruthTableConfig,
)
from monata.sim.digital_timing import (
    DigitalPropagationDelayArc as _DigitalPropagationDelayArc,
)
from monata.sim.digital_tasks import DigitalTruthTableTaskFactory as _DigitalTruthTableTaskFactory
from monata.sim.results import SimResult
from monata.sim.task import SimArtifactOptions, SimTask


__all__ = [
    "DigitalMeasurementName",
    "DigitalTruthTableMode",
    "DigitalTruthTableSpec",
    "DigitalTruthTable",
    "ExpectedTable",
    "SetupFn",
    "SubCircuitInput",
    "build_digital_truth_table_from_spec",
    "bit_combinations",
    "bits_to_text",
    "resolve_digital_measurements",
    "resolve_digital_truth_table_mode",
]

DigitalMeasurementName = Literal["truth_table", "max_propagation_delay"]
DigitalTruthTableMode = Literal["transient"]


def build_digital_truth_table_from_spec(
    library: Any,
    spec: DigitalTruthTableSpec,
    *,
    run_config: Any,
    mode: str,
    setup: SetupFn | None = None,
    projection_library: PdkProjectionOwner | None = None,
    period: float = 1e-9,
    step: float | None = None,
    truth_table_step: float | None = None,
    cycles_per_vector: int = 2,
    slots_per_task: int | None = None,
    transition: float = 0.0,
    skew_step: float = 0.0,
    load_cap: str | float | None = None,
    metadata: Mapping[str, Any] | None = None,
    backend_options: Mapping[str, Any] | None = None,
    artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None,
) -> "DigitalTruthTable":
    """Build a DigitalTruthTable from user-declared spec data."""

    resolved_metadata = {
        "library": getattr(library, "name", None),
        "cell": spec.dut,
        "stage": spec.stage,
        "oracle": spec.oracle,
        "testbench": f"digital_{mode}_truth_table",
        **dict(spec.metadata),
        **dict(metadata or {}),
    }
    corner = getattr(run_config, "corner", None)
    corner_name = getattr(corner, "name", None)
    if corner is not None:
        resolved_metadata.setdefault("corner", corner_name)
        to_dict = getattr(corner, "to_dict", None)
        if callable(to_dict):
            resolved_metadata.setdefault("corner_payload", to_dict())
    for name in (
        "model",
        "vdd_source",
        "model_flow_identity",
    ):
        if hasattr(run_config, name):
            value = getattr(run_config, name)
            resolved_metadata.setdefault(name, dict(value) if isinstance(value, Mapping) else value)

    return DigitalTruthTable(
        _load_schematic_from_library(library, spec.dut, run_config),
        inputs=spec.inputs,
        outputs=spec.outputs,
        expected=spec.expected,
        oracle=spec.oracle,
        dependencies=tuple(
            _load_schematic_from_library(library, name, run_config)
            for name in spec.dependencies
        ),
        rails=spec.rails,
        complement_inputs=spec.complement_inputs,
        vdd=float(getattr(run_config, "vdd", 1.0)),
        threshold=getattr(run_config, "threshold", None),
        period=period,
        step=step,
        truth_table_step=truth_table_step,
        cycles_per_vector=cycles_per_vector,
        slots_per_task=slots_per_task,
        transition=transition,
        skew_step=skew_step,
        load_cap=load_cap,
        setup=setup,
        library=projection_library,
        corner=corner,
        model_config=getattr(run_config, "model_config", None),
        metadata=resolved_metadata,
        backend_options=backend_options,
        artifacts=artifacts,
    )


def _run_sim_task(task: SimTask, executor) -> SimResult:
    if executor is None:
        from monata.sim.executor import LocalExecutor
        executor = LocalExecutor(max_workers=1)
    return executor.submit(task).result()


def _run_sim_tasks(tasks: Iterable[SimTask], executor) -> list[SimResult]:
    task_list = list(tasks)
    if executor is None:
        from monata.sim.executor import LocalExecutor
        executor = LocalExecutor()
    return [future.result() for future in executor.map(task_list)]


class DigitalTruthTable:
    def __init__(
        self,
        dut: SubCircuitInput,
        inputs: Iterable[str],
        outputs: Iterable[str],
        *,
        expected: ExpectedLike | None = None,
        oracle: str | None = None,
        claim: DigitalVerificationClaim | Mapping[str, Any] | None = None,
        dependencies: Iterable[SubCircuitInput] = (),
        rails: tuple[str, str] = ("vdd", "0"),
        complement_inputs: Iterable[str] = (),
        vdd: float = 1.0,
        threshold: float | None = None,
        period: float = 1e-9,
        step: float | None = None,
        truth_table_step: float | None = None,
        cycles_per_vector: int = 2,
        slots_per_task: int | None = None,
        transition: float = 0.0,
        skew_step: float = 0.0,
        sample_fraction: float = 0.9,
        load_cap: str | float | None = None,
        tolerance: float | None = None,
        comparator: DigitalComparator | DigitalOutputTolerance | None = None,
        setup: SetupFn | None = None,
        library: PdkProjectionOwner | None = None,
        corner: CornerLike = None,
        model_config: SimulationModelConfig | None = None,
        metadata: dict | None = None,
        backend_options: Mapping[str, Any] | None = None,
        artifacts: SimArtifactOptions | Mapping[str, Any] | str | Path | None = None,
    ):
        config = _DigitalTruthTableConfig.resolve(
            dut=dut,
            inputs=inputs,
            outputs=outputs,
            expected=expected,
            oracle=oracle,
            claim=claim,
            dependencies=dependencies,
            rails=rails,
            complement_inputs=complement_inputs,
            vdd=vdd,
            threshold=threshold,
            period=period,
            step=step,
            truth_table_step=truth_table_step,
            cycles_per_vector=cycles_per_vector,
            slots_per_task=slots_per_task,
            transition=transition,
            skew_step=skew_step,
            sample_fraction=sample_fraction,
            load_cap=load_cap,
            tolerance=tolerance,
            comparator=comparator,
            setup=setup,
            library=library,
            corner=corner,
            model_config=model_config,
            metadata=metadata,
            backend_options=backend_options,
            artifacts=artifacts,
        )
        self.dut = config.dut
        self.inputs = config.inputs
        self.outputs = config.outputs
        self.expected = config.expected
        self.dependencies = config.dependencies
        self.rails = config.rails
        self.complement_inputs = config.complement_inputs
        self.vdd = config.vdd
        self.threshold = config.threshold
        self.period = config.period
        self.step = config.step
        self.truth_table_step = config.truth_table_step
        self.cycles_per_vector = config.cycles_per_vector
        self.slots_per_task = config.slots_per_task
        self.transition = config.transition
        self.skew_step = config.skew_step
        self.sample_fraction = config.sample_fraction
        self.load_cap = config.load_cap
        self.tolerance = config.tolerance
        self.setup = config.setup
        self.library = config.library
        self.corner = config.corner
        self.metadata = config.metadata
        self.backend_options = config.backend_options
        self.artifacts = config.artifacts
        self.claim = config.claim
        self.comparator = config.comparator
        self.model_config = config.model_config
        self.model_flow = config.model_flow
        self._plan = _DigitalTruthTablePlan(self)

    def combinations(self) -> tuple[tuple[int, ...], ...]:
        return self._plan.combinations()

    def transient_tasks(
        self,
        *,
        stop: float | None = None,
        uic: bool = False,
        measurements: Iterable[str] | None = None,
        cycles_per_vector: int | None = None,
        slots_per_task: int | None = None,
    ) -> list[SimTask]:
        return _DigitalTruthTableTaskFactory(self, self._plan).transient_tasks(
            stop=stop,
            uic=uic,
            measurements=resolve_digital_measurements(measurements),
            cycles_per_vector=cycles_per_vector,
            slots_per_task=slots_per_task,
        )

    def transient_observation_tasks(
        self,
        observation: DigitalTransientObservation | Mapping[str, Any] | None = None,
        *,
        stop: float | None = None,
        uic: bool | None = None,
        measurements: Iterable[str] | None = None,
        cycles_per_vector: int | None = None,
        slots_per_task: int | None = None,
    ) -> list[SimTask]:
        resolved = DigitalTransientObservation.resolve(
            observation,
            stop=stop,
            uic=uic,
            cycles_per_vector=cycles_per_vector,
            slots_per_task=slots_per_task,
        )
        return self.transient_tasks(
            stop=resolved.stop,
            uic=resolved.uic,
            measurements=measurements,
            cycles_per_vector=resolved.cycles_per_vector,
            slots_per_task=resolved.slots_per_task,
        )

    def propagation_delay_task(
        self,
        arcs: Iterable[_DigitalPropagationDelayArc] | None = None,
        *,
        chunk_index: int | None = None,
        chunk_count: int | None = None,
        step: float | None = None,
        uic: bool = True,
    ) -> SimTask:
        return _DigitalTruthTableTaskFactory(self, self._plan).propagation_delay_task(
            arcs,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            step=step,
            uic=uic,
        )

    def propagation_delay_tasks(
        self,
        *,
        chunk_size: int | None = None,
    ) -> list[SimTask]:
        chunks = self.single_bit_arc_sequence_chunks(chunk_size=chunk_size)
        return [
            self.propagation_delay_task(
                chunk,
                chunk_index=index,
                chunk_count=len(chunks),
                uic=True,
            )
            for index, chunk in enumerate(chunks)
        ]

    def run(
        self,
        mode: str = "transient",
        executor=None,
        *,
        measurements: Iterable[str] | None = None,
    ) -> _DigitalTruthTableResult:
        resolved_mode = resolve_digital_truth_table_mode(mode)
        if resolved_mode == "transient":
            return self.run_transient(executor, measurements=measurements)
        raise ValueError(f"unsupported digital truth-table mode: {mode}")

    def run_transient(
        self,
        executor=None,
        *,
        stop: float | None = None,
        uic: bool = False,
        measurements: Iterable[str] | None = None,
    ) -> _DigitalTruthTableResult:
        return self.run_transient_observation(executor, stop=stop, uic=uic, measurements=measurements)

    def run_transient_observation(
        self,
        executor=None,
        observation: DigitalTransientObservation | Mapping[str, Any] | None = None,
        *,
        stop: float | None = None,
        uic: bool | None = None,
        measurements: Iterable[str] | None = None,
    ) -> _DigitalTruthTableResult:
        selected_measurements = resolve_digital_measurements(measurements)
        resolved = DigitalTransientObservation.resolve(observation, stop=stop, uic=uic)
        tasks = self.transient_observation_tasks(
            resolved,
            measurements=selected_measurements,
        )
        results = _run_sim_tasks(tasks, executor)
        return self.extract_transient_results(results, measurements=selected_measurements)

    def extract_transient_results(
        self,
        sim_results: Iterable[SimResult],
        *,
        measurements: Iterable[str] | None = None,
    ) -> _DigitalTruthTableResult:
        return _DigitalTruthTableResultExtractor(self, self._plan).transient(
            sim_results,
            measurements=resolve_digital_measurements(measurements),
        )

    def extract_propagation_delays(
        self,
        sim_result: SimResult,
        arcs: Iterable[_DigitalPropagationDelayArc] | None = None,
    ) -> tuple[_DigitalPropagationDelayRow, ...]:
        return _DigitalTruthTableResultExtractor(self, self._plan).propagation_delay(sim_result, arcs)

    def run_propagation_delay(
        self,
        executor=None,
    ) -> tuple[tuple[_DigitalPropagationDelayRow, ...], SimResult]:
        task = self.propagation_delay_task()
        sim_result = _run_sim_task(task, executor)
        return self.extract_propagation_delays(sim_result), sim_result

    def run_required_propagation_delay(
        self,
        executor=None,
    ) -> tuple[tuple[_DigitalPropagationDelayRow, ...], tuple[SimResult, ...]]:
        return self._run_propagation_delay_chunks(
            self.single_bit_arc_sequence_chunks(),
            executor=executor,
        )

    @property
    def dut_name(self) -> str:
        return _subckt_name(self.dut)

    def propagation_delay_arcs(self) -> tuple[_DigitalPropagationDelayArc, ...]:
        return self._plan.propagation_delay_arcs()

    def single_bit_arcs(self) -> tuple[_DigitalPropagationDelayArc, ...]:
        return self._plan.single_bit_arcs()

    @property
    def single_bit_arc_count(self) -> int:
        return self._plan.single_bit_arc_count

    @property
    def measurable_single_bit_arc_count(self) -> int:
        return self._plan.measurable_single_bit_arc_count()

    def single_bit_arc_sequence_chunks(
        self,
        *,
        chunk_size: int | None = None,
        require_expected: bool = True,
        require_transition: bool = True,
    ) -> tuple[tuple[_DigitalPropagationDelayArc, ...], ...]:
        return self._plan.single_bit_arc_sequence_chunks(
            chunk_size=chunk_size,
            require_expected=require_expected,
            require_transition=require_transition,
        )

    def propagation_delay_arc_chunks(
        self,
        *,
        chunk_size: int | None = None,
    ) -> tuple[tuple[_DigitalPropagationDelayArc, ...], ...]:
        return self._plan.propagation_delay_arc_chunks(chunk_size=chunk_size)

    def expected_for(self, inputs: tuple[int, ...]) -> tuple[int, ...] | None:
        if self.expected is None:
            return None
        if isinstance(self.expected, ExpectedTable):
            return self.expected(inputs)
        return self.expected(inputs)

    def _with_required_propagation_delay(
        self,
        result: _DigitalTruthTableResult,
        *,
        executor=None,
    ) -> _DigitalTruthTableResult:
        delays, sim_results = self.run_required_propagation_delay(executor=executor)
        return result.with_propagation_delays(delays, sim_results=sim_results)

    def _run_propagation_delay_chunks(
        self,
        chunks: Iterable[tuple[_DigitalPropagationDelayArc, ...]],
        *,
        executor=None,
    ) -> tuple[tuple[_DigitalPropagationDelayRow, ...], tuple[SimResult, ...]]:
        chunk_list = tuple(chunks)
        if not chunk_list:
            raise RuntimeError(f"{self.dut_name} propagation delay requires at least one chunk")
        tasks = [
            self.propagation_delay_task(
                chunk,
                chunk_index=index,
                chunk_count=len(chunk_list),
            )
            for index, chunk in enumerate(chunk_list)
        ]
        sim_results = _run_sim_tasks(tasks, executor)

        rows: list[_DigitalPropagationDelayRow] = []
        results: list[SimResult] = []
        for index, (chunk, sim_result) in enumerate(zip(chunk_list, sim_results)):
            try:
                rows.extend(self.extract_propagation_delays(sim_result, arcs=chunk))
                results.append(sim_result)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{self.dut_name} propagation delay chunk {index + 1}/{len(chunk_list)} failed: {exc}"
                ) from exc
        return tuple(rows), tuple(results)

def resolve_digital_measurements(
    measurements: Iterable[str] | None = None,
    *,
    default: Iterable[str] = ("truth_table",),
) -> tuple[DigitalMeasurementName, ...]:
    selected = tuple(dict.fromkeys(str(name) for name in (default if measurements is None else measurements)))
    if not selected:
        raise ValueError("digital measurement list must not be empty")
    supported = {"truth_table", "max_propagation_delay"}
    unknown = sorted(set(selected) - supported)
    if unknown:
        raise ValueError("unsupported digital measurements: " + ", ".join(unknown))
    return cast(tuple[DigitalMeasurementName, ...], selected)


def resolve_digital_truth_table_mode(mode: str) -> DigitalTruthTableMode:
    if mode == "transient":
        return "transient"
    raise ValueError(f"unsupported digital truth-table mode: {mode}")


def _load_schematic_from_library(library: Any, cell_name: str, run_config: Any) -> SubCircuitInput:
    del run_config
    from monata.views.declarative import schematic_view_to_subcircuit

    return schematic_view_to_subcircuit(
        library[cell_name]["schematic"],
        allow_trusted_python=False,
        reason="digital_truth_table spec",
    )
