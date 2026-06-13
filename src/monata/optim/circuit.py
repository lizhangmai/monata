"""CircuitOptimizer — high-level API for circuit parameter optimization."""

from __future__ import annotations

import re
from typing import Any, Callable, cast

from monata._json import json_safe as _json_safe
from monata.optim.base import (
    Constraint,
    DesignVariable,
    Objective,
    OptimResult,
    Optimizer,
    validate_optimization_problem,
)
from monata.sim.task import SimTask


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPTIMIZER_METADATA_KEY = "monata_optimizer"


class CircuitOptimizer:
    def __init__(
        self,
        circuit,
        analysis,
        executor=None,
        output_names=None,
        osdi_paths=None,
        metadata=None,
    ):
        self.circuit = circuit
        self.analysis = analysis
        self._executor = executor
        self.output_names = output_names
        self.osdi_paths = osdi_paths
        self.metadata: dict[str, Any] = cast(dict[str, Any], _json_safe(metadata or {}, strict=True))
        self._variables: list[DesignVariable] = []
        self._objectives: list[Objective] = []
        self._constraints: list[Constraint] = []
        self._metric_fns: dict[str, Callable] = {}

    def variable(self, element: str, param: str, min: float, max: float, scale: str = "linear"):
        name = f"{element}.{param}"
        self._variables.append(DesignVariable(name, min=min, max=max, scale=scale))

    def parameter(self, name: str, min: float, max: float, scale: str = "linear"):
        if not _IDENTIFIER_RE.match(name):
            raise ValueError(f"global parameter name must be a simple identifier: {name}")
        self._variables.append(DesignVariable(name, min=min, max=max, scale=scale))

    def maximize(self, name: str, metric_fn: Callable):
        _validate_metric_fn(metric_fn, name)
        self._objectives.append(Objective(name, direction="maximize"))
        self._metric_fns[name] = metric_fn

    def minimize(self, name: str, metric_fn: Callable):
        _validate_metric_fn(metric_fn, name)
        self._objectives.append(Objective(name, direction="minimize"))
        self._metric_fns[name] = metric_fn

    def constraint(
        self,
        name: str,
        metric_fn: Callable,
        min: float | None = None,
        max: float | None = None,
    ):
        _validate_metric_fn(metric_fn, name)
        self._constraints.append(Constraint(name, min=min, max=max))
        self._metric_fns[name] = metric_fn

    def run(self, optimizer: Optimizer | None = None, n_iter: int = 200) -> OptimResult:
        variables, objectives, constraints = validate_optimization_problem(
            self._variables,
            self._objectives,
            self._constraints,
            n_iter,
        )
        self._validate_native_parameters()

        if optimizer is None:
            from monata.optim.nsga2 import NSGA2Optimizer
            optimizer = NSGA2Optimizer()

        executor = self._executor
        if executor is None:
            from monata.sim.executor import LocalExecutor
            executor = LocalExecutor()

        evaluation_index = 0

        def eval_fn(params: dict) -> dict:
            nonlocal evaluation_index
            safe_params = _json_safe(params, strict=True)
            metadata = dict(self.metadata)
            metadata[_optimizer_metadata_key(metadata)] = {
                "evaluation_index": evaluation_index,
                "params": safe_params,
            }
            evaluation_index += 1
            task = SimTask(
                circuit=self.circuit,
                analysis_spec=self.analysis,
                param_overrides=params,
                output_names=self.output_names,
                osdi_paths=self.osdi_paths,
                metadata=metadata,
            )
            future = executor.submit(task)
            sim_result = future.result()
            if sim_result.status != "ok" and sim_result.metadata.get("reason") == "unsupported_param_overrides":
                message = sim_result.error_message or "optimizer mutation target failed"
                raise ValueError(message)

            metrics = {}
            for name, fn in self._metric_fns.items():
                if sim_result.status == "ok":
                    metrics[name] = fn(sim_result)
                else:
                    metrics[name] = float("inf") if any(
                        o.direction == "minimize" for o in self._objectives if o.name == name
                    ) else float("-inf")
            return metrics

        return optimizer.optimize(
            variables,
            objectives,
            constraints,
            eval_fn,
            n_iter,
        )

    def _validate_native_parameters(self) -> None:
        unsupported = [
            v.name for v in self._variables
            if not (_IDENTIFIER_RE.match(v.name) or _is_structured_target(v.name))
        ]
        if unsupported:
            raise ValueError(
                "optimizer variable names must be simple globals or element.param targets: "
                f"{', '.join(unsupported)}"
            )


def _optimizer_metadata_key(metadata: dict) -> str:
    if _OPTIMIZER_METADATA_KEY not in metadata:
        return _OPTIMIZER_METADATA_KEY
    index = 2
    while f"{_OPTIMIZER_METADATA_KEY}_{index}" in metadata:
        index += 1
    return f"{_OPTIMIZER_METADATA_KEY}_{index}"


def _validate_metric_fn(metric_fn: Callable, name: str) -> None:
    if not callable(metric_fn):
        raise TypeError(f"metric function for {name} must be callable")


def _is_structured_target(value: str) -> bool:
    if "." not in value:
        return False
    element, param = value.split(".", 1)
    return bool(element and param)
