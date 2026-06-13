"""Registry for view type factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from monata._paths import validate_path_segment
from monata.errors import ViewNotFoundError
ViewConfig: TypeAlias = Mapping[str, object]
MutableViewConfig: TypeAlias = dict[str, object]
ViewConfigOptions: TypeAlias = Mapping[str, object]
ViewFactory = Callable[[Any, ViewConfig], Any]
ViewConfigFactory = Callable[[str, ViewConfigOptions, "ViewSchema"], ViewConfig]
ViewGenerator = Callable[..., Path]


def _view_type_key(view_type: object) -> str:
    return validate_path_segment(view_type, "view type")


@dataclass(frozen=True)
class ViewSchema:
    """Declarative owner metadata for a view type."""

    factory: ViewFactory
    default_entry: str | None = None
    generated: bool = False
    config_factory: ViewConfigFactory | None = None
    generator: ViewGenerator | None = None

    def create_config(self, view_type: str, options: ViewConfigOptions) -> MutableViewConfig:
        if self.config_factory is not None:
            return dict(self.config_factory(view_type, options, self))

        entry = options.get("entry", self.default_entry or f"{view_type}.py")
        config: MutableViewConfig = {"entry": entry}
        if self.generated:
            config["generated"] = bool(options.get("generated", True))
        return config


class ViewRegistry:
    """Registry object for cell view schemas, factories, and generators."""

    def __init__(self) -> None:
        self._schemas: dict[str, ViewSchema] = {}

    def register(
        self,
        view_type: str,
        factory: ViewFactory,
        *,
        replace: bool = False,
        default_entry: str | None = None,
        generated: bool = False,
        config_factory: ViewConfigFactory | None = None,
        generator: ViewGenerator | None = None,
    ) -> None:
        key = _view_type_key(view_type)
        if key in self._schemas and not replace:
            raise ValueError(f"view type already registered: {key}")
        self._schemas[key] = ViewSchema(
            factory=factory,
            default_entry=default_entry,
            generated=generated,
            config_factory=config_factory,
            generator=generator,
        )

    def unregister(self, view_type: str) -> None:
        self._schemas.pop(_view_type_key(view_type), None)

    def get_schema(self, view_type: str) -> ViewSchema | None:
        return self._schemas.get(_view_type_key(view_type))

    def get_factory(self, view_type: str) -> ViewFactory | None:
        schema = self.get_schema(view_type)
        return schema.factory if schema is not None else None

    def create_config(self, view_type: str, options: ViewConfigOptions) -> MutableViewConfig:
        schema = self.get_schema(view_type)
        if schema is None:
            raise ValueError(f"unknown view type: {view_type}")
        return schema.create_config(view_type, options)

    def create(self, cell: Any, view_type: str, config: ViewConfig) -> Any:
        schema = self.get_schema(view_type)
        if schema is None:
            raise ViewNotFoundError(view_type, cell.name)
        return schema.factory(cell, config)

    def generate(self, cell: Any, view_type: str, **kwargs: Any) -> Path:
        schema = self.get_schema(view_type)
        if schema is None or schema.generator is None:
            raise ViewNotFoundError(view_type, cell.name)
        return schema.generator(cell, **kwargs)

    def list_view_types(self) -> list[str]:
        return sorted(self._schemas)


_DEFAULT_VIEW_REGISTRY = ViewRegistry()


def default_view_registry() -> ViewRegistry:
    return _DEFAULT_VIEW_REGISTRY


def register_view_type(
    view_type: str,
    factory: ViewFactory,
    *,
    replace: bool = False,
    default_entry: str | None = None,
    generated: bool = False,
    config_factory: ViewConfigFactory | None = None,
    generator: ViewGenerator | None = None,
) -> None:
    default_view_registry().register(
        view_type,
        factory,
        replace=replace,
        default_entry=default_entry,
        generated=generated,
        config_factory=config_factory,
        generator=generator,
    )


def unregister_view_type(view_type: str) -> None:
    default_view_registry().unregister(view_type)


def get_view_factory(view_type: str) -> ViewFactory | None:
    return default_view_registry().get_factory(view_type)


def get_view_schema(view_type: str) -> ViewSchema | None:
    return default_view_registry().get_schema(view_type)


def create_registered_view_config(view_type: str, **kwargs: object) -> MutableViewConfig:
    return default_view_registry().create_config(view_type, kwargs)


def create_registered_view(cell: Any, view_type: str, config: ViewConfig) -> Any:
    return default_view_registry().create(cell, view_type, config)


def generate_registered_view(cell: Any, view_type: str, **kwargs: Any) -> Path:
    return default_view_registry().generate(cell, view_type, **kwargs)


def _register_defaults() -> None:
    register_view_type(
        "schematic",
        _schematic_view,
        replace=True,
        default_entry="schematic.py",
        config_factory=_schematic_config,
    )
    register_view_type(
        "testbench",
        _testbench_view,
        replace=True,
        default_entry="testbench.py",
        config_factory=_testbench_config,
    )
    register_view_type(
        "netlist",
        _netlist_view,
        replace=True,
        default_entry="netlist.cir",
        generated=True,
        generator=_generate_netlist,
    )
    register_view_type(
        "symbol",
        _symbol_view,
        replace=True,
        default_entry="symbol.toml",
        generated=True,
        generator=_generate_symbol,
    )
    register_view_type(
        "simulation",
        _simulation_view,
        replace=True,
        default_entry="simulation.py",
        config_factory=_simulation_config,
    )
    register_view_type(
        "digital_truth_table",
        _digital_truth_table_view,
        replace=True,
        default_entry="digital_truth_table.py",
        config_factory=_digital_truth_table_config,
    )


def _schematic_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.schematic import SchematicView

    return SchematicView(cell=cell, entry=str(cfg["entry"]), cls_name=str(cfg["class"]))


def _testbench_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.testbench import TestbenchView

    return TestbenchView(cell=cell, entry=str(cfg["entry"]), function_name=str(cfg["function"]))


def _netlist_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.netlist import NetlistView

    return NetlistView(cell=cell, entry=str(cfg["entry"]))


def _symbol_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.symbol import SymbolView

    return SymbolView(cell=cell, entry=str(cfg["entry"]))


def _simulation_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.simulation import SimulationView

    return SimulationView(
        cell=cell,
        entry=str(cfg["entry"]),
        function_name=str(cfg.get("function", "main")),
        backend=str(cfg["backend"]) if cfg.get("backend") is not None else None,
        max_workers=_optional_int(cfg.get("max_workers")),
    )


def _digital_truth_table_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.digital_truth_table import DigitalTruthTableView

    return DigitalTruthTableView(
        cell=cell,
        entry=str(cfg["entry"]),
        function_name=str(cfg.get("function", "build_truth_table")),
        mode=str(cfg.get("mode", "transient")),
        simulation_view=str(cfg.get("simulation_view", "simulation")),
        config=cfg,
    )


def _schematic_config(view_type: str, options: ViewConfigOptions, schema: ViewSchema) -> MutableViewConfig:
    if "cls_name" not in options:
        raise KeyError("cls_name")
    return {
        "entry": options.get("entry", schema.default_entry or f"{view_type}.py"),
        "class": options["cls_name"],
    }


def _testbench_config(view_type: str, options: ViewConfigOptions, schema: ViewSchema) -> MutableViewConfig:
    return {
        "entry": options.get("entry", schema.default_entry or f"{view_type}.py"),
        "function": options.get("function_name", "main"),
    }


def _simulation_config(view_type: str, options: ViewConfigOptions, schema: ViewSchema) -> MutableViewConfig:
    config: MutableViewConfig = {
        "entry": options.get("entry", schema.default_entry or f"{view_type}.py"),
        "function": options.get("function_name", "main"),
    }
    for key in ("backend", "max_workers"):
        if key in options:
            config[key] = options[key]
    return config


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, str)):
        return int(value)
    raise TypeError(f"expected int-compatible value, got {type(value).__name__}")


def _digital_truth_table_config(
    view_type: str,
    options: ViewConfigOptions,
    schema: ViewSchema,
) -> MutableViewConfig:
    return {
        "entry": options.get("entry", schema.default_entry or f"{view_type}.py"),
        "function": options.get("function_name", "build_truth_table"),
        "mode": options.get("mode", "transient"),
        "simulation_view": options.get("simulation_view", "simulation"),
    }


def _generate_netlist(cell: Any, **kwargs: Any) -> Path:
    from monata.generation.netlist import generate_netlist

    return generate_netlist(cell, **kwargs)


def _generate_symbol(cell: Any, **kwargs: Any) -> Path:
    from monata.generation.symbol import generate_symbol

    return generate_symbol(cell, **kwargs)


_register_defaults()

__all__ = [
    "MutableViewConfig",
    "ViewConfig",
    "ViewConfigFactory",
    "ViewConfigOptions",
    "ViewFactory",
    "ViewGenerator",
    "ViewRegistry",
    "ViewSchema",
    "create_registered_view_config",
    "create_registered_view",
    "default_view_registry",
    "generate_registered_view",
    "get_view_factory",
    "get_view_schema",
    "register_view_type",
    "unregister_view_type",
]
