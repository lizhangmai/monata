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
_REMOVED_SCHEMATIC_PY_VIEW = "schematic" + "_py"
_REMOVED_PYTHON_SCHEMATIC_FORMAT = "python-" + "schematic"


def _view_type_key(view_type: object) -> str:
    return validate_path_segment(view_type, "view type")


def _format_key(view_format: object) -> str:
    if not isinstance(view_format, str) or not view_format:
        raise ValueError("view format must be a non-empty string")
    if any(char.isspace() for char in view_format) or any(char in view_format for char in "\r\n"):
        raise ValueError("view format must not contain whitespace or newlines")
    return view_format


@dataclass(frozen=True)
class ViewSchema:
    """Declarative owner metadata for a view type."""

    factory: ViewFactory
    default_entry: str | None = None
    view_format: str | None = None
    generated: bool = False
    trusted: bool = False
    schema_version: int | None = None
    config_factory: ViewConfigFactory | None = None
    generator: ViewGenerator | None = None

    def create_config(self, view_type: str, options: ViewConfigOptions) -> MutableViewConfig:
        if self.config_factory is not None:
            return dict(self.config_factory(view_type, options, self))

        entry = options.get("entry", self.default_entry or f"{view_type}.py")
        config: MutableViewConfig = {"entry": entry}
        if self.view_format is not None:
            config["format"] = options.get("format", self.view_format)
        if self.generated:
            config["generated"] = bool(options.get("generated", True))
        if self.trusted:
            config["trusted"] = bool(options.get("trusted", True))
        elif "trusted" in options:
            config["trusted"] = bool(options["trusted"])
        if self.schema_version is not None:
            config["schema_version"] = options.get("schema_version", self.schema_version)
        return config


class ViewRegistry:
    """Registry object for cell view schemas, factories, and generators."""

    def __init__(self) -> None:
        self._schemas: dict[str, ViewSchema] = {}
        self._formats: dict[str, ViewSchema] = {}

    def register(
        self,
        view_type: str,
        factory: ViewFactory,
        *,
        replace: bool = False,
        default_entry: str | None = None,
        view_format: str | None = None,
        generated: bool = False,
        trusted: bool = False,
        schema_version: int | None = None,
        config_factory: ViewConfigFactory | None = None,
        generator: ViewGenerator | None = None,
    ) -> None:
        key = _view_type_key(view_type)
        if key in self._schemas and not replace:
            raise ValueError(f"view type already registered: {key}")
        fmt = _format_key(view_format) if view_format is not None else None
        if fmt is not None and fmt in self._formats and not replace:
            raise ValueError(f"view format already registered: {fmt}")
        if replace and key in self._schemas:
            old = self._schemas[key]
            self._formats = {name: schema for name, schema in self._formats.items() if schema is not old}
        schema = ViewSchema(
            factory=factory,
            default_entry=default_entry,
            view_format=fmt,
            generated=generated,
            trusted=trusted,
            schema_version=schema_version,
            config_factory=config_factory,
            generator=generator,
        )
        self._schemas[key] = schema
        if fmt is not None:
            self._formats[fmt] = schema

    def unregister(self, view_type: str) -> None:
        schema = self._schemas.pop(_view_type_key(view_type), None)
        if schema is not None:
            self._formats = {name: item for name, item in self._formats.items() if item is not schema}

    def get_schema(self, view_type: str) -> ViewSchema | None:
        return self._schemas.get(_view_type_key(view_type))

    def get_schema_for_format(self, view_format: str) -> ViewSchema | None:
        return self._formats.get(_format_key(view_format))

    def get_factory(self, view_type: str) -> ViewFactory | None:
        schema = self.get_schema(view_type)
        return schema.factory if schema is not None else None

    def create_config(self, view_type: str, options: ViewConfigOptions) -> MutableViewConfig:
        _reject_removed_schematic_view(view_type, options.get("format"))
        schema = None
        requested_format = options.get("format")
        if requested_format is not None:
            schema = self.get_schema_for_format(str(requested_format))
            if schema is None:
                raise ValueError(f"unknown view format: {requested_format}")
        if schema is None:
            schema = self.get_schema(view_type)
        if schema is None:
            raise ValueError(f"unknown view type: {view_type}")
        return schema.create_config(view_type, options)

    def create(self, cell: Any, view_type: str, config: ViewConfig) -> Any:
        schema = self._schema_for_view_config(view_type, config)
        if schema is None:
            raise ViewNotFoundError(view_type, cell.name)
        return schema.factory(cell, config)

    def _schema_for_view_config(self, view_type: str, config: ViewConfig) -> ViewSchema | None:
        if "format" in config:
            view_format = _format_key(config["format"])
            _reject_removed_schematic_view(view_type, view_format)
            _validate_trusted_format(view_format, config)
            schema = self.get_schema_for_format(view_format)
            if schema is None:
                raise ValueError(f"unknown view format: {view_format}")
            return schema
        _reject_removed_view_metadata(view_type, config)
        return self.get_schema(view_type)

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
    view_format: str | None = None,
    generated: bool = False,
    trusted: bool = False,
    schema_version: int | None = None,
    config_factory: ViewConfigFactory | None = None,
    generator: ViewGenerator | None = None,
) -> None:
    default_view_registry().register(
        view_type,
        factory,
        replace=replace,
        default_entry=default_entry,
        view_format=view_format,
        generated=generated,
        trusted=trusted,
        schema_version=schema_version,
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
        _schematic_json_view,
        replace=True,
        default_entry="schematic.monata.json",
        view_format="monata-schematic-json",
        schema_version=2,
        config_factory=_schematic_config,
    )
    register_view_type(
        "testbench",
        _testbench_json_view,
        replace=True,
        default_entry="testbench.monata.json",
        view_format="monata-testbench-json",
        schema_version=1,
        config_factory=_testbench_config,
    )
    register_view_type(
        "testbench_py",
        _testbench_python_view,
        replace=True,
        default_entry="testbench.py",
        view_format="python-testbench",
        trusted=True,
        config_factory=_testbench_py_config,
    )
    register_view_type(
        "netlist",
        _netlist_view,
        replace=True,
        default_entry="netlist.cir",
        view_format="spice",
        generated=True,
        generator=_generate_netlist,
    )
    register_view_type(
        "symbol",
        _symbol_json_view,
        replace=True,
        default_entry="symbol.monata.json",
        view_format="monata-symbol-json",
        schema_version=1,
        generated=True,
        config_factory=_symbol_config,
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


def _schematic_json_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.declarative import SchematicJsonView

    return SchematicJsonView(
        cell=cell,
        entry=str(cfg["entry"]),
        generated=bool(cfg.get("generated", False)),
        schema_version=_optional_int(cfg.get("schema_version")),
    )


def _testbench_json_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.declarative import TestbenchJsonView

    return TestbenchJsonView(
        cell=cell,
        entry=str(cfg["entry"]),
        generated=bool(cfg.get("generated", False)),
        schema_version=_optional_int(cfg.get("schema_version")),
    )


def _testbench_python_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.testbench import TestbenchView

    trusted = _trusted_python_config(cfg, view_format="python-testbench")
    return TestbenchView(
        cell=cell,
        entry=str(cfg["entry"]),
        function_name=str(cfg["function"]),
        trusted=trusted,
    )


def _netlist_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.netlist import NetlistView

    return NetlistView(cell=cell, entry=str(cfg["entry"]))


def _symbol_json_view(cell: Any, cfg: ViewConfig) -> Any:
    from monata.views.declarative import SymbolJsonView

    return SymbolJsonView(
        cell=cell,
        entry=str(cfg["entry"]),
        generated=bool(cfg.get("generated", True)),
        schema_version=_optional_int(cfg.get("schema_version")),
    )


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
    if any(key in options for key in ("cls_name", "class")):
        raise ValueError(
            "monata-schematic-json views cannot include Python class metadata; "
            "Python schematic metadata is not supported for the canonical schematic view"
        )
    return {
        "entry": options.get("entry", schema.default_entry or "schematic.monata.json"),
        "format": "monata-schematic-json",
        "schema_version": options.get("schema_version", 2),
    }


def _unsupported_python_schematic_message() -> str:
    return (
        "legacy Python schematic format is no longer supported for canonical schematic views; "
        "use monata-schematic-json structured schematic data"
    )


def _reject_removed_schematic_view(view_type: object, view_format: object | None = None) -> None:
    if view_type == _REMOVED_SCHEMATIC_PY_VIEW:
        raise ValueError("removed Python schematic view type is no longer built in; " + _unsupported_python_schematic_message())
    if view_format == _REMOVED_PYTHON_SCHEMATIC_FORMAT:
        raise ValueError(_unsupported_python_schematic_message())


def _testbench_config(view_type: str, options: ViewConfigOptions, schema: ViewSchema) -> MutableViewConfig:
    entry = str(options.get("entry", schema.default_entry or "testbench.monata.json"))
    if any(key in options for key in ("function_name", "function")):
        raise ValueError(
            "monata-testbench-json views cannot include Python function metadata; "
            "use testbench_py with trusted = true"
        )
    return {
        "entry": entry,
        "format": "monata-testbench-json",
        "schema_version": options.get("schema_version", 1),
    }


def _symbol_config(view_type: str, options: ViewConfigOptions, schema: ViewSchema) -> MutableViewConfig:
    entry = str(options.get("entry", schema.default_entry or "symbol.monata.json"))
    if entry.endswith(".toml"):
        raise ValueError("symbol.toml views are not supported; use symbol.monata.json")
    return {
        "entry": entry,
        "format": "monata-symbol-json",
        "schema_version": options.get("schema_version", 1),
        "generated": bool(options.get("generated", True)),
    }


def _testbench_py_config(view_type: str, options: ViewConfigOptions, schema: ViewSchema) -> MutableViewConfig:
    _require_trusted_option(options, view_format="python-testbench")
    return {
        "entry": options.get("entry", schema.default_entry or "testbench.py"),
        "format": "python-testbench",
        "trusted": True,
        "function": options.get("function_name", "main"),
    }


def _reject_removed_view_metadata(view_type: str, config: ViewConfig) -> None:
    _reject_removed_schematic_view(view_type, config.get("format"))
    if view_type == "schematic" and "class" in config:
        raise ValueError(
            "schematic Python class metadata is no longer supported; "
            "use monata-schematic-json structured schematic data"
        )
    if view_type == "testbench" and "function" in config:
        raise ValueError(
            "testbench Python views require format = 'python-testbench' and trusted = true; "
            "unformatted function metadata is not supported"
        )
    if view_type == "symbol" and str(config.get("entry", "")).endswith(".toml"):
        raise ValueError("symbol.toml views are not supported; use symbol.monata.json")


def _require_trusted_option(options: ViewConfigOptions, *, view_format: str) -> None:
    if options.get("trusted") is not True:
        raise ValueError(f"{view_format} views require trusted = true")


def _validate_trusted_format(view_format: str, config: ViewConfig) -> None:
    if view_format == "python-testbench":
        _trusted_python_config(config, view_format=view_format)


def _trusted_python_config(config: ViewConfig, *, view_format: str) -> bool:
    if config.get("trusted") is not True:
        raise ValueError(f"{view_format} views require trusted = true")
    return True


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
