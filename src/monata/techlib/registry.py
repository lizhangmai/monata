"""Technology-library object model and discovery registry."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from monata._paths import validate_path_segment
from monata.corner import CornerLike, OperatingCorner, coerce_operating_corner
from monata.models.manifest import ModelSelection
from monata.models.flow import ModelFlowRecipe
from monata.models.registry import ModelEntry
from monata.techlib.parse import parse_corner, parse_device, parse_model_deck
from monata.techlib.projection import PDKInstance, ValidatedPDKInstance, _project_params
from monata.techlib.schema import (
    DeviceCell,
    DeviceView,
    ModelDeck,
    TechlibAttachment,
    TechlibDiscoveryError,
    TechlibError,
    _assert_projectable_view,
    _index_by_name,
)

ENTRY_POINT_GROUP = "monata.techlibs"
_TECHLIB_ROOT_FIELDS = frozenset({
    "techlib",
    "model_decks",
    "model_flows",
    "corners",
    "devices",
    "provenance",
})
_TECHLIB_HEADER_FIELDS = frozenset({"name", "description", "default_corner"})


def _reject_unknown_fields(
    data: Mapping[str, Any], allowed: frozenset[str], subject: str
) -> None:
    unknown = sorted(key for key in data if key not in allowed)
    if unknown:
        raise TechlibError(f"{subject} has unknown fields: {', '.join(unknown)}")


class Techlib:
    """Loaded technology-library metadata."""

    def __init__(
        self,
        name: str,
        root: Path,
        description: str = "",
        devices: Iterable[DeviceCell] = (),
        model_decks: Iterable[ModelDeck] = (),
        model_flows: Iterable[ModelFlowRecipe] = (),
        corners: Iterable[OperatingCorner] = (),
        default_corner: str | None = None,
        provenance: dict[str, Any] | None = None,
    ):
        self.name = validate_path_segment(name, "techlib name")
        self.root = Path(root)
        self.description = description
        self.devices = _index_by_name(devices, "device")
        self.model_decks = _index_by_name(model_decks, "model deck")
        self.model_flows = tuple(model_flows)
        self.corners = _index_by_name(
            (
                _materialize_corner(corner, self.name, self.root, self.model_decks)
                for corner in corners
            ),
            "corner",
        )
        self.default_corner = default_corner
        self.provenance = dict(provenance or {})

    @classmethod
    def load(cls, path: str | Path) -> "Techlib":
        root = Path(path)
        if root.is_file():
            toml_path = root
            root = root.parent
        else:
            toml_path = root / "techlib.toml"
        with open(toml_path, "rb") as file:
            data = tomllib.load(file)
        return cls.from_dict(data, root=root)

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: str | Path) -> "Techlib":
        _reject_unknown_fields(data, _TECHLIB_ROOT_FIELDS, "techlib.toml")
        try:
            header = cast(dict[str, Any], data["techlib"])
        except KeyError as exc:
            raise TechlibError("techlib.toml is missing [techlib]") from exc
        _reject_unknown_fields(header, _TECHLIB_HEADER_FIELDS, "techlib header")
        model_decks = [parse_model_deck(item) for item in data.get("model_decks", [])]
        model_flows = [ModelFlowRecipe.from_dict(item) for item in data.get("model_flows", [])]
        corners = [parse_corner(item) for item in data.get("corners", [])]
        devices = [parse_device(item) for item in data.get("devices", [])]
        techlib = cls(
            name=header["name"],
            root=Path(root),
            description=header.get("description", ""),
            devices=devices,
            model_decks=model_decks,
            model_flows=model_flows,
            corners=corners,
            default_corner=header.get("default_corner"),
            provenance=data.get("provenance", {}),
        )
        techlib._validate_references()
        return techlib

    def _validate_references(self) -> None:
        if self.default_corner and self.default_corner not in self.corners:
            raise TechlibError(f"default corner references unknown corner: {self.default_corner}")
        for corner in self.corners.values():
            if corner.model_deck not in self.model_decks:
                raise TechlibError(
                    f"corner {corner.name} references unknown model deck: {corner.model_deck}"
                )
        for device in self.devices.values():
            declared = set(device.pins)
            for view in device.views.values():
                _validate_view_references(device, view, declared, self.model_decks, self.corners)

    def device(self, name: str) -> DeviceCell:
        try:
            return self.devices[name]
        except KeyError as exc:
            raise TechlibError(f"unknown device cell: {self.name}/{name}") from exc

    def corner(self, name: CornerLike = None) -> OperatingCorner:
        if isinstance(name, OperatingCorner):
            if name.techlib not in (None, self.name):
                raise TechlibError(
                    f"corner {name.name} belongs to techlib {name.techlib}, not {self.name}"
                )
            if name.name in self.corners:
                base = self.corners[name.name]
                return base.with_updates(
                    temperature=name.temperature,
                    voltages=dict(name.voltages),
                    process=name.process or base.process,
                    metadata={**base.metadata, **name.metadata},
                )
            return name
        if name is not None and not isinstance(name, str):
            return self.corner(coerce_operating_corner(name))
        corner_name: str | None = name or self.default_corner
        if not corner_name:
            raise TechlibError(f"techlib {self.name} has no default corner")
        try:
            return self.corners[corner_name]
        except KeyError as exc:
            raise TechlibError(f"unknown corner: {self.name}/{corner_name}") from exc

    def supported_corner_names(self) -> tuple[str, ...]:
        return tuple(self.corners)

    def default_operating_corner(self) -> OperatingCorner:
        return self.corner(None)

    def validate_corner_metadata(
        self,
        corner: CornerLike = None,
        *,
        require_nominal_vdd: bool = False,
        require_model_deck: bool = False,
        require_model_section: bool = False,
        required_device_defaults: Mapping[str, Iterable[str]] | None = None,
    ) -> OperatingCorner:
        """Return a corner only if required simulator metadata is present."""

        resolved = self.corner(corner)
        missing: list[str] = []
        if require_nominal_vdd and resolved.nominal_vdd is None:
            missing.append("nominal_vdd")
        if require_model_deck and not resolved.model_deck:
            missing.append("model_deck")
        if require_model_section and not resolved.section:
            missing.append("section")
        for device, params in (required_device_defaults or {}).items():
            defaults = resolved.defaults_for_device(device)
            for param in params:
                if param not in defaults:
                    missing.append(f"device_defaults.{device}.{param}")
        if missing:
            raise TechlibError(
                f"{self.name}/{resolved.name} is missing required corner metadata: "
                f"{', '.join(missing)}"
            )
        return resolved

    def model_deck(self, name: str) -> ModelDeck:
        try:
            return self.model_decks[name]
        except KeyError as exc:
            raise TechlibError(f"unknown model deck: {self.name}/{name}") from exc

    def model_selection(self, corner: CornerLike = None) -> ModelSelection:
        resolved_corner = self.corner(corner)
        if resolved_corner.model_deck is None:
            raise TechlibError(f"corner {resolved_corner.name} has no model deck")
        deck = self.model_deck(resolved_corner.model_deck)
        entry = ModelEntry(
            name=resolved_corner.name,
            family=self.name,
            module_name=deck.name,
            model_file=deck.resolve_path(self.root),
            lib_section=resolved_corner.section,
            provenance=self.provenance,
        )
        return ModelSelection([entry]).validate_files()

    def resolve_model_flow(
        self,
        corner: CornerLike = None,
        *,
        simulator_profile: Any = None,
        model_config: Any = None,
    ):
        """Resolve simulator-aware model artifacts for a corner.

        Existing ``model_selection()`` remains the concrete model-card projection.
        This method is the explicit simulator-aware entry point.
        """

        from monata.models.resolver import resolve_model_flow

        return resolve_model_flow(
            self,
            corner,
            simulator_profile=simulator_profile,
            model_config=model_config,
        )

    def validate_instance(
        self,
        instance: PDKInstance,
        corner: CornerLike = None,
        require_projectable: bool = False,
    ) -> ValidatedPDKInstance:
        if instance.lib != self.name:
            raise TechlibError(
                f"PDK instance {instance.name} targets {instance.lib}, not techlib {self.name}"
            )
        device = self.device(instance.cell)
        view = device.view(instance.view)
        if require_projectable:
            _assert_projectable_view(view)
        declared_pins = set(device.pins)
        missing = [pin for pin in view.pin_order if pin not in instance.pins]
        unknown = [pin for pin in instance.pins if pin not in declared_pins]
        if missing:
            raise TechlibError(f"PDK instance {instance.name} is missing pins: {missing}")
        if unknown:
            raise TechlibError(f"PDK instance {instance.name} has unknown pins: {unknown}")
        unknown_params = [param for param in instance.params if param not in device.params]
        if unknown_params:
            raise TechlibError(
                f"PDK instance {instance.name} has unknown params: {unknown_params}"
            )
        ordered_nets = tuple(instance.pins[pin] for pin in view.pin_order)
        resolved_corner = self.corner(corner) if corner or self.default_corner else None
        projected_params = _project_params(device, view, instance, resolved_corner)
        if view.primitive == "mos" and view.model_for_corner(resolved_corner) is None:
            corner_name = resolved_corner.name if resolved_corner is not None else "<none>"
            raise TechlibError(
                f"view {device.name}/{view.name} has no model for corner: {corner_name}"
            )
        if (
            resolved_corner is not None
            and view.model_deck is not None
            and resolved_corner.model_deck != view.model_deck
        ):
            raise TechlibError(
                f"corner {resolved_corner.name} uses model deck {resolved_corner.model_deck}, "
                f"but view {device.name}/{view.name} expects {view.model_deck}"
            )
        return ValidatedPDKInstance(
            instance=instance,
            techlib=self,
            device=device,
            view=view,
            ordered_nets=ordered_nets,
            projected_params=projected_params,
            corner=resolved_corner,
        )


class TechlibRegistry:
    """Registry for explicit and package-discovered techlibs."""

    def __init__(
        self,
        search_paths: Iterable[str | Path] | None = None,
        auto_discover: bool = True,
        *,
        strict_discovery: bool = False,
    ):
        self._techlibs: dict[str, Techlib] = {}
        self._search_paths: list[Path] = []
        self._discovery_errors: list[TechlibDiscoveryError] = []
        self._strict_discovery = strict_discovery
        for path in search_paths or ():
            self.add_search_path(path)
        if auto_discover:
            self.discover_entry_points()

    def add_search_path(self, path: str | Path) -> None:
        root = Path(path)
        self._search_paths.append(root)
        self._scan(root)

    def add_techlib(self, techlib: Techlib) -> Techlib:
        existing = self._techlibs.get(techlib.name)
        if existing is not None:
            if existing.root.resolve() == techlib.root.resolve():
                return existing
            raise TechlibError(
                f"duplicate techlib name {techlib.name}: {existing.root} and {techlib.root}"
            )
        self._techlibs[techlib.name] = techlib
        return techlib

    def _scan(self, root: Path) -> None:
        if not root.exists():
            return
        if (root / "techlib.toml").exists():
            self.add_techlib(Techlib.load(root))
            return
        if not root.is_dir():
            return
        for child in root.iterdir():
            if child.is_dir() and (child / "techlib.toml").exists():
                self.add_techlib(Techlib.load(child))

    @property
    def discovery_errors(self) -> tuple[TechlibDiscoveryError, ...]:
        return tuple(self._discovery_errors)

    def discover_entry_points(self, group: str = ENTRY_POINT_GROUP, *, strict: bool | None = None) -> None:
        fail_fast = self._strict_discovery if strict is None else strict
        for entry_point in _entry_points(group):
            try:
                loaded = entry_point.load()
                for path in _entry_point_paths(loaded):
                    self.add_search_path(path)
            except Exception as exc:
                diagnostic = TechlibDiscoveryError(
                    group=group,
                    entry_point=_entry_point_name(entry_point),
                    message=f"{type(exc).__name__}: {exc}",
                )
                self._discovery_errors.append(diagnostic)
                if fail_fast:
                    raise TechlibError(
                        f"failed to discover techlib entry point {diagnostic.entry_point}: "
                        f"{diagnostic.message}"
                    ) from exc

    def list_techlibs(self) -> list[str]:
        return sorted(self._techlibs)

    def resolve(self, name: str) -> Techlib:
        try:
            return self._techlibs[name]
        except KeyError as exc:
            raise TechlibError(f"unknown attached techlib: {name}") from exc

    def validate_instance(
        self,
        instance: PDKInstance,
        attachments: Iterable[TechlibAttachment | str] | None = None,
        corner: CornerLike = None,
        require_projectable: bool = False,
    ) -> ValidatedPDKInstance:
        selected_corner = corner
        if attachments is not None:
            matched = False
            for attachment in _attachments(attachments):
                if attachment.name == instance.lib:
                    matched = True
                    selected_corner = selected_corner or attachment.default_corner
                    break
            if not matched:
                raise TechlibError(f"PDK instance {instance.name} uses unattached techlib: {instance.lib}")
        return self.resolve(instance.lib).validate_instance(
            instance,
            corner=selected_corner,
            require_projectable=require_projectable,
        )

    def __getitem__(self, name: str) -> Techlib:
        return self.resolve(name)

    def __contains__(self, name: str) -> bool:
        return name in self._techlibs


def _validate_view_references(
    device: DeviceCell,
    view: DeviceView,
    declared_pins: set[str],
    model_decks: dict[str, ModelDeck],
    corners: dict[str, OperatingCorner],
) -> None:
    unknown = [pin for pin in view.pin_order if pin not in declared_pins]
    if unknown:
        raise TechlibError(
            f"view {device.name}/{view.name} pin_order references unknown pins: {unknown}"
        )
    if view.model_deck and view.model_deck not in model_decks:
        raise TechlibError(
            f"view {device.name}/{view.name} references unknown model deck: {view.model_deck}"
        )
    unknown_model_corners = [
        corner for corner in view.corner_models if corner not in corners
    ]
    if unknown_model_corners:
        raise TechlibError(
            f"view {device.name}/{view.name} corner_models references unknown corners: "
            f"{unknown_model_corners}"
        )
    unknown_params = [param for param in view.params if param not in device.params]
    if unknown_params:
        raise TechlibError(
            f"view {device.name}/{view.name} references unknown params: {unknown_params}"
        )


def _materialize_corner(
    corner: OperatingCorner,
    techlib_name: str,
    root: Path,
    model_decks: dict[str, ModelDeck],
) -> OperatingCorner:
    if corner.model_deck is None:
        raise TechlibError(f"corner {corner.name} must declare model_deck")
    try:
        deck = model_decks[corner.model_deck]
    except KeyError as exc:
        raise TechlibError(
            f"corner {corner.name} references unknown model deck: {corner.model_deck}"
        ) from exc
    return corner.with_updates(
        techlib=techlib_name,
        model_file=str(deck.resolve_path(root)),
        device_defaults={device: dict(values) for device, values in corner.device_defaults.items()},
        process=corner.process or corner.name,
    )


def _entry_points(group: str):
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return entry_points.select(group=group)
    return cast(Any, entry_points).get(group, ())


def _entry_point_paths(value: Any) -> list[Path]:
    if callable(value):
        value = value()
    if isinstance(value, str | Path):
        return [Path(value)]
    return [Path(item) for item in value]


def _entry_point_name(entry_point: Any) -> str:
    name = getattr(entry_point, "name", None)
    if name:
        return str(name)
    value = getattr(entry_point, "value", None)
    if value:
        return str(value)
    return repr(entry_point)


def _attachments(items: Iterable[TechlibAttachment | str]) -> list[TechlibAttachment]:
    result = []
    for item in items:
        if isinstance(item, TechlibAttachment):
            result.append(item)
        else:
            result.append(TechlibAttachment(str(item)))
    return result


__all__ = [
    "ENTRY_POINT_GROUP",
    "Techlib",
    "TechlibRegistry",
]
