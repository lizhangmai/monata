"""Model registry for optional compiled model artifacts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from monata.models.cache import ModelCache
from monata.models.compiler import ModelCompiler
from monata.models.diagnostics import ModelDiagnostic, ModelDiagnosticError

_logger = logging.getLogger(__name__)


class ModelEntry:
    """A registered model with its metadata."""

    __slots__ = (
        "name",
        "family",
        "level",
        "version",
        "osdi_path",
        "module_name",
        "source_va",
        "model_file",
        "lib_section",
        "provenance",
        "parameters",
        "include_paths",
    )

    def __init__(
        self,
        name,
        family,
        osdi_path=None,
        module_name=None,
        level=None,
        version=None,
        source_va=None,
        model_file=None,
        lib_section=None,
        provenance=None,
        parameters=None,
        include_paths=None,
    ):
        self.name = name
        self.family = family
        self.level = level
        self.version = version
        self.osdi_path = str(osdi_path) if osdi_path else None
        self.module_name = module_name or name
        self.source_va = str(source_va) if source_va else None
        self.model_file = str(model_file) if model_file else None
        self.lib_section = lib_section
        self.provenance = dict(provenance or {})
        self.parameters = dict(parameters or {})
        self.include_paths = [str(path) for path in (include_paths or ())]

    def to_dict(self):
        data = {
            "name": self.name,
            "family": self.family,
        }
        optional = {
            "level": self.level,
            "version": self.version,
            "osdi_path": self.osdi_path,
            "module_name": self.module_name,
            "source_va": self.source_va,
            "model_file": self.model_file,
            "lib_section": self.lib_section,
            "provenance": self.provenance,
            "parameters": self.parameters,
            "include_paths": self.include_paths,
        }
        for key, value in optional.items():
            if value not in (None, {}, []):
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, d):
        for key in ("name", "family"):
            if d.get(key) in (None, ""):
                raise KeyError(key)
        unknown = sorted(key for key in d if key not in cls.__slots__)
        if unknown:
            raise TypeError(f"unknown model entry fields: {', '.join(unknown)}")
        return cls(**{k: d.get(k) for k in cls.__slots__})


class ModelRegistry:
    """Discover and resolve compiled model artifacts.

    The registry maintains a mapping from (family, level, version) tuples to
    compiled .osdi files for optional simulator/model tooling.

    Search paths (in priority order):
        1. Explicitly registered models (via register())
        2. Project-local models (working directory)
        3. User cache (~/.cache/monata/models/)
        4. System-installed models ($CONDA_PREFIX/lib/vacask/mod/, etc.)
    """

    def __init__(self, search_paths=None, auto_discover=True):
        self._entries: dict[tuple, ModelEntry] = {}
        self._entry_candidates: dict[tuple, list[ModelEntry]] = {}
        self._entry_order: list[ModelEntry] = []
        self._search_paths: list[Path] = []
        self._compiler = ModelCompiler()
        self._cache = ModelCache()

        if search_paths:
            self._search_paths = [Path(p) for p in search_paths]
        else:
            self._search_paths = self._default_search_paths()

        if auto_discover:
            self._discover()

    @staticmethod
    def _default_search_paths() -> list[Path]:
        paths: list[Path] = []
        env = os.environ.get("MONATA_OSDI_PATH")
        if env:
            paths.extend(Path(p) for p in env.split(os.pathsep))

        conda = os.environ.get("CONDA_PREFIX")
        if conda:
            # VACASK installs built-in models here
            vacask_mod = Path(conda) / "lib" / "vacask" / "mod"
            if vacask_mod.is_dir():
                paths.append(vacask_mod)
            # ngspice OSDI path
            ngspice_osdi = Path(conda) / "lib" / "ngspice"
            if ngspice_osdi.is_dir():
                paths.append(ngspice_osdi)

        return paths

    def _discover(self):
        """Scan search paths for .osdi files and register them."""
        for search_path in self._search_paths:
            if not search_path.is_dir():
                continue
            for osdi_file in sorted(search_path.rglob("*.osdi")):
                self._register_from_osdi(osdi_file)

    def _register_from_osdi(self, osdi_path: Path):
        """Infer model metadata from an .osdi file path and register it.

        Convention: the .osdi filename is the module name.
        Subdirectory structure hints at family (e.g., spice/bsim4.osdi).
        """
        module_name = osdi_path.stem
        family = self._infer_family(module_name)
        level, version = self._infer_level_version(module_name)

        entry = ModelEntry(
            name=module_name,
            family=family,
            osdi_path=osdi_path,
            module_name=module_name,
            level=level,
            version=version,
        )
        self._add_entry(entry)

    @staticmethod
    def _infer_family(module_name: str) -> str:
        """Map module name to device family."""
        name = module_name.lower().replace("sp_", "")
        family_hints = {
            "resistor": "r", "capacitor": "c", "inductor": "l",
            "diode": "d", "bjt": "bjt", "vbic": "bjt",
            "hicum": "bjt", "mextram": "bjt",
            "jfet": "jfet", "mes": "mes",
            "mos": "mos", "bsim3": "mos", "bsim4": "mos",
            "bsimbulk": "mos", "bsim6": "mos",
            "ekv": "mos", "psp": "mos", "hisim2": "mos",
            "bsimcmg": "mos", "bsimimg": "mos",
            "bsimsoi": "soi", "lutsoi": "soi", "hisimsoi": "soi",
            "asmhemt": "hemt", "angelov": "hemt", "mvsg": "hemt",
        }
        for hint, fam in family_hints.items():
            if hint in name:
                return fam
        return "unknown"

    @staticmethod
    def _infer_level_version(module_name: str) -> tuple:
        """Try to extract level/version from module name. Returns (level, version)."""
        # Most auto-discovered models don't encode level in filename
        return (None, None)

    def register(self, family, osdi_path, module_name, level=None, version=None, source_va=None):
        """Explicitly register a model.

        Args:
            family: Device family ("mos", "bjt", "d", "r", etc.)
            osdi_path: Path to the compiled .osdi file.
            module_name: OSDI module name used in netlist.
            level: SPICE model level (int or None).
            version: Model version string (or None).
            source_va: Original .va source path (for recompilation).
        """
        entry = ModelEntry(
            name=module_name,
            family=family,
            osdi_path=osdi_path,
            module_name=module_name,
            level=level,
            version=version,
            source_va=source_va,
        )
        self._add_entry(entry)
        _logger.debug("Registered model: %s at %s", module_name, osdi_path)

    def _add_entry(self, entry: ModelEntry) -> None:
        key = (entry.family, entry.level, entry.version)
        self._entry_candidates.setdefault(key, []).append(entry)
        self._entry_order.append(entry)
        if key not in self._entries:
            self._entries[key] = entry

    def register_va(
        self,
        va_path,
        family,
        module_name,
        level=None,
        version=None,
        output_dir=None,
        include_paths=None,
    ):
        """Compile a .va source and register the resulting .osdi.

        Uses the cache to avoid recompilation if the source hasn't changed.

        Args:
            va_path: Path to the Verilog-A source.
            family: Device family.
            module_name: OSDI module name.
            level: SPICE model level.
            version: Model version string.
            output_dir: Where to place the .osdi (defaults to cache).
            include_paths: Source files that affect compilation.

        Returns:
            Path to the compiled .osdi.
        """
        va_path = Path(va_path).resolve()
        include_paths = [Path(path).resolve() for path in (include_paths or ())]
        cached = self._cache.lookup(va_path, include_paths=include_paths)
        if cached:
            osdi_path = cached
        else:
            osdi_path = self._compiler.compile_osdi(
                va_path,
                output_dir=output_dir,
                include_paths=include_paths,
            )
            osdi_path = self._cache.store(va_path, osdi_path, include_paths=include_paths)

        entry = ModelEntry(
            name=module_name,
            family=family,
            osdi_path=osdi_path,
            module_name=module_name,
            level=level,
            version=version,
            source_va=va_path,
            include_paths=include_paths,
        )
        self._add_entry(entry)
        _logger.debug("Registered model: %s at %s", module_name, osdi_path)
        return osdi_path

    def resolve(self, family, level=None, version=None) -> ModelEntry | None:
        """Resolve a (family, level, version) to a ModelEntry.

        Fallback order:
            1. Exact match (family, level, version)
            2. (family, level, None)
            3. (family, None, None)
        """
        for key in self._resolution_keys(family, level=level, version=version):
            if key in self._entries:
                return self._entries[key]
        return None

    @staticmethod
    def _resolution_keys(family, level=None, version=None) -> tuple[tuple, tuple, tuple]:
        return (
            (family, level, version),
            (family, level, None),
            (family, None, None),
        )

    def resolve_candidates(self, family, level=None, version=None) -> list[ModelEntry]:
        """Return all candidates matching the first non-empty fallback key."""
        for key in self._resolution_keys(family, level=level, version=version):
            candidates = self._entry_candidates.get(key, [])
            if candidates:
                return list(candidates)
        return []

    def resolve_strict(self, family, level=None, version=None) -> ModelEntry | None:
        """Resolve one model, raising a diagnostic if the selection is ambiguous."""
        candidates = self.resolve_candidates(family, level=level, version=version)
        if len(candidates) > 1:
            raise ModelDiagnosticError(
                ModelDiagnostic(
                    code="model_selection_ambiguous",
                    message="multiple model entries match the requested selection",
                    context={
                        "family": family,
                        "level": level,
                        "version": version,
                        "candidates": [entry.to_dict() for entry in candidates],
                    },
                )
            )
        return candidates[0] if candidates else None

    def load_entries(self, entries) -> None:
        """Load serialized model entries while preserving duplicate candidates."""
        for item in entries:
            entry = item if isinstance(item, ModelEntry) else ModelEntry.from_dict(item)
            self._add_entry(entry)

    def to_dict(self) -> dict:
        """Serialize registered entries in insertion order."""
        return {"entries": [entry.to_dict() for entry in self._entry_order]}

    def resolve_osdi(self, family, level=None, version=None) -> str | None:
        """Convenience: resolve and return just the .osdi path string."""
        entry = self.resolve(family, level, version)
        return entry.osdi_path if entry else None

    def resolve_osdi_strict(self, family, level=None, version=None) -> str | None:
        """Resolve an OSDI path, diagnosing ambiguous registry candidates."""
        entry = self.resolve_strict(family, level=level, version=version)
        return entry.osdi_path if entry else None

    def osdi_paths(self, family=None, level=None, version=None) -> list[str]:
        """Return OSDI paths suitable for ``SimTask.osdi_paths``.

        With no family, all registered model paths are returned. With a family,
        the normal ``resolve`` fallback is used and zero or one path is returned.
        """
        if family is None:
            return [entry.osdi_path for entry in self.list_models() if entry.osdi_path]
        path = self.resolve_osdi(family, level=level, version=version)
        return [path] if path else []

    def list_models(self, family=None) -> list[ModelEntry]:
        """List all registered models, optionally filtered by family."""
        entries = list(self._entry_order)
        if family:
            entries = [e for e in entries if e.family == family]
        return sorted(entries, key=lambda e: (e.family, e.level or 0, e.version or ""))

    def families(self) -> list[str]:
        """Return all registered device families."""
        return sorted(set(e.family for e in self._entry_order))
