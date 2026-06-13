"""Generic third-party SPICE model/subcircuit library indexing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable, Literal, Mapping, cast

_CATALOG_METADATA_FIELDS = frozenset({"name", "kind", "category", "tags"})


class SpiceLibraryError(ValueError):
    """Raised for invalid SPICE library operations."""


@dataclass(frozen=True)
class SpiceLibraryItem:
    """Indexed SPICE model or subcircuit metadata."""

    kind: Literal["model", "subckt"]
    name: str
    path: Path
    line: int
    pins: tuple[str, ...] = ()
    model_type: str | None = None
    section: str | None = None
    category: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self, root: Path) -> dict[str, object]:
        payload = asdict(self)
        payload["path"] = str(self.path.resolve().relative_to(root.resolve()))
        payload["pins"] = list(self.pins)
        payload["tags"] = list(self.tags)
        return payload

    def source_text(self) -> str:
        """Return the source `.model` or `.subckt` block for this indexed item."""

        return _item_source_text(self)

    def ordered_nodes(self, pins: Mapping[str, Any]) -> tuple[str, ...]:
        """Return instance nodes ordered by this subcircuit item's indexed pins."""

        if self.kind != "subckt":
            raise SpiceLibraryError("only indexed subcircuits can order pin connections")
        if not self.pins:
            raise SpiceLibraryError(f"subcircuit {self.name} has no indexed pins")
        mapped = {str(pin): net for pin, net in pins.items()}
        missing = [pin for pin in self.pins if pin not in mapped]
        if missing:
            raise SpiceLibraryError(f"subcircuit {self.name} is missing pin(s): {', '.join(missing)}")
        known = set(self.pins)
        unknown = [pin for pin in mapped if pin not in known]
        if unknown:
            raise SpiceLibraryError(f"subcircuit {self.name} has unknown pin(s): {', '.join(unknown)}")
        return tuple(str(mapped[pin]) for pin in self.pins)

    @classmethod
    def from_dict(cls, root: Path, payload: dict[str, object]) -> SpiceLibraryItem:
        pins_payload = payload.get("pins", ())
        pins = tuple(str(pin) for pin in pins_payload) if isinstance(pins_payload, (list, tuple)) else ()
        tags_payload = payload.get("tags", ())
        tags = tuple(str(tag) for tag in tags_payload) if isinstance(tags_payload, (list, tuple)) else ()
        return cls(
            kind=_item_kind(payload["kind"]),
            name=str(payload["name"]),
            path=root / str(payload["path"]),
            line=int(str(payload["line"])),
            pins=pins,
            model_type=str(payload["model_type"]) if payload.get("model_type") is not None else None,
            section=str(payload["section"]) if payload.get("section") is not None else None,
            category=str(payload["category"]) if payload.get("category") is not None else None,
            tags=tags,
        )


@dataclass(frozen=True)
class SpiceLibraryAsset:
    """A source SPICE file or `.lib` section with its indexed items."""

    path: Path
    section: str | None
    items: tuple[SpiceLibraryItem, ...] = ()

    @property
    def models(self) -> dict[str, SpiceLibraryItem]:
        return {item.name: item for item in self.items if item.kind == "model"}

    @property
    def subcircuits(self) -> dict[str, SpiceLibraryItem]:
        return {item.name: item for item in self.items if item.kind == "subckt"}

    @property
    def categories(self) -> dict[str, tuple[SpiceLibraryItem, ...]]:
        result: dict[str, list[SpiceLibraryItem]] = {}
        for item in self.items:
            if item.category is not None:
                result.setdefault(item.category, []).append(item)
        return {category: tuple(items) for category, items in sorted(result.items())}

    def source_text(self) -> str:
        """Return this source file or `.lib` section as text."""

        if self.section is None:
            return self.path.read_text()
        return _section_source_text(self.path, self.section)


@dataclass(frozen=True)
class SpiceLibraryReference:
    """A netlist reference to an indexed SPICE library item."""

    item: SpiceLibraryItem

    @property
    def path(self) -> Path:
        return self.item.path

    @property
    def section(self) -> str | None:
        return self.item.section

    @property
    def directive_name(self) -> Literal["include", "lib"]:
        return "lib" if self.section else "include"

    def apply(self, scope: Any, *, once: bool = True) -> SpiceLibraryReference:
        """Attach this reference to a Circuit/SubCircuit-like scope."""

        if self.section:
            if not once or not _has_lib(scope, self.path, self.section):
                scope.lib(self.path, self.section)
        elif not once or str(self.path) not in getattr(scope, "includes", ()):
            scope.include(self.path)
        return self

    def instantiate(
        self,
        scope: Any,
        name: str,
        pins: Mapping[str, Any],
        *,
        once: bool = True,
        **params: Any,
    ) -> Any:
        """Attach this subcircuit reference and create a named-pin instance."""

        if self.item.kind != "subckt":
            raise SpiceLibraryError("only subcircuit library references can be instantiated")
        nodes = self.item.ordered_nodes(pins)
        self.apply(scope, once=once)
        return scope.instance(name, nodes, self.item.name, **params)


class SpiceLibrary:
    """Recursive index for arbitrary SPICE ``.lib/.cir/.mod`` assets."""

    EXTENSIONS = (".spice", ".lib", ".cir", ".mod", ".lib@xyce", ".mod@xyce")
    CACHE_FILENAME = ".monata-spice-library.json"

    def __init__(
        self,
        root: str | Path,
        *,
        scan: bool = True,
        cache_name: str = CACHE_FILENAME,
        follow_references: bool = False,
        catalog: str | Path | Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.cache_path = self.root / cache_name if self.root.is_dir() else self.root.parent / cache_name
        self.follow_references = follow_references
        self._items: dict[tuple[str, str], SpiceLibraryItem] = {}
        if scan:
            self.scan()
        elif self.cache_path.exists():
            self.load_cache()
        if catalog is not None:
            if isinstance(catalog, Mapping):
                self.apply_catalog(catalog)
            else:
                self.load_catalog(catalog)

    @property
    def items(self) -> tuple[SpiceLibraryItem, ...]:
        return tuple(sorted(self._items.values(), key=lambda item: (item.kind, item.name.lower(), str(item.path))))

    @property
    def models(self) -> dict[str, SpiceLibraryItem]:
        return {item.name: item for item in self.items if item.kind == "model"}

    @property
    def subcircuits(self) -> dict[str, SpiceLibraryItem]:
        return {item.name: item for item in self.items if item.kind == "subckt"}

    @property
    def categories(self) -> dict[str, tuple[SpiceLibraryItem, ...]]:
        result: dict[str, list[SpiceLibraryItem]] = {}
        for item in self.items:
            if item.category is None:
                continue
            result.setdefault(item.category, []).append(item)
        return {category: tuple(items) for category, items in sorted(result.items())}

    @property
    def assets(self) -> tuple[SpiceLibraryAsset, ...]:
        grouped: dict[tuple[Path, str | None], list[SpiceLibraryItem]] = {}
        for item in self.items:
            grouped.setdefault((item.path, item.section), []).append(item)
        return tuple(
            SpiceLibraryAsset(
                path=path,
                section=section,
                items=tuple(sorted(items, key=lambda item: (item.kind, item.name.lower(), item.line))),
            )
            for (path, section), items in sorted(
                grouped.items(),
                key=lambda entry: (str(entry[0][0]), entry[0][1] or ""),
            )
        )

    def category_path(self, category: str) -> Path:
        """Return the filesystem path for a library category directory."""

        self._require_directory_root("category paths")
        return self.root.joinpath(*_category_parts(category))

    def add_category(self, category: str) -> Path:
        """Create a category directory under this library root and return it."""

        path = self.category_path(category)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_categories(self) -> tuple[str, ...]:
        """Return category directories relative to this library root."""

        self._require_directory_root("category listing")
        if not self.root.exists():
            return ()
        return tuple(
            sorted(
                path.relative_to(self.root).as_posix()
                for path in self.root.rglob("*")
                if path.is_dir()
            )
        )

    def load_catalog(self, path: str | Path) -> SpiceLibrary:
        """Apply JSON/TOML item category metadata from ``path`` to the current index."""

        self.apply_catalog(_read_catalog(Path(path)))
        return self

    def apply_catalog(self, catalog: Mapping[str, Any]) -> SpiceLibrary:
        """Apply external category/tag metadata to indexed library items."""

        for key, metadata in _catalog_entries(catalog):
            item = self._catalog_item(key, metadata)
            category = _catalog_category(metadata.get("category"), fallback=item.category)
            tags = _catalog_tags(item, category, metadata.get("tags"))
            self._items[(item.kind, item.name.lower())] = replace(item, category=category, tags=tags)
        return self

    def scan(self) -> None:
        self._items.clear()
        paths = [self.root] if self.root.is_file() else _spice_paths(self.root, self.EXTENSIONS)
        seen: set[tuple[Path, str | None]] = set()
        for path in paths:
            self._scan_file(path, seen=seen)
        self.save_cache()

    def save_cache(self) -> Path:
        if self.root.is_dir():
            self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(self.root),
            "items": [item.to_dict(self.root if self.root.is_dir() else self.root.parent) for item in self.items],
        }
        self.cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return self.cache_path

    def load_cache(self) -> None:
        payload = json.loads(self.cache_path.read_text())
        root = self.root if self.root.is_dir() else self.root.parent
        self._items.clear()
        for item_payload in payload.get("items", ()):
            item = SpiceLibraryItem.from_dict(root, item_payload)
            self._items[(item.kind, item.name.lower())] = item

    def find(self, pattern: str, *, kind: Literal["model", "subckt"] | None = None) -> tuple[SpiceLibraryItem, ...]:
        regex = re.compile(pattern, re.IGNORECASE)
        return tuple(item for item in self.items if (kind is None or item.kind == kind) and regex.search(item.name))

    def search(self, pattern: str, *, kind: Literal["model", "subckt"] | None = None) -> tuple[SpiceLibraryItem, ...]:
        """Return indexed items whose names match ``pattern``."""

        return self.find(pattern, kind=kind)

    def by_category(self, category: str) -> tuple[SpiceLibraryItem, ...]:
        key = category.lower()
        return tuple(item for item in self.items if item.category == key)

    def tagged(self, tag: str) -> tuple[SpiceLibraryItem, ...]:
        key = _tag(tag)
        return tuple(item for item in self.items if key in item.tags)

    def include_path(self, name: str, *, kind: Literal["model", "subckt"] | None = None) -> Path:
        return self.get(name, kind=kind).path

    def source(self, name: str, *, kind: Literal["model", "subckt"] | None = None) -> str:
        """Return source text for an indexed model or subcircuit."""

        return self.get(name, kind=kind).source_text()

    def asset(self, path: str | Path, *, section: str | None = None) -> SpiceLibraryAsset:
        """Return the indexed source asset for ``path`` and optional `.lib` section."""

        target = Path(path)
        section_key = _section_key(section)
        for asset in self.assets:
            if _same_path(asset.path, target) and _section_key(asset.section) == section_key:
                return asset
        suffix = f" section {section!r}" if section is not None else ""
        raise KeyError(f"{target}{suffix}")

    def reference(
        self,
        name: str,
        *,
        kind: Literal["model", "subckt"] | None = None,
    ) -> SpiceLibraryReference:
        """Return the include/lib reference needed for an indexed item."""

        return SpiceLibraryReference(self.get(name, kind=kind))

    def attach(
        self,
        scope: Any,
        name: str,
        *,
        kind: Literal["model", "subckt"] | None = None,
        once: bool = True,
    ) -> SpiceLibraryReference:
        """Attach an indexed model/subcircuit file to a native netlist scope."""

        return self.reference(name, kind=kind).apply(scope, once=once)

    def instantiate(
        self,
        scope: Any,
        instance_name: str,
        subckt_name: str,
        pins: Mapping[str, Any],
        *,
        once: bool = True,
        **params: Any,
    ) -> Any:
        """Attach and instantiate an indexed subcircuit using its pin metadata."""

        return self.reference(subckt_name, kind="subckt").instantiate(
            scope,
            instance_name,
            pins,
            once=once,
            **params,
        )

    def get(self, name: str, *, kind: Literal["model", "subckt"] | None = None) -> SpiceLibraryItem:
        key_name = name.lower()
        if kind is not None:
            try:
                return self._items[(kind, key_name)]
            except KeyError as exc:
                raise KeyError(name) from exc
        matches = [item for item in self.items if item.name.lower() == key_name]
        if not matches:
            raise KeyError(name)
        if len(matches) > 1:
            raise SpiceLibraryError(f"ambiguous SPICE library item {name!r}; pass kind=...")
        return matches[0]

    def __getitem__(self, name: str) -> SpiceLibraryItem:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        try:
            self.get(name)
        except (KeyError, SpiceLibraryError):
            return False
        return True

    def __bool__(self) -> bool:
        return bool(self._items)

    def _require_directory_root(self, operation: str) -> None:
        if self.root.exists() and not self.root.is_dir():
            raise SpiceLibraryError(f"SPICE library {operation} require a directory root")

    def _catalog_item(self, key: str, metadata: Mapping[str, Any]) -> SpiceLibraryItem:
        kind, name = _catalog_lookup(key, metadata)
        try:
            return self.get(name, kind=kind)
        except KeyError as exc:
            raise SpiceLibraryError(f"catalog item not found: {key}") from exc

    def _scan_file(
        self,
        path: Path,
        *,
        seen: set[tuple[Path, str | None]],
        section_filter: str | None = None,
    ) -> None:
        resolved = path.resolve()
        scan_key = (resolved, section_filter.lower() if section_filter else None)
        if scan_key in seen:
            return
        seen.add(scan_key)
        if not path.exists() or not path.is_file():
            return

        section: str | None = None
        for line_number, text in _logical_lines(path.read_text()):
            lowered = text.lower()
            if lowered.startswith(".lib "):
                tokens = text.split()
                if len(tokens) == 2:
                    section = tokens[1]
                    continue
            if lowered.startswith(".endl"):
                section = None
                continue
            if self.follow_references and _section_matches(section, section_filter):
                reference = _referenced_spice_file(path, text)
                if reference is not None:
                    reference_path, reference_section = reference
                    self._scan_file(reference_path, seen=seen, section_filter=reference_section)
                    continue
            item = _item_from_line(path, line_number, text, section=section)
            if item is not None:
                if _section_matches(section, section_filter):
                    self._items[(item.kind, item.name.lower())] = item


def _spice_paths(root: Path, extensions: Iterable[str]) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    suffixes = {suffix.lower() for suffix in extensions}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _category_parts(category: str) -> tuple[str, ...]:
    text = str(category).strip()
    parts = tuple(part.strip() for part in text.replace("\\", "/").split("/"))
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise SpiceLibraryError(f"invalid SPICE library category: {category!r}")
    return parts


def _read_catalog(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
    elif suffix == ".toml":
        payload = tomllib.loads(path.read_text())
    else:
        raise SpiceLibraryError(f"unsupported SPICE library catalog format: {path.suffix}")
    if not isinstance(payload, Mapping):
        raise SpiceLibraryError("SPICE library catalog must be a mapping")
    return payload


def _catalog_entries(catalog: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    items = catalog.get("items", catalog)
    entries: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(items, Mapping):
        for key, metadata in items.items():
            entry_key = str(key)
            entries.append((entry_key, _catalog_metadata(metadata, subject=f"SPICE library catalog item {entry_key}")))
        return tuple(entries)
    if isinstance(items, list):
        for index, metadata in enumerate(items):
            entry = _catalog_metadata(metadata, subject=f"SPICE library catalog items[{index}]")
            if "name" not in entry:
                raise SpiceLibraryError("SPICE library catalog list entries require a name")
            entries.append((str(entry["name"]), entry))
        return tuple(entries)
    raise SpiceLibraryError("SPICE library catalog items must be a mapping or list")


def _catalog_metadata(value: Any, *, subject: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        metadata = {str(key): metadata_value for key, metadata_value in value.items()}
        unknown = sorted(key for key in metadata if key not in _CATALOG_METADATA_FIELDS)
        if unknown:
            raise SpiceLibraryError(f"{subject} has unknown fields: {', '.join(unknown)}")
        return metadata
    if isinstance(value, str):
        return {"category": value}
    raise SpiceLibraryError("SPICE library catalog item metadata must be a mapping or category string")


def _catalog_lookup(key: str, metadata: Mapping[str, Any]) -> tuple[Literal["model", "subckt"] | None, str]:
    lookup = str(metadata.get("name", key))
    kind_payload = metadata.get("kind")
    if ":" in lookup:
        prefix, name = lookup.split(":", 1)
        prefix = prefix.strip().lower()
        if prefix in {"model", "subckt"}:
            kind_payload = kind_payload or prefix
            lookup = name
    name = lookup.strip()
    if not name:
        raise SpiceLibraryError("SPICE library catalog item name is required")
    kind = _item_kind(kind_payload) if kind_payload is not None else None
    return kind, name


def _catalog_category(value: Any, *, fallback: str | None) -> str | None:
    if value is None:
        return fallback
    category = str(value).strip().replace("\\", "/").lower()
    if not category:
        raise SpiceLibraryError("SPICE library catalog category is required")
    _category_parts(category)
    return category


def _catalog_tags(item: SpiceLibraryItem, category: str | None, payload: Any) -> tuple[str, ...]:
    tags = set(item.tags)
    if category is not None:
        tags.add(_tag(category))
    for tag in _catalog_tag_values(payload):
        tags.add(_tag(tag))
    return tuple(sorted(tag for tag in tags if tag))


def _catalog_tag_values(payload: Any) -> tuple[str, ...]:
    if payload is None:
        return ()
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, (list, tuple, set, frozenset)):
        return tuple(str(tag) for tag in payload)
    raise SpiceLibraryError("SPICE library catalog tags must be a string or iterable")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("*", ";")):
            continue
        if stripped.startswith("+") and result:
            previous_line, previous_text = result[-1]
            result[-1] = (previous_line, f"{previous_text} {stripped[1:].strip()}")
            continue
        command = _strip_comment(stripped)
        if command:
            result.append((line_number, command))
    return result


def _strip_comment(text: str) -> str:
    for marker in (";", "$"):
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text


def _item_kind(value: object) -> Literal["model", "subckt"]:
    text = str(value)
    if text not in {"model", "subckt"}:
        raise SpiceLibraryError(f"invalid SPICE library item kind: {text}")
    return cast(Literal["model", "subckt"], text)


def _item_from_line(path: Path, line: int, text: str, *, section: str | None) -> SpiceLibraryItem | None:
    tokens = text.split()
    if len(tokens) < 2:
        return None
    command = tokens[0].lower()
    if command == ".subckt" and len(tokens) >= 2:
        pins = tuple(token for token in tokens[2:] if "=" not in token)
        category = _subckt_category(tokens[1], pins)
        return SpiceLibraryItem(
            "subckt",
            tokens[1],
            path,
            line,
            pins=pins,
            section=section,
            category=category,
            tags=_item_tags("subckt", tokens[1], category=category, section=section, pins=pins),
        )
    if command == ".model" and len(tokens) >= 3:
        category = _model_category(tokens[2])
        return SpiceLibraryItem(
            "model",
            tokens[1],
            path,
            line,
            model_type=tokens[2],
            section=section,
            category=category,
            tags=_item_tags("model", tokens[1], category=category, section=section, model_type=tokens[2]),
        )
    return None


def _referenced_spice_file(source_path: Path, text: str) -> tuple[Path, str | None] | None:
    tokens = text.split()
    if len(tokens) < 2:
        return None
    command = tokens[0].lower()
    if command == ".include":
        return _resolve_reference_path(source_path, tokens[1]), None
    if command == ".lib" and len(tokens) >= 3:
        return _resolve_reference_path(source_path, tokens[1]), _unquote(tokens[2])
    return None


def _resolve_reference_path(source_path: Path, token: str) -> Path:
    path = Path(_unquote(token))
    if path.is_absolute():
        return path
    return source_path.parent / path


def _unquote(token: str) -> str:
    text = token.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _section_matches(current: str | None, requested: str | None) -> bool:
    return requested is None or (current is not None and current.lower() == requested.lower())


def _item_source_text(item: SpiceLibraryItem) -> str:
    if not item.path.exists():
        raise FileNotFoundError(item.path)
    lines = item.path.read_text().splitlines()
    start = item.line - 1
    if start < 0 or start >= len(lines):
        raise SpiceLibraryError(f"invalid source line for {item.kind} {item.name}: {item.line}")
    if item.kind == "subckt":
        return _subckt_source_text(item, lines, start)
    return _continued_source_text(lines, start)


def _subckt_source_text(item: SpiceLibraryItem, lines: list[str], start: int) -> str:
    selected: list[str] = []
    for line in lines[start:]:
        selected.append(line)
        if line.strip().lower().startswith(".ends"):
            return "\n".join(selected) + "\n"
    raise SpiceLibraryError(f"subcircuit {item.name} is missing .ends")


def _continued_source_text(lines: list[str], start: int) -> str:
    selected = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.lstrip().startswith("+"):
            break
        selected.append(line)
    return "\n".join(selected) + "\n"


def _section_source_text(path: Path, section: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    selected: list[str] = []
    in_section = False
    for raw in path.read_text().splitlines():
        command = _strip_comment(raw.strip())
        if not in_section:
            if _section_header_matches(command, section):
                selected.append(raw)
                in_section = True
            continue
        selected.append(raw)
        if command.lower().startswith(".endl"):
            return "\n".join(selected) + "\n"
    if in_section:
        raise SpiceLibraryError(f"SPICE library section {section!r} is missing .endl")
    raise SpiceLibraryError(f"SPICE library section not found: {section}")


def _section_header_matches(command: str, section: str) -> bool:
    tokens = command.split()
    return len(tokens) == 2 and tokens[0].lower() == ".lib" and _unquote(tokens[1]).lower() == section.lower()


def _model_category(model_type: str) -> str:
    key = _tag(model_type)
    if key in {"nmos", "pmos", "vdmos", "bsim3", "bsim4", "bsimcmg"}:
        return "mosfet"
    if key in {"npn", "pnp"}:
        return "bjt"
    if key in {"d", "diode"}:
        return "diode"
    if key in {"njf", "pjf", "jfet"}:
        return "jfet"
    return key


def _subckt_category(name: str, pins: tuple[str, ...]) -> str:
    key = _tag(name)
    pin_set = {_tag(pin) for pin in pins}
    if "opamp" in key or {"inp", "inn", "out"}.issubset(pin_set):
        return "amplifier"
    if key.startswith(("inv", "nand", "nor", "xor", "xnor", "buf")):
        return "logic"
    return "subcircuit"


def _item_tags(
    kind: Literal["model", "subckt"],
    name: str,
    *,
    category: str,
    section: str | None,
    pins: tuple[str, ...] = (),
    model_type: str | None = None,
) -> tuple[str, ...]:
    tags = {kind, category, _tag(name)}
    if section:
        tags.add(f"section:{_tag(section)}")
    if model_type:
        tags.add(_tag(model_type))
    if pins:
        tags.add(f"pins:{len(pins)}")
    return tuple(sorted(tag for tag in tags if tag))


def _tag(value: str) -> str:
    return re.sub(r"[^a-z0-9_:+.-]+", "_", str(value).strip().lower()).strip("_")


def _section_key(section: str | None) -> str | None:
    return section.lower() if section is not None else None


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left == right


def _has_lib(scope: Any, path: Path, section: str) -> bool:
    expected = (str(path), section)
    for directive in getattr(scope, "directives", ()):
        if getattr(directive, "raw", False):
            continue
        if getattr(directive, "name", None) != "lib":
            continue
        if tuple(str(arg) for arg in getattr(directive, "args", ())) == expected:
            return True
    return False


__all__ = [
    "SpiceLibrary",
    "SpiceLibraryAsset",
    "SpiceLibraryError",
    "SpiceLibraryItem",
    "SpiceLibraryReference",
]
