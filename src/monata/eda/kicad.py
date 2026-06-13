"""KiCad XML netlist adapter for Monata.

This module converts KiCad's exported XML netlist into Monata's native record
IR. It intentionally avoids a schematic-object wrapper layer: EDA files are
front-end inputs, and `Circuit` remains the simulation/rendering boundary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import keyword
from pathlib import Path
import re
from typing import Any, Literal, Mapping
from xml.etree import ElementTree

from monata._paths import toml_string, validate_path_segment
from monata.netlist import Circuit, Element
from monata.netlist.ir import NetlistError
from monata.netlist.ngspice import render_ngspice

KiCadImportAction = Literal["project", "ignore", "unsupported"]

KICAD_NETLIST_ENTRY = "netlist.cir"
KICAD_IMPORT_METADATA_ENTRY = "import.toml"
_DIRECT_KINDS = frozenset({"R", "C", "L", "V", "I", "D", "J", "M", "Q"})
_SUBCIRCUIT_PREFIXES = frozenset({"U", "X"})
_MODEL_KINDS = frozenset({"D", "J", "M", "Q", "X"})


@dataclass(frozen=True)
class KiCadNodeRef:
    """A component pin reference attached to a KiCad net."""

    ref: str
    pin: str
    pin_function: str | None = None
    pin_type: str | None = None


@dataclass(frozen=True)
class KiCadNet:
    """A KiCad net with attached component pins."""

    code: str | None
    name: str
    nodes: tuple[KiCadNodeRef, ...]


@dataclass(frozen=True)
class KiCadComponent:
    """A KiCad component record relevant to simulation projection."""

    ref: str
    value: str
    fields: Mapping[str, str] = field(default_factory=dict)
    lib: str | None = None
    part: str | None = None
    description: str | None = None

    def field_value(self, *names: str) -> str | None:
        """Return the first matching user field, case-insensitively."""

        lower = {key.lower(): value for key, value in self.fields.items()}
        for name in names:
            value = lower.get(name.lower())
            if value:
                return value
        return None


@dataclass(frozen=True)
class KiCadNetlist:
    """Parsed KiCad XML netlist."""

    title: str
    source: str | None
    components: tuple[KiCadComponent, ...]
    nets: tuple[KiCadNet, ...]
    path: str | None = None

    def component(self, ref: str) -> KiCadComponent:
        key = ref.lower()
        for component in self.components:
            if component.ref.lower() == key:
                return component
        raise KeyError(ref)


@dataclass(frozen=True)
class KiCadImportIssue:
    """A diagnostic found while planning KiCad projection."""

    message: str
    ref: str | None = None


@dataclass(frozen=True)
class KiCadImportStep:
    """One component projection decision."""

    action: KiCadImportAction
    ref: str
    kind: str | None
    nodes: tuple[str, ...] = ()
    model: str | None = None
    value: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class KiCadImportPlan:
    """Inspectable KiCad import plan before native IR projection."""

    netlist: KiCadNetlist
    steps: tuple[KiCadImportStep, ...]
    issues: tuple[KiCadImportIssue, ...] = ()
    policy: "KiCadImportPolicy" = field(default_factory=lambda: KiCadImportPolicy())

    @property
    def supported(self) -> bool:
        return not self.issues

    @property
    def projected_count(self) -> int:
        return self.count("project")

    @property
    def ignored_count(self) -> int:
        return self.count("ignore")

    @property
    def unsupported_count(self) -> int:
        return self.count("unsupported")

    def count(self, action: KiCadImportAction) -> int:
        return sum(1 for step in self.steps if step.action == action)

    def require_supported(self) -> None:
        if self.issues:
            issue = self.issues[0]
            ref = f" {issue.ref}" if issue.ref else ""
            raise KiCadImportError(f"unsupported KiCad component{ref}: {issue.message}")

    def to_circuit(self, *, title: str | None = None) -> Circuit:
        self.require_supported()
        return _plan_to_circuit(self, title=title)


@dataclass(frozen=True)
class KiCadImportPolicy:
    """Controls conservative KiCad component-to-SPICE projection."""

    ground_names: tuple[str, ...] = ("0", "GND", "GNDA", "DGND", "AGND", "VSS")
    value_fields: tuple[str, ...] = ("spice_value", "sim_value")
    model_fields: tuple[str, ...] = ("spice_model", "model", "sim_model")
    subckt_fields: tuple[str, ...] = ("spice_subckt", "subckt", "sim_subckt", "spice_model", "model")
    kind_fields: tuple[str, ...] = ("spice_kind", "spice_prefix", "sim_kind")
    pin_order_fields: tuple[str, ...] = ("spice_pins", "sim_pins")
    ignore_fields: tuple[str, ...] = ("spice_ignore", "sim_ignore", "monata_ignore")
    ref_kind_overrides: Mapping[str, str] = field(default_factory=dict)


class KiCadImportError(ValueError):
    """Raised when a KiCad netlist cannot be projected into native Monata IR."""


def parse_kicad_netlist(text_or_path: str | Path) -> KiCadNetlist:
    """Parse KiCad XML netlist text or an explicit Path."""

    text, source_path = _read_text_or_path(text_or_path)
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise KiCadImportError(f"invalid KiCad XML netlist: {exc}") from exc
    if root.tag != "export":
        raise KiCadImportError(f"expected KiCad <export> root, got <{root.tag}>")
    source = _find_text(root, "design/source")
    title = Path(source).stem if source else "KiCad netlist"
    components = tuple(_parse_component(comp) for comp in root.findall("components/comp"))
    nets = tuple(_parse_net(net) for net in root.findall("nets/net"))
    return KiCadNetlist(title=title, source=source, components=components, nets=nets, path=source_path)


def inspect_kicad_netlist(
    text_or_path: str | Path,
    *,
    policy: KiCadImportPolicy | None = None,
) -> KiCadImportPlan:
    """Inspect how a KiCad XML netlist would project into Monata."""

    import_policy = policy or KiCadImportPolicy()
    netlist = parse_kicad_netlist(text_or_path)
    pin_nets = _component_pin_nets(netlist, import_policy)
    steps: list[KiCadImportStep] = []
    issues: list[KiCadImportIssue] = []
    for component in netlist.components:
        step = _component_step(component, pin_nets.get(component.ref, {}), import_policy)
        steps.append(step)
        if step.action == "unsupported":
            issues.append(KiCadImportIssue(step.detail, ref=component.ref))
    return KiCadImportPlan(netlist=netlist, steps=tuple(steps), issues=tuple(issues), policy=import_policy)


def kicad_netlist_to_circuit(
    text_or_path: str | Path,
    *,
    title: str | None = None,
    policy: KiCadImportPolicy | None = None,
) -> Circuit:
    """Project a supported KiCad XML netlist into Monata's native Circuit IR."""

    return inspect_kicad_netlist(text_or_path, policy=policy).to_circuit(title=title)


def kicad_netlist_to_python(
    text_or_path: str | Path,
    *,
    function_name: str = "build",
    title: str | None = None,
    policy: KiCadImportPolicy | None = None,
) -> str:
    """Convert a supported KiCad XML netlist into Python builder source."""

    _validate_function_name(function_name)
    circuit = kicad_netlist_to_circuit(text_or_path, title=title, policy=policy)
    return _circuit_python(circuit, function_name=function_name)


def import_kicad_netlist(
    project,
    text_or_path: str | Path,
    *,
    library_name: str = "imported",
    cell_name: str | None = None,
    policy: KiCadImportPolicy | None = None,
):
    """Import a KiCad XML netlist as a Monata Cell with a generated netlist view."""

    text, source_path = _read_text_or_path(text_or_path)
    plan = inspect_kicad_netlist(text, policy=policy)
    circuit = plan.to_circuit()
    lib = _ensure_library(project, library_name)
    safe_cell_name = _safe_import_name(
        cell_name or _default_cell_name(source_path, plan.netlist),
        "KiCad imported cell name",
    )
    cell = lib.create_cell(safe_cell_name, description=f"Imported KiCad netlist from {source_path or 'XML text'}")
    cell.write_generated_view("netlist", entry=KICAD_NETLIST_ENTRY, content=render_ngspice(circuit))
    _write_import_metadata(cell, plan=plan, source_path=source_path)
    return cell


def _parse_component(node: ElementTree.Element) -> KiCadComponent:
    ref = node.attrib.get("ref", "").strip()
    if not ref:
        raise KiCadImportError("KiCad component is missing ref")
    libsource = node.find("libsource")
    fields = {
        field_node.attrib.get("name", "").strip(): (field_node.text or "").strip()
        for field_node in node.findall("fields/field")
        if field_node.attrib.get("name", "").strip()
    }
    return KiCadComponent(
        ref=ref,
        value=(_text(node.find("value")) or ref),
        fields=fields,
        lib=libsource.attrib.get("lib") if libsource is not None else None,
        part=libsource.attrib.get("part") if libsource is not None else None,
        description=libsource.attrib.get("description") if libsource is not None else None,
    )


def _parse_net(node: ElementTree.Element) -> KiCadNet:
    code = node.attrib.get("code")
    name = node.attrib.get("name") or (f"net_{code}" if code else "net")
    refs = []
    for ref_node in node.findall("node"):
        ref = ref_node.attrib.get("ref", "").strip()
        pin = ref_node.attrib.get("pin", "").strip()
        if ref and pin:
            refs.append(
                KiCadNodeRef(
                    ref=ref,
                    pin=pin,
                    pin_function=ref_node.attrib.get("pinfunction"),
                    pin_type=ref_node.attrib.get("pintype"),
                )
            )
    return KiCadNet(code=code, name=name, nodes=tuple(refs))


def _component_pin_nets(
    netlist: KiCadNetlist,
    policy: KiCadImportPolicy,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for net in netlist.nets:
        node_name = _node_name(net, policy)
        for node in net.nodes:
            result[node.ref][node.pin] = node_name
    return result


def _component_step(
    component: KiCadComponent,
    pin_nets: Mapping[str, str],
    policy: KiCadImportPolicy,
) -> KiCadImportStep:
    if _truthy(component.field_value(*policy.ignore_fields)):
        return KiCadImportStep("ignore", component.ref, None, detail="component marked ignored")
    kind = _component_kind(component, policy)
    if kind is None:
        return KiCadImportStep("unsupported", component.ref, None, detail="missing supported SPICE kind")
    pin_order = _pin_order(component, pin_nets, policy)
    nodes = tuple(pin_nets[pin] for pin in pin_order if pin in pin_nets)
    try:
        element = _component_element(component, kind, nodes, policy)
    except (KiCadImportError, NetlistError) as exc:
        return KiCadImportStep("unsupported", component.ref, kind, nodes=nodes, detail=str(exc))
    return KiCadImportStep(
        "project",
        component.ref,
        element.kind,
        nodes=element.nodes,
        model=element.model,
        value=str(element.value) if element.value is not None else None,
        detail="projected to native Element",
    )


def _component_element(
    component: KiCadComponent,
    kind: str,
    nodes: tuple[str, ...],
    policy: KiCadImportPolicy,
) -> Element:
    if kind in _SUBCIRCUIT_PREFIXES:
        kind = "X"
    if kind in {"R", "C", "L", "V", "I"}:
        return Element(kind, component.ref, nodes, value=_component_value(component, policy))
    if kind in _MODEL_KINDS:
        return Element(kind, component.ref, nodes, model=_component_model(component, kind, policy))
    raise KiCadImportError(f"unsupported SPICE kind: {kind}")


def _plan_to_circuit(plan: KiCadImportPlan, *, title: str | None) -> Circuit:
    circuit = Circuit(title or plan.netlist.title)
    for step in plan.steps:
        if step.action != "project" or step.kind is None:
            continue
        component = plan.netlist.component(step.ref)
        circuit.add(_component_element(component, step.kind, step.nodes, plan.policy))
    return circuit


def _circuit_python(circuit: Circuit, *, function_name: str) -> str:
    lines = [
        "from monata.netlist import Circuit, Element",
        "",
        "",
        f"def {function_name}():",
        f"    circuit = Circuit({circuit.title!r})",
    ]
    for element in circuit.elements:
        lines.append(f"    circuit.add({_element_python(element)})")
    lines.append("    return circuit")
    return "\n".join(lines) + "\n"


def _element_python(element: Element) -> str:
    args = [
        repr(element.kind),
        repr(element.name),
        repr(element.nodes),
    ]
    if element.value is not None:
        args.append(f"value={element.value!r}")
    if element.model is not None:
        args.append(f"model={element.model!r}")
    if element.params:
        args.append(f"params={dict(element.params)!r}")
    if element.comment is not None:
        args.append(f"comment={element.comment!r}")
    if not element.enabled:
        args.append("enabled=False")
    if element.raw_suffix is not None:
        args.append(f"raw_suffix={element.raw_suffix!r}")
    return f"Element({', '.join(args)})"


def _component_kind(component: KiCadComponent, policy: KiCadImportPolicy) -> str | None:
    override = component.field_value(*policy.kind_fields)
    if override:
        return override.strip().upper()[:1]
    for prefix, kind in policy.ref_kind_overrides.items():
        if component.ref.upper().startswith(prefix.upper()):
            return kind.upper()[:1]
    ref_prefix = _ref_prefix(component.ref)
    if ref_prefix in _DIRECT_KINDS or ref_prefix in _SUBCIRCUIT_PREFIXES:
        return ref_prefix
    return None


def _component_value(component: KiCadComponent, policy: KiCadImportPolicy) -> str:
    return component.field_value(*policy.value_fields) or component.value


def _component_model(component: KiCadComponent, kind: str, policy: KiCadImportPolicy) -> str:
    fields = policy.subckt_fields if kind in _SUBCIRCUIT_PREFIXES else policy.model_fields
    return component.field_value(*fields) or component.value


def _pin_order(
    component: KiCadComponent,
    pin_nets: Mapping[str, str],
    policy: KiCadImportPolicy,
) -> tuple[str, ...]:
    explicit = component.field_value(*policy.pin_order_fields)
    if explicit:
        return tuple(part for part in re.split(r"[\s,]+", explicit.strip()) if part)
    return tuple(sorted(pin_nets, key=_natural_key))


def _node_name(net: KiCadNet, policy: KiCadImportPolicy) -> str:
    name = net.name.strip() or (f"net_{net.code}" if net.code else "net")
    ground_names = {ground.lower() for ground in policy.ground_names}
    if name.lower() in ground_names:
        return "0"
    return re.sub(r"\s+", "_", name)


def _ref_prefix(ref: str) -> str:
    match = re.match(r"[A-Za-z]+", ref)
    if match is None:
        return ""
    return match.group(0).upper()[:1]


def _natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "y", "on"})


def _find_text(root: ElementTree.Element, path: str) -> str | None:
    return _text(root.find(path))


def _text(node: ElementTree.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text or None


def _read_text_or_path(text_or_path: str | Path) -> tuple[str, str | None]:
    if isinstance(text_or_path, Path):
        return text_or_path.read_text(), str(text_or_path)
    return text_or_path, None


def _ensure_library(project, name: str):
    if name in project.list_libraries():
        return project.get_library(name)
    return project.create_library(name)


def _write_import_metadata(cell, *, plan: KiCadImportPlan, source_path: str | None) -> Path:
    metadata_path = cell.path / KICAD_IMPORT_METADATA_ENTRY
    metadata_path.write_text(
        "[import]\n"
        'format = "kicad-xml-netlist"\n'
        f'source = "{toml_string(source_path or "<memory>")}"\n'
        f'title = "{toml_string(plan.netlist.title)}"\n'
        f'kicad_source = "{toml_string(plan.netlist.source or "")}"\n'
        f"components = {len(plan.netlist.components)}\n"
        f"nets = {len(plan.netlist.nets)}\n"
        f"projected = {plan.projected_count}\n"
        f"ignored = {plan.ignored_count}\n"
    )
    return metadata_path


def _default_cell_name(source_path: str | None, netlist: KiCadNetlist) -> str:
    if source_path:
        return Path(source_path).stem
    if netlist.source:
        return Path(netlist.source).stem
    return netlist.title or "kicad_netlist"


def _safe_import_name(value: str, label: str) -> str:
    result = "".join(
        char if char.isascii() and (char.isalnum() or char in "._-") else "_"
        for char in str(value).strip()
    )
    result = result.strip("._-") or "kicad_netlist"
    return validate_path_segment(result, label)


def _validate_function_name(name: str) -> None:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"invalid function_name: {name!r}")


__all__ = [
    "KiCadComponent",
    "KiCadImportError",
    "KiCadImportIssue",
    "KiCadImportPlan",
    "KiCadImportPolicy",
    "KiCadImportStep",
    "KiCadNet",
    "KiCadNetlist",
    "KiCadNodeRef",
    "import_kicad_netlist",
    "inspect_kicad_netlist",
    "kicad_netlist_to_circuit",
    "kicad_netlist_to_python",
    "parse_kicad_netlist",
]
