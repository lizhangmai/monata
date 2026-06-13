"""Internal state containers for native netlist scopes."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class ElementIndex:
    """Case-insensitive element lookup index for one Circuit/SubCircuit scope."""

    def __init__(self, error_type: type[Exception]) -> None:
        self._error_type = error_type
        self._element_names: set[str] = set()
        self._rendered_element_names: set[str] = set()
        self._element_by_name: OrderedDict[str, Any] = OrderedDict()
        self._element_by_kind_name: OrderedDict[str, Any] = OrderedDict()
        self._element_by_instance_name: OrderedDict[str, Any] = OrderedDict()
        self._ambiguous_element_names: set[str] = set()

    def add(self, element: Any) -> Any:
        key = f"{element.kind}:{element.name.lower()}"
        if key in self._element_names:
            raise self._error_type(f"duplicate element name in scope: {element.name}")
        instance_name = _instance_name(element)
        instance_key = instance_name.lower()
        if instance_key in self._rendered_element_names:
            raise self._error_type(f"duplicate rendered element name in scope: {instance_name}")
        self._element_names.add(key)
        self._rendered_element_names.add(instance_key)
        self._element_by_kind_name[key] = element
        self._element_by_instance_name[instance_key] = element

        bare_key = element.name.lower()
        if bare_key in self._element_by_name:
            self._ambiguous_element_names.add(bare_key)
            del self._element_by_name[bare_key]
        elif bare_key not in self._ambiguous_element_names:
            self._element_by_name[bare_key] = element
        return element

    def get(self, name: str, kind: str | None = None) -> Any | None:
        text = str(name)
        if kind is not None:
            kind_text = str(kind).upper()
            element = self._element_by_kind_name.get(f"{kind_text}:{text.lower()}")
            if element is not None:
                return element
            element = self._element_by_instance_name.get(text.lower())
            if element is not None and element.kind == kind_text:
                return element
            return None
        key = text.lower()
        if key in self._ambiguous_element_names:
            return None
        return self._element_by_name.get(key) or self._element_by_instance_name.get(key)

    def rebuild(self, elements: list[Any]) -> None:
        self.clear()
        for element in elements:
            self.add(element)

    def clear(self) -> None:
        self._element_names.clear()
        self._rendered_element_names.clear()
        self._element_by_name.clear()
        self._element_by_kind_name.clear()
        self._element_by_instance_name.clear()
        self._ambiguous_element_names.clear()


def _instance_name(element: Any) -> str:
    name = str(element.name)
    kind = str(element.kind).upper()
    return name if name.startswith(kind) else f"{kind}{name}"


class PDKInstanceQueue:
    """Source-level PDK instances awaiting techlib projection."""

    def __init__(self) -> None:
        self.items: list[Any] = []

    def append(self, instance: Any) -> Any:
        self.items.append(instance)
        return instance


class ScopeState:
    """Mutable collections owned by a native netlist scope."""

    def __init__(self, error_type: type[Exception]) -> None:
        self._error_type = error_type
        self.includes: list[str] = []
        self.params: OrderedDict[str, Any] = OrderedDict()
        self.directives: list[Any] = []
        self.elements: list[Any] = []
        self.explicit_nodes: list[str] = []
        self.pdk_instances = PDKInstanceQueue()
        self.element_index = ElementIndex(error_type)

    def add_element(self, element: Any) -> Any:
        self.element_index.add(element)
        self.elements.append(element)
        return element

    def get_element(self, name: str, kind: str | None = None) -> Any | None:
        return self.element_index.get(name, kind=kind)

    def remove_element(self, element: Any) -> Any:
        for index, candidate in enumerate(self.elements):
            if candidate is element:
                removed = self.elements.pop(index)
                self.rebuild_element_index()
                return removed
        name = getattr(element, "name", element)
        raise self._error_type(f"element not found in scope: {name}")

    def replace_element(self, old: Any, new: Any) -> Any:
        for index, candidate in enumerate(self.elements):
            if candidate is old:
                self.elements[index] = new
                try:
                    self.rebuild_element_index()
                except Exception:
                    self.elements[index] = old
                    self.rebuild_element_index()
                    raise
                return new
        name = getattr(old, "name", old)
        raise self._error_type(f"element not found in scope: {name}")

    def rebuild_element_index(self) -> None:
        self.element_index.rebuild(self.elements)

    def add_pdk_instance(self, instance: Any) -> Any:
        return self.pdk_instances.append(instance)
