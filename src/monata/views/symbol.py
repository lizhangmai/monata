from collections.abc import Mapping
from typing import Any

from monata._config import read_toml, reject_unknown_fields
from monata.views.base import View
from monata.errors import ViewNotGeneratedError

_POWER_NAMES = ("vdd", "vcc", "gnd", "vss")
_OUTPUT_NAMES = ("out",)
_INPUT_NAMES = ("in", "inp", "inn")
_SYMBOL_CONFIG_FIELDS = frozenset({"symbol", "pins"})
_SYMBOL_TABLE_FIELDS = frozenset({"name"})
_SYMBOL_PIN_FIELDS = frozenset({"name", "direction"})


def _validate_symbol_data(data: Mapping[str, Any]) -> None:
    reject_unknown_fields(data, _SYMBOL_CONFIG_FIELDS, "symbol.toml")
    symbol = data.get("symbol")
    if symbol is None:
        raise ValueError("symbol.toml is missing [symbol]")
    if not isinstance(symbol, Mapping):
        raise ValueError("[symbol] must be a table")
    reject_unknown_fields(symbol, _SYMBOL_TABLE_FIELDS, "symbol table")
    if "name" not in symbol:
        raise ValueError("symbol table is missing name")
    pins = data.get("pins", [])
    if not isinstance(pins, list):
        raise ValueError("symbol pins must be an array of tables")
    for index, pin in enumerate(pins):
        if not isinstance(pin, Mapping):
            raise ValueError(f"symbol pins[{index}] must be a table")
        reject_unknown_fields(pin, _SYMBOL_PIN_FIELDS, f"symbol pins[{index}]")


def infer_pin_direction(pin_name: str) -> str:
    name = pin_name.lower()
    for power in _POWER_NAMES:
        if power in name:
            return "inout"
    for out in _OUTPUT_NAMES:
        if out in name:
            return "output"
    for inp in _INPUT_NAMES:
        if inp in name:
            return "input"
    return "inout"


class SymbolView(View):
    def __init__(self, cell, entry: str):
        super().__init__(
            view_type="symbol",
            cell=cell,
            entry=entry,
            generated=True,
            format="monata-symbol-toml",
            trusted=False,
        )

    def load(self) -> dict:
        file_path = self.path() / self._entry
        if not file_path.exists():
            raise ViewNotGeneratedError("symbol", self._cell.name)

        data = read_toml(file_path)
        _validate_symbol_data(data)

        return {
            "name": data["symbol"]["name"],
            "pins": data.get("pins", []),
        }
