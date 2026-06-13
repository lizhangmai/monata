"""Shared SPICE dot-command import contract."""

from __future__ import annotations

DOT_FLOW_COMMANDS = frozenset({"end", "ends", "subckt"})
STRUCTURED_DOT_COMMANDS = frozenset(
    {
        "global",
        "ic",
        "include",
        "lib",
        "meas",
        "measure",
        "model",
        "nodeset",
        "option",
        "options",
        "param",
        "probe",
        "print",
        "save",
        "title",
    }
)
RAW_PRESERVED_DOT_COMMANDS = frozenset(
    {
        "ac",
        "control",
        "dc",
        "disto",
        "distortion",
        "csparam",
        "endc",
        "elseif",
        "else",
        "endif",
        "endl",
        "func",
        "four",
        "fourier",
        "if",
        "noise",
        "op",
        "plot",
        "pss",
        "pz",
        "sens",
        "step",
        "temp",
        "tf",
        "tran",
        "width",
    }
)
DOT_COMMANDS_WITH_IMPORT_CONTRACT = (
    DOT_FLOW_COMMANDS | STRUCTURED_DOT_COMMANDS | RAW_PRESERVED_DOT_COMMANDS
)
SUPPORTED_DOT_COMMANDS = DOT_COMMANDS_WITH_IMPORT_CONTRACT
UNCLASSIFIED_DOT_COMMANDS = SUPPORTED_DOT_COMMANDS - DOT_COMMANDS_WITH_IMPORT_CONTRACT
UNKNOWN_DOT_COMMAND_CONTRACTS = DOT_COMMANDS_WITH_IMPORT_CONTRACT - SUPPORTED_DOT_COMMANDS

__all__ = [
    "DOT_COMMANDS_WITH_IMPORT_CONTRACT",
    "DOT_FLOW_COMMANDS",
    "RAW_PRESERVED_DOT_COMMANDS",
    "STRUCTURED_DOT_COMMANDS",
    "SUPPORTED_DOT_COMMANDS",
    "UNCLASSIFIED_DOT_COMMANDS",
    "UNKNOWN_DOT_COMMAND_CONTRACTS",
]
