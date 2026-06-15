from collections import OrderedDict

import pytest

from monata.netlist import Circuit, MutationError, SourceValue, SubCircuit, apply_mutation, project_param_overrides, render_ngspice


def _circuit():
    circuit = Circuit("mut")
    circuit.resistor("1", "in", "out", "1k")
    circuit.vdc("DD", "vdd", "0", 1.2)
    circuit.mos("M1", "out", "in", "0", "0", "nmos", w="1u")
    circuit.instance("X1", ("in", "out"), "gain", m=1)
    return circuit


def test_apply_mutation_updates_element_value_without_mutating_original():
    circuit = _circuit()
    mutated = apply_mutation(circuit, "R1.R", "2k")

    assert circuit.element("1", kind="R").value == "1k"
    assert mutated.element("1", kind="R").value == "2k"
    assert "R1 in out 2k" in render_ngspice(mutated)


def test_apply_mutation_updates_source_value_element_param_and_instance_param():
    circuit = _circuit()
    mutated = apply_mutation(circuit, "VDD.V", 1.8)
    mutated = apply_mutation(mutated, "M1.w", "2u")
    mutated = apply_mutation(mutated, "X1.m", 4)

    assert mutated.element("DD", kind="V").value == SourceValue("DC", (1.8,))
    assert mutated.element("M1", kind="M").params["w"] == "2u"
    assert mutated.element("X1", kind="X").params["m"] == 4


def test_apply_mutation_preserves_source_value_local_params():
    circuit = Circuit("pwl")
    circuit.vpwl("shape", "out", "0", (0, 0), ("1n", 1), repeat_time="10n")

    mutated = apply_mutation(circuit, "Vshape.V", "500p")

    assert mutated.element("shape", kind="V").value == SourceValue(
        "PWL",
        ("500p", 0, "1n", 1),
        OrderedDict([("r", "10n")]),
    )


def test_apply_mutation_matches_existing_param_case_insensitively():
    circuit = _circuit()
    mutated = apply_mutation(circuit, "M1.W", "3u")

    assert mutated.element("M1", kind="M").params == {"w": "3u"}


def test_project_param_overrides_splits_globals_and_structured_mutations():
    circuit = _circuit()
    projection = project_param_overrides(circuit, {"gain": 2, "R1.R": "3k"})

    assert projection.param_overrides == {"gain": 2}
    assert projection.circuit.element("1", kind="R").value == "3k"
    assert projection.metadata["structured_mutations"] == [
        {"target": "gain", "kind": "global_param"},
        {"target": "R1.R", "kind": "structured"},
    ]


def test_apply_mutation_reports_missing_targets():
    with pytest.raises(MutationError, match="mutation target not found"):
        apply_mutation(_circuit(), "R404.R", "2k")


def test_apply_mutation_reports_ambiguous_subcircuit_targets():
    circuit = Circuit("ambiguous")
    left = SubCircuit("left", ("in", "out"))
    left.mos("M1", "out", "in", "0", "0", "nmos", w="1u")
    right = SubCircuit("right", ("in", "out"))
    right.mos("M1", "out", "in", "0", "0", "nmos", w="2u")
    circuit.subckt(left)
    circuit.subckt(right)

    with pytest.raises(MutationError, match="ambiguous mutation target"):
        apply_mutation(circuit, "M1.w", "3u")


def test_project_param_overrides_mutates_explicit_raw_directive_provenance():
    circuit = Circuit("raw")
    circuit.raw_directive(".control")
    circuit.raw_directive("let gain = 1")
    circuit.raw_directive(".endc")

    projection = project_param_overrides(circuit, {"raw.1": "let gain = 2"})

    assert "let gain = 2" in render_ngspice(projection.circuit)
    assert projection.metadata["structured_mutations"] == [
        {"target": "raw.1", "kind": "raw_directive", "index": 1}
    ]


def test_project_param_overrides_reports_missing_raw_directive_provenance():
    circuit = Circuit("raw")
    circuit.raw_directive(".control")

    with pytest.raises(MutationError, match="raw directive mutation target not found"):
        project_param_overrides(circuit, {"raw.4": "let gain = 2"})
