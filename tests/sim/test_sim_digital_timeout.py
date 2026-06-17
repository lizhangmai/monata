from __future__ import annotations

from monata.digital.stim import DigitalStimulusConfig
from monata.netlist import SubCircuit


def test_digital_stimulus_threads_timeout_into_sim_tasks() -> None:
    dut = SubCircuit("and2", ("a", "b", "y"))
    stim = DigitalStimulusConfig(
        dut=dut,
        inputs=("a", "b"),
        outputs=("y",),
        period=1e-9,
    )

    timed = stim.build_tasks(timeout=12.5)
    untimed = stim.build_tasks(timeout=None)

    assert timed
    assert untimed
    assert {task.timeout for task in timed} == {12.5}
    assert {task.timeout for task in untimed} == {None}
