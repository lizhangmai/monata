from monata.netlist import Circuit


def _dc_circuit():
    circuit = Circuit("dc smoke")
    circuit.voltage("1", "in", "0", "0")
    circuit.resistor("1", "in", "0", "1k")
    return circuit


def _dc_dual_sweep_circuit():
    circuit = Circuit("dc dual sweep smoke")
    circuit.voltage("1", "in", "0", "0")
    circuit.voltage("2", "bias", "0", "0")
    circuit.resistor("1", "in", "out", "1k")
    circuit.resistor("2", "out", "bias", "1k")
    return circuit


def _tran_circuit():
    circuit = Circuit("tran smoke")
    circuit.voltage("1", "in", "0", "pulse(0 1 0 1n 1n 5n 10n)")
    circuit.resistor("1", "in", "out", "1k")
    circuit.capacitor("1", "out", "0", "1n")
    return circuit


def _ac_circuit():
    circuit = Circuit("ac smoke")
    circuit.voltage("1", "in", "0", "dc 0 ac 1")
    circuit.resistor("1", "in", "out", "1k")
    circuit.capacitor("1", "out", "0", "1n")
    return circuit


def _distortion_circuit():
    circuit = Circuit("distortion smoke")
    circuit.voltage("1", "in", "0", "dc 0 ac 1 distof1 1 0")
    circuit.resistor("1", "in", "out", "1k")
    circuit.capacitor("1", "out", "0", "1n")
    return circuit


def _fourier_circuit():
    circuit = Circuit("fourier smoke")
    circuit.voltage("1", "in", "0", "sin(0 1 1k)")
    circuit.resistor("1", "in", "out", "1k")
    circuit.capacitor("1", "out", "0", "1n")
    return circuit
