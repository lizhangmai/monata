

from monata.netlist import SubCircuit

class Inverter(SubCircuit):
    NAME = "inverter"
    NODES = ("vin", "out", "vdd", "gnd")

    def build(self):
        self.include("/models/nmos.mod")
        self.include("/models/pmos.mod")
        self.param("scale", 1)
        self.mos("mn", d="out", g="vin", s="gnd", b="gnd", model="nmos", w="1u", l="45n")
        self.mos("mp", d="out", g="vin", s="vdd", b="vdd", model="pmos", w="2u", l="45n")
