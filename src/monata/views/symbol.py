_POWER_NAMES = ("vdd", "vcc", "gnd", "vss")
_OUTPUT_NAMES = ("out",)
_INPUT_NAMES = ("in", "inp", "inn")


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
