"""Internal authoring API mixins for native netlist scopes."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from monata.netlist.ir import Directive, Element, ModelCard, SubCircuit


class ScopeDirectiveApi:
    """Directive and metadata helpers shared by Circuit and SubCircuit."""

    includes: list[str]
    params: OrderedDict[str, Any]
    directives: list[Any]

    def include(self, path: Any) -> None:
        from monata.netlist.ir import NetlistError, _assert_single_line
        from monata.spice_library import SpiceLibraryItem, SpiceLibraryReference

        if isinstance(path, SpiceLibraryReference):
            path.apply(self)
            return
        if isinstance(path, SpiceLibraryItem):
            SpiceLibraryReference(path).apply(self)
            return

        path_text = str(path)
        if not path_text:
            raise NetlistError("include path is required")
        _assert_single_line(path_text, "include path")
        if path_text not in self.includes:
            self.includes.append(path_text)

    def param(self, name: str, value: Any) -> None:
        from monata.netlist.ir import NetlistError, _assert_single_line

        if not name:
            raise NetlistError("parameter name is required")
        _assert_single_line(name, "parameter name")
        _assert_single_line(value, f"parameter {name} value")
        self.params[name] = value

    def parameter(self, name: str, expression: Any) -> None:
        self.param(name, expression)

    def directive(self, directive_name: str, *args: Any, **params: Any) -> Directive:
        from monata.netlist.ir import Directive, _params

        directive = Directive(directive_name, tuple(args), _params(params))
        self.directives.append(directive)
        return directive

    def raw_directive(self, line: str) -> Directive:
        from monata.netlist.ir import Directive

        directive = Directive(line, raw=True)
        self.directives.append(directive)
        return directive

    def raw_block(self, lines: str | Iterable[Any]) -> tuple[Directive, ...]:
        from monata.netlist.ir import NetlistError

        raw_lines = lines.splitlines() if isinstance(lines, str) else [str(line) for line in lines]
        directives = tuple(self.raw_directive(line) for line in raw_lines if line.strip())
        if not directives:
            raise NetlistError("raw block requires at least one non-empty line")
        return directives

    @property
    def raw_spice(self) -> str:
        return "\n".join(str(directive.name) for directive in self.directives if directive.raw).rstrip()

    @raw_spice.setter
    def raw_spice(self, lines: Any) -> None:
        from monata.netlist.ir import Directive

        raw_lines = str(lines).splitlines()
        raw_directives = [Directive(line, raw=True) for line in raw_lines if line.strip()]
        self.directives[:] = [
            *raw_directives,
            *(directive for directive in self.directives if not directive.raw),
        ]

    def model(self, model_name: Any, model_type: str | None = None, **params: Any) -> Directive:
        from monata.netlist.ir import ModelCard, NetlistError

        if isinstance(model_name, ModelCard):
            if model_type is not None or params:
                raise NetlistError("ModelCard cannot be combined with model_type or extra params")
            directive = model_name.to_directive()
            self.directives.append(directive)
            return directive
        if model_type is None:
            raise NetlistError(".model requires name and type")
        return self.directive("model", model_name, model_type, **params)

    def model_card(self, model_name: str, model_type: str, **params: Any) -> ModelCard:
        from monata.netlist.ir import ModelCard

        card = ModelCard.create(model_name, model_type, **params)
        self.model(card)
        return card

    def lib(self, path: str | Path, section: str | None = None) -> Directive:
        args = (str(path),) if section is None else (str(path), section)
        for directive in self.directives:
            if getattr(directive, "raw", False):
                continue
            if getattr(directive, "name", None) == "lib" and getattr(directive, "args", ()) == args:
                return directive
        return self.directive("lib", *args)

    def model_ref(
        self,
        *,
        techlib: str,
        corner: str,
        deck: str | None = None,
        section: str | None = None,
        simulator: str | None = None,
    ) -> Directive:
        params = {"techlib": techlib, "corner": corner}
        if deck is not None:
            params["deck"] = deck
        if section is not None:
            params["section"] = section
        if simulator is not None:
            params["simulator"] = simulator
        return self.directive("monata_model_ref", **params)

    def global_(self, *nodes: str) -> Directive:
        return self.directive("global", *nodes)

    def nodeset(self, **nodes: Any) -> Directive:
        return self.directive("nodeset", **nodes)

    def ic(self, **nodes: Any) -> Directive:
        return self.directive("ic", **nodes)

    def options(self, *flags: str, **options: Any) -> Directive:
        from monata.netlist.ir import NetlistError

        params = dict(options)
        for flag in reversed(flags):
            name = str(flag)
            if not name:
                raise NetlistError("option flag is required")
            if name in params:
                raise NetlistError(f"option flag conflicts with keyed option: {name}")
            params = {name: True, **params}
        return self.directive("options", **params)

    def save(self, *vectors: str) -> Directive:
        return self.directive("save", *vectors)

    def probe(self, *vectors: str) -> Directive:
        return self.directive("probe", *vectors)

    def print_(self, analysis: str, *vectors: str) -> Directive:
        return self.directive("print", analysis, *vectors)

    def measure(self, analysis: str, name: str, *expressions: str) -> Directive:
        from monata.netlist.ir import NetlistError

        if not expressions:
            raise NetlistError(".measure requires at least one expression")
        return self.directive("measure", analysis, name, *expressions)


class ScopePrimitiveBase:
    """Base hooks required by primitive construction mixins."""

    def add(self, element: Element) -> Element:
        raise NotImplementedError

    def _add_pdk_instance(self, instance: Any) -> Any:
        raise NotImplementedError


class ScopePassiveApi(ScopePrimitiveBase):
    """Passive primitive construction helpers."""

    def resistor(
        self,
        name: str,
        n1: str,
        n2: str,
        value: Any,
        *,
        model: str | None = None,
        ac: Any = None,
        multiplier: Any = None,
        scale: Any = None,
        temperature: Any = None,
        device_temperature: Any = None,
        noisy: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(
            params,
            ac=ac,
            m=multiplier,
            scale=scale,
            temp=temperature,
            dtemp=device_temperature,
            noisy=noisy,
        )
        return self.add(_element("R", name, (n1, n2), value=value, model=model, params=mapped))

    def semiconductor_resistor(
        self,
        name: str,
        n1: str,
        n2: str,
        value: Any,
        model: str,
        *,
        length: Any = None,
        width: Any = None,
        multiplier: Any = None,
        ac: Any = None,
        scale: Any = None,
        temperature: Any = None,
        device_temperature: Any = None,
        noisy: Any = None,
        **params: Any,
    ) -> Element:
        mapped = _merge_spice_params(
            params,
            l=length,
            w=width,
            m=multiplier,
            ac=ac,
            scale=scale,
            temp=temperature,
            dtemp=device_temperature,
            noisy=noisy,
        )
        return self.resistor(name, n1, n2, value, model=model, **mapped)

    def behavioral_resistor(
        self,
        name: str,
        n1: str,
        n2: str,
        expression: Any,
        *,
        tc1: Any = None,
        tc2: Any = None,
        **params: Any,
    ) -> Element:
        mapped = _merge_spice_params(params, tc1=tc1, tc2=tc2)
        return self.resistor(name, n1, n2, expression, **mapped)

    def capacitor(
        self,
        name: str,
        n1: str,
        n2: str,
        value: Any,
        *,
        model: str | None = None,
        multiplier: Any = None,
        scale: Any = None,
        temperature: Any = None,
        device_temperature: Any = None,
        initial_condition: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(
            params,
            m=multiplier,
            scale=scale,
            temp=temperature,
            dtemp=device_temperature,
            ic=initial_condition,
        )
        return self.add(_element("C", name, (n1, n2), value=value, model=model, params=mapped))

    def semiconductor_capacitor(
        self,
        name: str,
        n1: str,
        n2: str,
        value: Any,
        model: str,
        *,
        length: Any = None,
        width: Any = None,
        multiplier: Any = None,
        scale: Any = None,
        temperature: Any = None,
        device_temperature: Any = None,
        initial_condition: Any = None,
        **params: Any,
    ) -> Element:
        mapped = _merge_spice_params(
            params,
            l=length,
            w=width,
            m=multiplier,
            scale=scale,
            temp=temperature,
            dtemp=device_temperature,
            ic=initial_condition,
        )
        return self.capacitor(name, n1, n2, value, model=model, **mapped)

    def behavioral_capacitor(
        self,
        name: str,
        n1: str,
        n2: str,
        expression: Any,
        *,
        tc1: Any = None,
        tc2: Any = None,
        **params: Any,
    ) -> Element:
        mapped = _merge_spice_params(params, tc1=tc1, tc2=tc2)
        return self.capacitor(name, n1, n2, expression, **mapped)

    def inductor(
        self,
        name: str,
        n1: str,
        n2: str,
        value: Any,
        *,
        model: str | None = None,
        turns_ratio: Any = None,
        multiplier: Any = None,
        scale: Any = None,
        temperature: Any = None,
        device_temperature: Any = None,
        initial_condition: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(
            params,
            nt=turns_ratio,
            m=multiplier,
            scale=scale,
            temp=temperature,
            dtemp=device_temperature,
            ic=initial_condition,
        )
        return self.add(_element("L", name, (n1, n2), value=value, model=model, params=mapped))

    def behavioral_inductor(
        self,
        name: str,
        n1: str,
        n2: str,
        expression: Any,
        *,
        tc1: Any = None,
        tc2: Any = None,
        **params: Any,
    ) -> Element:
        mapped = _merge_spice_params(params, tc1=tc1, tc2=tc2)
        return self.inductor(name, n1, n2, expression, **mapped)

    res = resistor
    cap = capacitor
    ind = inductor
    semi_resistor = semiconductor_resistor
    semi_capacitor = semiconductor_capacitor
    R = resistor
    C = capacitor
    L = inductor
    SemiconductorResistor = semiconductor_resistor
    SemiconductorCapacitor = semiconductor_capacitor
    BehavioralResistor = behavioral_resistor
    BehavioralCapacitor = behavioral_capacitor
    BehavioralInductor = behavioral_inductor


class ScopeSourceApi(ScopePrimitiveBase):
    """Independent source construction helpers."""

    def voltage(self, name: str, p: str, n: str, value: Any, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("V", name, (p, n), value=value, params=params))

    def current(self, name: str, p: str, n: str, value: Any, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("I", name, (p, n), value=value, params=params))

    def vdc(self, name: str, p: str, n: str, value: Any, **params: Any) -> Element:
        from monata.netlist.ir import SourceValue

        return self.voltage(name, p, n, SourceValue("DC", (value,)), **params)

    def idc(self, name: str, p: str, n: str, value: Any, **params: Any) -> Element:
        from monata.netlist.ir import SourceValue

        return self.current(name, p, n, SourceValue("DC", (value,)), **params)

    def vac(self, name: str, p: str, n: str, dc: Any, ac: Any, **params: Any) -> Element:
        from monata.netlist.ir import SourceValue

        return self.voltage(name, p, n, SourceValue("AC", (dc, ac)), **params)

    def iac(self, name: str, p: str, n: str, dc: Any, ac: Any, **params: Any) -> Element:
        from monata.netlist.ir import SourceValue

        return self.current(name, p, n, SourceValue("AC", (dc, ac)), **params)

    def vpulse(
        self,
        name: str,
        p: str,
        n: str,
        initial: Any,
        pulsed: Any,
        delay: Any,
        rise: Any,
        fall: Any,
        width: Any,
        period: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.voltage(
            name,
            p,
            n,
            SourceValue("PULSE", (initial, pulsed, delay, rise, fall, width, period)),
            **params,
        )

    def ipulse(
        self,
        name: str,
        p: str,
        n: str,
        initial: Any,
        pulsed: Any,
        delay: Any,
        rise: Any,
        fall: Any,
        width: Any,
        period: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.current(
            name,
            p,
            n,
            SourceValue("PULSE", (initial, pulsed, delay, rise, fall, width, period)),
            **params,
        )

    def vsin(
        self,
        name: str,
        p: str,
        n: str,
        offset: Any,
        amplitude: Any,
        frequency: Any,
        delay: Any = 0,
        damping: Any = 0,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.voltage(
            name,
            p,
            n,
            SourceValue("SIN", (offset, amplitude, frequency, delay, damping)),
            **params,
        )

    def ac_line(
        self,
        name: str,
        p: str,
        n: str,
        rms_voltage: Any = 230,
        frequency: Any = 50,
        *,
        offset: Any = 0,
        delay: Any = 0,
        damping: Any = 0,
        **params: Any,
    ) -> Element:
        from monata.units import rms_to_amplitude

        return self.vsin(
            name,
            p,
            n,
            offset,
            rms_to_amplitude(rms_voltage),
            frequency,
            delay,
            damping,
            **params,
        )

    def isin(
        self,
        name: str,
        p: str,
        n: str,
        offset: Any,
        amplitude: Any,
        frequency: Any,
        delay: Any = 0,
        damping: Any = 0,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.current(
            name,
            p,
            n,
            SourceValue("SIN", (offset, amplitude, frequency, delay, damping)),
            **params,
        )

    def vexp(
        self,
        name: str,
        p: str,
        n: str,
        initial: Any,
        pulsed: Any,
        rise_delay: Any,
        rise_tau: Any,
        fall_delay: Any,
        fall_tau: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.voltage(
            name,
            p,
            n,
            SourceValue("EXP", (initial, pulsed, rise_delay, rise_tau, fall_delay, fall_tau)),
            **params,
        )

    def iexp(
        self,
        name: str,
        p: str,
        n: str,
        initial: Any,
        pulsed: Any,
        rise_delay: Any,
        rise_tau: Any,
        fall_delay: Any,
        fall_tau: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.current(
            name,
            p,
            n,
            SourceValue("EXP", (initial, pulsed, rise_delay, rise_tau, fall_delay, fall_tau)),
            **params,
        )

    def vpwl(
        self,
        name: str,
        p: str,
        n: str,
        *points: Any,
        repeat_time: Any | None = None,
        delay_time: Any | None = None,
        dc: Any | None = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _pwl_source_value

        value = _pwl_source_value(points, repeat_time=repeat_time, delay_time=delay_time, dc=dc)
        return self.voltage(name, p, n, value, **params)

    def ipwl(
        self,
        name: str,
        p: str,
        n: str,
        *points: Any,
        repeat_time: Any | None = None,
        delay_time: Any | None = None,
        dc: Any | None = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _pwl_source_value

        value = _pwl_source_value(points, repeat_time=repeat_time, delay_time=delay_time, dc=dc)
        return self.current(name, p, n, value, **params)

    def vsffm(
        self,
        name: str,
        p: str,
        n: str,
        offset: Any,
        amplitude: Any,
        carrier_frequency: Any,
        modulation_index: Any,
        signal_frequency: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.voltage(
            name,
            p,
            n,
            SourceValue("SFFM", (offset, amplitude, carrier_frequency, modulation_index, signal_frequency)),
            **params,
        )

    def isffm(
        self,
        name: str,
        p: str,
        n: str,
        offset: Any,
        amplitude: Any,
        carrier_frequency: Any,
        modulation_index: Any,
        signal_frequency: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.current(
            name,
            p,
            n,
            SourceValue("SFFM", (offset, amplitude, carrier_frequency, modulation_index, signal_frequency)),
            **params,
        )

    def vam(
        self,
        name: str,
        p: str,
        n: str,
        amplitude: Any,
        offset: Any,
        modulating_frequency: Any,
        carrier_frequency: Any,
        delay: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.voltage(
            name,
            p,
            n,
            SourceValue("AM", (amplitude, offset, modulating_frequency, carrier_frequency, delay)),
            **params,
        )

    def iam(
        self,
        name: str,
        p: str,
        n: str,
        amplitude: Any,
        offset: Any,
        modulating_frequency: Any,
        carrier_frequency: Any,
        delay: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue

        return self.current(
            name,
            p,
            n,
            SourceValue("AM", (amplitude, offset, modulating_frequency, carrier_frequency, delay)),
            **params,
        )

    def vtrrandom(
        self,
        name: str,
        p: str,
        n: str,
        distribution: str,
        duration: Any,
        delay: Any = 0,
        parameter1: Any = 1,
        parameter2: Any = 0,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue, _trrandom_values

        return self.voltage(
            name,
            p,
            n,
            SourceValue("TRRANDOM", _trrandom_values(distribution, duration, delay, parameter1, parameter2)),
            **params,
        )

    def itrrandom(
        self,
        name: str,
        p: str,
        n: str,
        distribution: str,
        duration: Any,
        delay: Any = 0,
        parameter1: Any = 1,
        parameter2: Any = 0,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import SourceValue, _trrandom_values

        return self.current(
            name,
            p,
            n,
            SourceValue("TRRANDOM", _trrandom_values(distribution, duration, delay, parameter1, parameter2)),
            **params,
        )

    vsource = voltage
    isource = current
    V = voltage
    I = current  # noqa: E741 - SPICE current-source element alias.


class ScopeDeviceApi(ScopePrimitiveBase):
    """Semiconductor and controlled-element construction helpers."""

    def mos(
        self,
        name: str,
        d: str,
        g: str,
        s: str,
        b: str,
        model: str,
        *,
        width: Any = None,
        length: Any = None,
        multiplier: Any = None,
        area_drain: Any = None,
        area_source: Any = None,
        perimeter_drain: Any = None,
        perimeter_source: Any = None,
        drain_squares: Any = None,
        source_squares: Any = None,
        off: bool | None = None,
        initial_condition: Any = None,
        temperature: Any = None,
        fins: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(
            params,
            w=width,
            l=length,
            m=multiplier,
            ad=area_drain,
            as_=area_source,
            pd=perimeter_drain,
            ps=perimeter_source,
            nrd=drain_squares,
            nrs=source_squares,
            off=off,
            ic=_device_initial_condition(initial_condition, expected=3),
            temp=temperature,
            nfin=fins,
        )
        return self.add(_element("M", name, (d, g, s, b), model=model, params=mapped))

    def diode(
        self,
        name: str,
        anode: str,
        cathode: str,
        model: str,
        *,
        area: Any = None,
        multiplier: Any = None,
        junction_perimeter: Any = None,
        off: bool | None = None,
        initial_condition: Any = None,
        temperature: Any = None,
        device_temperature: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(
            params,
            area=area,
            m=multiplier,
            pj=junction_perimeter,
            off=off,
            ic=_device_initial_condition(initial_condition),
            temp=temperature,
            dtemp=device_temperature,
        )
        return self.add(_element("D", name, (anode, cathode), model=model, params=mapped))

    def bjt(
        self,
        name: str,
        collector: str,
        base: str,
        emitter: str,
        model: str,
        substrate: str | None = None,
        *,
        area: Any = None,
        area_collector: Any = None,
        area_base: Any = None,
        multiplier: Any = None,
        off: bool | None = None,
        initial_condition: Any = None,
        temperature: Any = None,
        device_temperature: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        nodes = (collector, base, emitter) if substrate is None else (collector, base, emitter, substrate)
        mapped = _merge_spice_params(
            params,
            area=area,
            areac=area_collector,
            areab=area_base,
            m=multiplier,
            off=off,
            ic=_device_initial_condition(initial_condition, expected=2),
            temp=temperature,
            dtemp=device_temperature,
        )
        return self.add(_element("Q", name, nodes, model=model, params=mapped))

    def jfet(
        self,
        name: str,
        drain: str,
        gate: str,
        source: str,
        model: str,
        *,
        area: Any = None,
        multiplier: Any = None,
        off: bool | None = None,
        initial_condition: Any = None,
        temperature: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(
            params,
            area=area,
            m=multiplier,
            off=off,
            ic=_device_initial_condition(initial_condition, expected=2),
            temp=temperature,
        )
        return self.add(_element("J", name, (drain, gate, source), model=model, params=mapped))

    def mesfet(
        self,
        name: str,
        drain: str,
        gate: str,
        source: str,
        model: str,
        *,
        area: Any = None,
        multiplier: Any = None,
        off: bool | None = None,
        initial_condition: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(
            params,
            area=area,
            m=multiplier,
            off=off,
            ic=_device_initial_condition(initial_condition, expected=2),
        )
        return self.add(_element("Z", name, (drain, gate, source), model=model, params=mapped))

    def vcvs(self, name: str, p: str, n: str, cp: str, cn: str, gain: Any, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("E", name, (p, n, cp, cn), value=gain, params=params))

    def vccs(
        self,
        name: str,
        p: str,
        n: str,
        cp: str,
        cn: str,
        gain: Any,
        *,
        multiplier: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(params, m=multiplier)
        return self.add(_element("G", name, (p, n, cp, cn), value=gain, params=mapped))

    def cccs(
        self,
        name: str,
        p: str,
        n: str,
        source: str,
        gain: Any,
        *,
        multiplier: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(params, m=multiplier)
        return self.add(_element("F", name, (p, n), value=gain, model=source, params=mapped))

    def ccvs(self, name: str, p: str, n: str, source: str, gain: Any, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("H", name, (p, n), value=gain, model=source, params=params))

    def nonlinear_voltage_source(self, name: str, p: str, n: str, expression: Any, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("E", name, (p, n), value=_nonlinear_source_value(expression), params=params))

    def nonlinear_current_source(self, name: str, p: str, n: str, expression: Any, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("G", name, (p, n), value=_nonlinear_source_value(expression), params=params))

    def table_voltage_source(
        self,
        name: str,
        p: str,
        n: str,
        expression: Any,
        points: Iterable[tuple[Any, Any]],
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("E", name, (p, n), value=_table_source_value(expression, points), params=params))

    def table_current_source(
        self,
        name: str,
        p: str,
        n: str,
        expression: Any,
        points: Iterable[tuple[Any, Any]],
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("G", name, (p, n), value=_table_source_value(expression, points), params=params))

    def laplace_voltage_source(
        self,
        name: str,
        p: str,
        n: str,
        input_expression: Any,
        transfer_function: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(
            _element("E", name, (p, n), value=_laplace_source_value(input_expression, transfer_function), params=params)
        )

    def laplace_current_source(
        self,
        name: str,
        p: str,
        n: str,
        input_expression: Any,
        transfer_function: Any,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(
            _element("G", name, (p, n), value=_laplace_source_value(input_expression, transfer_function), params=params)
        )

    def poly_voltage_source(
        self,
        name: str,
        p: str,
        n: str,
        controls: Iterable[tuple[Any, Any]],
        coefficients: Iterable[Any],
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("E", name, (p, n), value=_poly_source_value(controls, coefficients), params=params))

    def poly_current_source(
        self,
        name: str,
        p: str,
        n: str,
        controls: Iterable[tuple[Any, Any]],
        coefficients: Iterable[Any],
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("G", name, (p, n), value=_poly_source_value(controls, coefficients), params=params))

    def switch(
        self,
        name: str,
        p: str,
        n: str,
        cp: str,
        cn: str,
        model: str,
        *,
        initial_state: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(params, initial_state=_switch_initial_state(initial_state))
        return self.add(_element("S", name, (p, n, cp, cn), model=model, params=mapped))

    def current_switch(
        self,
        name: str,
        p: str,
        n: str,
        source: str,
        model: str,
        *,
        initial_state: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        mapped = _merge_spice_params(params, initial_state=_switch_initial_state(initial_state))
        return self.add(_element("W", name, (p, n), value=source, model=model, params=mapped))

    def behavioral(
        self,
        name: str,
        nodes: Iterable[str],
        expression: Any = None,
        *,
        current_expression: Any = None,
        voltage_expression: Any = None,
        tc1: Any = None,
        tc2: Any = None,
        temperature: Any = None,
        device_temperature: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        value, mapped = _behavioral_source_value_and_params(
            expression=expression,
            current_expression=current_expression,
            voltage_expression=voltage_expression,
            params=params,
            tc1=tc1,
            tc2=tc2,
            temperature=temperature,
            device_temperature=device_temperature,
        )
        return self.add(_element("B", name, tuple(nodes), value=value, params=mapped))

    def behavioral_voltage(
        self,
        name: str,
        p: str,
        n: str,
        expression: Any,
        **params: Any,
    ) -> Element:
        return self.behavioral(name, (p, n), voltage_expression=expression, **params)

    def behavioral_current(
        self,
        name: str,
        p: str,
        n: str,
        expression: Any,
        **params: Any,
    ) -> Element:
        return self.behavioral(name, (p, n), current_expression=expression, **params)

    def coupled_inductor(
        self,
        name: str,
        inductors: Iterable[str],
        coupling: Any,
        *,
        validate_inductors: bool = False,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        refs = _inductor_references(inductors)
        _validate_coupling(coupling)
        if validate_inductors:
            for ref in refs:
                if _scope_inductor(self, ref) is None:
                    from monata.netlist.ir import NetlistError

                    raise NetlistError(f"coupled inductor {name} references unknown inductor {ref}")
        return self.add(_element("K", name, refs, value=coupling, params=params))

    def transmission_line(
        self,
        name: str,
        n1: str,
        n2: str,
        n3: str,
        n4: str,
        value: Any = None,
        *,
        impedance: Any = None,
        time_delay: Any = None,
        frequency: Any = None,
        normalized_length: Any = None,
        initial_condition: Any = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        if value is None:
            value = _transmission_line_value(
                impedance=impedance,
                time_delay=time_delay,
                frequency=frequency,
                normalized_length=normalized_length,
                initial_condition=initial_condition,
            )
        elif any(item is not None for item in (impedance, time_delay, frequency, normalized_length, initial_condition)):
            from monata.netlist.ir import NetlistError

            raise NetlistError("raw transmission line value cannot be combined with semantic line parameters")
        return self.add(_element("T", name, (n1, n2, n3, n4), value=value, params=params))

    def lossy_line(self, name: str, n1: str, n2: str, n3: str, n4: str, model: str, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("O", name, (n1, n2, n3, n4), model=model, params=params))

    def txl_line(self, name: str, n1: str, n2: str, n3: str, n4: str, model: str, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("Y", name, (n1, n2, n3, n4), model=model, params=params))

    def coupled_multiconductor_line(
        self,
        name: str,
        nodes: Iterable[str],
        model: str,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("P", name, tuple(nodes), model=model, params=params))

    def distributed_rc_line(
        self,
        name: str,
        output: str,
        input_: str,
        capacitance_node: str,
        model: str,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("U", name, (output, input_, capacitance_node), model=model, params=params))

    def gss_device(self, name: str, nodes: Iterable[str], model: str, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("N", name, tuple(nodes), model=model, params=params))

    def code_model(self, name: str, nodes: Iterable[str], model: str, **params: Any) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element("A", name, tuple(nodes), model=model, params=params))

    def arbitrary(
        self,
        kind: str,
        name: str,
        nodes: Iterable[str],
        value: Any = None,
        model: str | None = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        return self.add(_element(kind, name, tuple(nodes), value=value, model=model, params=params))

    d = diode
    q = bjt
    j = jfet
    z = mesfet
    cpl_line = coupled_multiconductor_line
    lossless_line = transmission_line
    rc_line = distributed_rc_line
    A = code_model
    B = behavioral
    D = diode
    E = vcvs
    F = cccs
    G = vccs
    H = ccvs
    J = jfet
    K = coupled_inductor
    M = mos
    N = gss_device
    O = lossy_line  # noqa: E741 - SPICE lossy-line element alias.
    P = coupled_multiconductor_line
    Q = bjt
    S = switch
    T = transmission_line
    U = distributed_rc_line
    W = current_switch
    Y = txl_line
    Z = mesfet


class ScopeInstanceApi(ScopePrimitiveBase):
    """Subcircuit and source-level PDK instance helpers."""

    def instance(
        self,
        name: str,
        nodes: Iterable[str],
        subckt: str | type[SubCircuit] | SubCircuit,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import NetlistError, SubCircuit, _element

        if isinstance(subckt, str):
            model = subckt
        elif isinstance(subckt, type) and issubclass(subckt, SubCircuit):
            model = subckt.subckt_name()
        elif isinstance(subckt, SubCircuit):
            model = subckt.name
        else:
            raise NetlistError("subckt must be a name or SubCircuit")
        return self.add(_element("X", name, tuple(nodes), model=model, params=params))

    def instance_pins(
        self,
        name: str,
        subckt: str | type[SubCircuit] | SubCircuit,
        pins: Mapping[str, Any],
        *,
        pin_order: Iterable[str] | None = None,
        **params: Any,
    ) -> Element:
        from monata.netlist.ir import _element

        model, ordered_pins = _subckt_model_and_pin_order(subckt, pin_order=pin_order)
        nodes = _ordered_instance_nodes(name, pins, ordered_pins)
        return self.add(_element("X", name, nodes, model=model, params=params))

    def pdk_instance(
        self,
        name: str,
        *,
        lib: str,
        cell: str,
        view: str,
        pins: dict[str, str],
        params: dict[str, Any] | None = None,
    ):
        from monata.techlib.projection import pdk_instance

        instance = pdk_instance(name, lib=lib, cell=cell, view=view, pins=pins, params=params)
        return self._add_pdk_instance(instance)

    x = instance
    X = instance


def _merge_spice_params(params: dict[str, Any], **mapped: Any) -> dict[str, Any]:
    from monata.netlist.ir import NetlistError

    result = dict(params)
    for key, value in mapped.items():
        if value is None:
            continue
        if key in result:
            raise NetlistError(f"parameter {key} was provided both directly and through a semantic alias")
        result[key] = value
    return result


def _switch_initial_state(value: Any) -> str | None:
    from monata.netlist.ir import NetlistError

    if value is None:
        return None
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).lower()
    if text not in {"on", "off"}:
        raise NetlistError("switch initial_state must be 'on', 'off', or a bool")
    return text


def _behavioral_source_value_and_params(
    *,
    expression: Any,
    current_expression: Any,
    voltage_expression: Any,
    params: dict[str, Any],
    tc1: Any,
    tc2: Any,
    temperature: Any,
    device_temperature: Any,
) -> tuple[Any, dict[str, Any]]:
    from monata.netlist.ir import NetlistError

    if expression is not None and (current_expression is not None or voltage_expression is not None):
        raise NetlistError("behavioral source expression cannot be combined with current_expression/voltage_expression")

    mapped = _merge_spice_params(
        params,
        i=current_expression,
        v=voltage_expression,
        tc1=tc1,
        tc2=tc2,
        temp=temperature,
        dtemp=device_temperature,
    )
    value = expression
    if value is None:
        for key in ("i", "v"):
            if key in mapped:
                value = f"{key}={mapped.pop(key)}"
                break
    if value is None:
        raise NetlistError("behavioral source requires expression, current_expression, or voltage_expression")
    return value, mapped


def _nonlinear_source_value(expression: Any) -> str:
    text = str(expression)
    first = text.split(None, 1)[0].lower() if text.strip() else ""
    if first.startswith(("value=", "vol=", "cur=")) or text.upper().startswith(("TABLE", "POLY", "LAPLACE")):
        return text
    return f"value={_braced_expression(text)}"


def _braced_expression(expression: str) -> str:
    stripped = expression.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    return f"{{{expression}}}"


def _table_source_value(expression: Any, points: Iterable[tuple[Any, Any]]) -> str:
    from monata.netlist.ir import NetlistError

    pairs = tuple(points)
    if not pairs:
        raise NetlistError("TABLE source requires at least one point")
    rendered: list[str] = []
    for point in pairs:
        try:
            x_value, y_value = point
        except (TypeError, ValueError) as exc:
            raise NetlistError("TABLE source points must be two-value pairs") from exc
        rendered.append(f"({x_value},{y_value})")
    return f"TABLE {_braced_expression(str(expression))} = {' '.join(rendered)}"


def _laplace_source_value(input_expression: Any, transfer_function: Any) -> str:
    from monata.netlist.ir import NetlistError

    if not str(input_expression).strip():
        raise NetlistError("LAPLACE source requires an input expression")
    if not str(transfer_function).strip():
        raise NetlistError("LAPLACE source requires a transfer function")
    return f"LAPLACE {_braced_expression(str(input_expression))} {_braced_expression(str(transfer_function))}"


def _poly_source_value(controls: Iterable[tuple[Any, Any]], coefficients: Iterable[Any]) -> str:
    from monata.netlist.ir import NetlistError

    control_pairs = tuple(controls)
    if not control_pairs:
        raise NetlistError("POLY source requires at least one control node pair")
    rendered_controls: list[str] = []
    for control in control_pairs:
        if isinstance(control, str):
            raise NetlistError("POLY source controls must be two-value node pairs")
        try:
            positive, negative = control
        except (TypeError, ValueError) as exc:
            raise NetlistError("POLY source controls must be two-value node pairs") from exc
        rendered_controls.extend([str(positive), str(negative)])

    coeffs = tuple(coefficients)
    if not coeffs:
        raise NetlistError("POLY source requires at least one coefficient")
    return f"POLY({len(control_pairs)}) {' '.join(rendered_controls)} {' '.join(str(value) for value in coeffs)}"


def _subckt_model_and_pin_order(
    subckt: str | type[SubCircuit] | SubCircuit,
    *,
    pin_order: Iterable[str] | None,
) -> tuple[str, tuple[str, ...]]:
    from monata.netlist.ir import NetlistError, SubCircuit

    if isinstance(subckt, str):
        if pin_order is None:
            raise NetlistError("pin_order is required when instantiating a named subcircuit by pins")
        model = subckt
        order = tuple(str(pin) for pin in pin_order)
    elif isinstance(subckt, type) and issubclass(subckt, SubCircuit):
        instance = subckt()
        model = instance.name
        order = tuple(str(pin) for pin in (pin_order if pin_order is not None else instance.nodes))
    elif isinstance(subckt, SubCircuit):
        model = subckt.name
        order = tuple(str(pin) for pin in (pin_order if pin_order is not None else subckt.nodes))
    else:
        raise NetlistError("subckt must be a name or SubCircuit")
    if not model:
        raise NetlistError("subcircuit model name is required")
    if not order:
        raise NetlistError("subcircuit pin order is required")
    return model, order


def _ordered_instance_nodes(
    instance_name: str,
    pins: Mapping[str, Any],
    pin_order: tuple[str, ...],
) -> tuple[str, ...]:
    from monata.netlist.ir import NetlistError, _assert_single_line

    mapped = {str(pin): net for pin, net in pins.items()}
    for pin in mapped:
        _assert_single_line(pin, f"X{instance_name} pin name")
    missing = [pin for pin in pin_order if pin not in mapped]
    if missing:
        raise NetlistError(f"X{instance_name} is missing subcircuit pin(s): {', '.join(missing)}")
    known = set(pin_order)
    unknown = [pin for pin in mapped if pin not in known]
    if unknown:
        raise NetlistError(f"X{instance_name} has unknown subcircuit pin(s): {', '.join(unknown)}")
    return tuple(str(mapped[pin]) for pin in pin_order)


def _inductor_references(inductors: Iterable[str]) -> tuple[str, ...]:
    from monata.netlist.ir import NetlistError

    refs = tuple(_inductor_reference(inductor) for inductor in inductors)
    if len(refs) < 2:
        raise NetlistError("coupled inductor requires at least two inductor references")
    return refs


def _inductor_reference(value: Any) -> str:
    text = str(value)
    return text if text.upper().startswith("L") else f"L{text}"


def _scope_inductor(scope: ScopePrimitiveBase, ref: str) -> Element | None:
    candidates = [ref]
    if ref.upper().startswith("L") and len(ref) > 1:
        candidates.append(ref[1:])
    for candidate in candidates:
        element = getattr(scope, "get_element")(candidate, kind="L")
        if element is not None:
            return element
    return None


def _validate_coupling(value: Any) -> None:
    from math import isfinite

    from monata.netlist.ir import NetlistError

    if isinstance(value, bool):
        raise NetlistError("coupled inductor coupling must be a finite scalar or parameter expression")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if not isfinite(number) or number < -1 or number > 1:
        raise NetlistError("coupled inductor coupling must be between -1 and 1")


def _transmission_line_value(
    *,
    impedance: Any,
    time_delay: Any,
    frequency: Any,
    normalized_length: Any,
    initial_condition: Any,
) -> str:
    from monata.netlist.ir import NetlistError

    if time_delay is None and frequency is None:
        raise NetlistError("transmission line requires time_delay or frequency with normalized_length")
    if time_delay is not None and (frequency is not None or normalized_length is not None):
        raise NetlistError("transmission line time_delay cannot be combined with frequency/normalized_length")
    if frequency is not None and normalized_length is None:
        raise NetlistError("transmission line frequency requires normalized_length")
    if normalized_length is not None and frequency is None:
        raise NetlistError("transmission line normalized_length requires frequency")

    parts = [f"z0={50 if impedance is None else impedance}"]
    if time_delay is not None:
        parts.append(f"td={time_delay}")
    else:
        parts.extend([f"f={frequency}", f"nl={normalized_length}"])
    if initial_condition is not None:
        parts.append(f"ic={_line_initial_condition(initial_condition)}")
    return " ".join(parts)


def _line_initial_condition(value: Any) -> str:
    from monata.netlist.ir import NetlistError

    if isinstance(value, str):
        return value
    try:
        values = tuple(value)
    except TypeError as exc:
        raise NetlistError("transmission line initial_condition must be a four-value iterable or string") from exc
    if len(values) != 4:
        raise NetlistError("transmission line initial_condition expects four values: v1, i1, v2, i2")
    return ",".join(str(item) for item in values)


def _device_initial_condition(value: Any, *, expected: int | None = None) -> str | None:
    from monata.netlist.ir import NetlistError

    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        values = tuple(value)
    except TypeError:
        return str(value)
    if expected is not None and len(values) != expected:
        raise NetlistError(f"device initial_condition expects {expected} values")
    return ",".join(str(item) for item in values)
