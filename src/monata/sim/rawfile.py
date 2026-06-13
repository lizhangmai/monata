"""Public loaders for simulator result files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from monata.sim.backends.ngspice_output import RawfileResult, RawfileVector, parse_rawfile
from monata.sim.results import AnalysisResult, Waveform
from monata.sim._vector_identity import p3_vector_metadata, pole_zero_vector_kind
from monata.sim.vector_names import normalize_vector_name

_RAW_KIND_UNITS = {
    "time": "s",
    "frequency": "Hz",
    "voltage": "V",
    "current": "A",
}
_RAW_KIND_QUANTITIES = {
    "time": "time",
    "frequency": "frequency",
    "voltage": "voltage",
    "current": "current",
}
_ANALYSES_WITHOUT_ABSCISSA = {"op", "pz"}


@dataclass(frozen=True)
class _RawfileCaseContext:
    nodes: Mapping[str, str]
    elements: Mapping[str, str]

    @property
    def active(self) -> bool:
        return bool(self.nodes or self.elements)


def load_ngspice_rawfile(
    path: str | Path,
    *,
    analysis: str | None = None,
    output_names: Iterable[str] | None = None,
    node_names: Iterable[str] | None = None,
    element_names: Iterable[str] | None = None,
) -> AnalysisResult:
    """Load an ngspice rawfile as a typed Monata ``AnalysisResult``.

    The returned result keeps the rawfile scale vector as ``abscissa`` for
    swept analyses and exposes all non-scale vectors unless ``output_names``
    selects a subset. Operating-point and pole-zero rawfiles have no true sweep
    axis, so all variables are exposed as waveforms and ``abscissa`` is
    ``None``. Names may be raw vector names such as ``v(out)`` or their
    normalized Monata lookup names such as ``out``. ``node_names`` and
    ``element_names`` can be supplied to restore the original circuit casing for
    display names when a rawfile lowercases ``v(node)`` or ``i(element)``
    vectors.
    """

    rawfile = parse_rawfile(path)
    return _rawfile_to_analysis_result(
        rawfile,
        analysis=analysis,
        output_names=output_names,
        source_path=Path(path),
        case_context=_case_context(node_names=node_names, element_names=element_names),
    )


def _rawfile_to_analysis_result(
    rawfile: RawfileResult,
    *,
    analysis: str | None,
    output_names: Iterable[str] | None,
    source_path: Path | None,
    case_context: _RawfileCaseContext | None = None,
) -> AnalysisResult:
    case_context = case_context or _case_context()
    analysis_name = analysis or _analysis_from_plotname(rawfile.plotname)
    abscissa = (
        None
        if _analysis_without_abscissa(analysis_name)
        else _abscissa_waveform(rawfile.scale, case_context=case_context)
    )
    selected_vectors = _select_vectors(
        rawfile,
        output_names,
        case_context=case_context,
        include_scale=abscissa is None,
    )
    waveforms = {
        _cased_vector_name(vector.name, case_context): _waveform_from_raw_vector(
            vector,
            abscissa=abscissa,
            case_context=case_context,
            analysis=analysis_name,
        )
        for vector in selected_vectors
    }
    metadata = {
        **rawfile.metadata,
        "analysis": analysis_name,
        "simulator": "ngspice",
        "source_path": str(source_path) if source_path is not None else None,
        "scale_vector": rawfile.scale.name if abscissa is not None else None,
        "vector_raw_names": {name: waveform.raw_vector_name for name, waveform in waveforms.items()},
        "vector_units": _vector_units_metadata(analysis_name, waveforms),
        "vector_quantities": {
            name: waveform.quantity
            for name, waveform in waveforms.items()
            if waveform.quantity is not None
        },
        "vector_kinds": {name: waveform.vector_kind for name, waveform in waveforms.items()},
        "vector_metadata": _vector_metadata(analysis_name, waveforms),
    }
    if case_context.active:
        metadata["rawfile_case_map"] = {
            "nodes": dict(case_context.nodes),
            "elements": dict(case_context.elements),
        }
    return AnalysisResult(
        analysis=analysis_name,
        waveforms=waveforms,
        abscissa=abscissa,
        metadata={key: value for key, value in metadata.items() if value is not None},
        source="rawfile",
    )


def _abscissa_waveform(vector: RawfileVector, *, case_context: _RawfileCaseContext) -> Waveform:
    cased_name = _cased_vector_name(vector.name, case_context)
    identity = normalize_vector_name(cased_name)
    return Waveform(
        name=identity.display_name,
        data=_raw_abscissa_data(vector),
        unit=_raw_unit(vector),
        quantity=_raw_quantity(vector) or identity.quantity,
        source_vector=cased_name,
        raw_vector_name=vector.name,
        vector_kind="abscissa",
        metadata={"rawfile_index": vector.index, "rawfile_kind": vector.kind},
    )


def _waveform_from_raw_vector(
    vector: RawfileVector,
    *,
    abscissa: Waveform | None,
    case_context: _RawfileCaseContext,
    analysis: str | None,
) -> Waveform:
    cased_name = _cased_vector_name(vector.name, case_context)
    identity = normalize_vector_name(cased_name)
    vector_kind = _raw_vector_kind(vector, identity.vector_kind, analysis=analysis)
    quantity = _raw_vector_quantity(vector, identity_quantity=identity.quantity, vector_kind=vector_kind, analysis=analysis)
    unit = _raw_vector_unit(vector, quantity=quantity, analysis=analysis)
    metadata = {
        "rawfile_index": vector.index,
        "rawfile_kind": vector.kind,
        **p3_vector_metadata(analysis, cased_name, vector_kind),
    }
    return Waveform(
        name=identity.display_name,
        data=vector.data,
        unit=unit,
        quantity=quantity,
        source_vector=cased_name,
        raw_vector_name=vector.name,
        vector_kind=vector_kind,
        abscissa=abscissa.name if abscissa is not None else None,
        metadata=metadata,
        display_name=identity.display_name,
        normalized_name=identity.normalized_name,
        source="rawfile",
        extraction="rawfile",
    )


def _select_vectors(
    rawfile: RawfileResult,
    output_names: Iterable[str] | None,
    *,
    case_context: _RawfileCaseContext,
    include_scale: bool = False,
) -> tuple[RawfileVector, ...]:
    if include_scale:
        vectors = tuple(rawfile.vectors)
    else:
        vectors = tuple(vector for vector in rawfile.vectors if vector.index != rawfile.scale.index)
    if output_names is None:
        return vectors
    selected: list[RawfileVector] = []
    seen: set[int] = set()
    for name in output_names:
        vector = _lookup_vector(vectors, name, case_context=case_context)
        if vector is None:
            if _matches_vector(rawfile.scale, name, case_context=case_context):
                raise KeyError(f"rawfile vector {name!r} is the scale vector; use result.abscissa")
            available = ", ".join(vector.name for vector in vectors)
            raise KeyError(f"rawfile vector not found: {name}; available: {available}")
        if vector.index not in seen:
            seen.add(vector.index)
            selected.append(vector)
    return tuple(selected)


def _lookup_vector(
    vectors: Iterable[RawfileVector],
    name: str,
    *,
    case_context: _RawfileCaseContext,
) -> RawfileVector | None:
    selected: RawfileVector | None = None
    selected_score: int | None = None
    for vector in vectors:
        score = _vector_match_score(vector, name, case_context=case_context)
        if score is not None and (selected_score is None or score < selected_score):
            selected = vector
            selected_score = score
    return selected


def _matches_vector(vector: RawfileVector, name: str, *, case_context: _RawfileCaseContext) -> bool:
    return _vector_match_score(vector, name, case_context=case_context) is not None


def _vector_match_score(vector: RawfileVector, name: str, *, case_context: _RawfileCaseContext) -> int | None:
    candidate = str(name).strip()
    if not candidate:
        return None
    cased_name = _cased_vector_name(vector.name, case_context)
    identities = (normalize_vector_name(vector.name), normalize_vector_name(cased_name))
    normalized_candidate = normalize_vector_name(candidate).normalized_name
    exact_values = {
        vector.name.lower(),
        cased_name.lower(),
    }
    normalized_values: set[str] = set()
    for identity in identities:
        exact_values.add(identity.display_name.lower())
        normalized_values.add(identity.normalized_name.lower())
    if candidate.lower() in exact_values:
        return 0
    if normalized_candidate.lower() in normalized_values:
        return 1
    return None


def _case_context(
    *,
    node_names: Iterable[str] | None = None,
    element_names: Iterable[str] | None = None,
) -> _RawfileCaseContext:
    return _RawfileCaseContext(
        nodes=_case_lookup(node_names),
        elements=_case_lookup(element_names),
    )


def _case_lookup(names: Iterable[str] | None) -> dict[str, str]:
    return {str(name).lower(): str(name) for name in names or ()}


def _cased_vector_name(name: str, case_context: _RawfileCaseContext) -> str:
    text = str(name).strip()
    lower = text.lower()
    if lower.startswith("v(") and lower.endswith(")"):
        inner = text[2:-1]
        if "," in inner:
            nodes = tuple(part.strip() for part in inner.split(","))
            cased_nodes = tuple(case_context.nodes.get(node.lower(), node) for node in nodes)
            return f"v({','.join(cased_nodes)})"
        return f"v({case_context.nodes.get(inner.lower(), inner)})"
    if lower.startswith("i(") and lower.endswith(")"):
        inner = text[2:-1]
        return f"i({case_context.elements.get(inner.lower(), inner)})"
    if lower.endswith("#branch"):
        inner = text[:-7]
        return f"{case_context.elements.get(inner.lower(), inner)}#branch"
    return text


def _raw_unit(vector: RawfileVector) -> str | None:
    return _RAW_KIND_UNITS.get(vector.kind.lower())


def _raw_quantity(vector: RawfileVector) -> str | None:
    return _RAW_KIND_QUANTITIES.get(vector.kind.lower())


def _raw_vector_quantity(
    vector: RawfileVector,
    *,
    identity_quantity: str | None,
    vector_kind: str,
    analysis: str | None,
) -> str | None:
    if analysis == "sens":
        return "sensitivity"
    if analysis == "pz":
        return vector_kind
    return _raw_quantity(vector) or identity_quantity


def _raw_vector_unit(vector: RawfileVector, *, quantity: str | None, analysis: str | None) -> str | None:
    if analysis in {"pz", "sens"}:
        return None
    if analysis == "disto":
        if quantity == "voltage":
            return "V"
        if quantity == "current":
            return "A"
    return _raw_unit(vector)


def _raw_abscissa_data(vector: RawfileVector) -> np.ndarray:
    if vector.kind.lower() not in {"time", "frequency"}:
        return vector.data
    if not np.iscomplexobj(vector.data):
        return vector.data
    imaginary = np.imag(vector.data)
    if not np.allclose(imaginary, 0.0):
        return vector.data
    return np.real(vector.data)


def _raw_vector_kind(vector: RawfileVector, inferred: str, *, analysis: str | None = None) -> str:
    if analysis == "sens":
        return "sensitivity"
    if analysis == "pz":
        return pole_zero_vector_kind(vector.name)
    if analysis == "disto":
        return "distortion"
    kind = vector.kind.lower()
    name = str(vector.name).strip().lower()
    if kind == "current":
        if name.startswith("i(") or name.endswith("#branch"):
            return "branch_current"
        if name.startswith("@"):
            return "node_current"
        if inferred != "unknown":
            return inferred
        return "branch_current"
    if inferred != "unknown":
        return inferred
    if kind == "voltage":
        return "node_voltage"
    return "unknown"


def _vector_units_metadata(analysis: str | None, waveforms: Mapping[str, Waveform]) -> dict[str, str | None]:
    if analysis in {"pz", "sens"}:
        return {name: waveform.unit for name, waveform in waveforms.items()}
    return {name: waveform.unit for name, waveform in waveforms.items() if waveform.unit is not None}


def _vector_metadata(analysis: str | None, waveforms: Mapping[str, Waveform]) -> dict[str, dict[str, str]]:
    if analysis not in {"disto", "pz", "sens"}:
        return {}
    metadata: dict[str, dict[str, str]] = {}
    for name, waveform in waveforms.items():
        source_vector = waveform.source_vector or name
        vector_metadata = p3_vector_metadata(analysis, source_vector, waveform.vector_kind)
        if vector_metadata:
            metadata[name] = vector_metadata
    return metadata


def _analysis_from_plotname(plotname: str | None) -> str | None:
    text = (plotname or "").strip().lower()
    if not text:
        return None
    padded = f" {text} "
    if " noise " in padded:
        return "noise"
    if "sensitivity" in text or text in {"sens", "sens analysis"}:
        return "sens"
    if " dc " in padded:
        return "dc"
    if " ac " in padded:
        return "ac"
    if "transient" in text or " tran " in padded or text in {"tran", "tran analysis"}:
        return "tran"
    if "operating point" in text or text in {"op", "op analysis"}:
        return "op"
    if "pole-zero" in text or "pole zero" in text or text in {"pz", "pz analysis"}:
        return "pz"
    if "distortion" in text or text in {"disto", "disto analysis"}:
        return "disto"
    if "transfer function" in text or text in {"tf", "tf analysis"}:
        return "tf"
    if "fourier" in text or text in {"four", "fourier"}:
        return "four"
    return None


def _analysis_without_abscissa(analysis: str | None) -> bool:
    return analysis in _ANALYSES_WITHOUT_ABSCISSA


__all__ = ["load_ngspice_rawfile"]
