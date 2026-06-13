"""HDF5 simulation result export helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from monata._json import json_safe as _json_safe
from monata.corner import corner_from_payload, corner_to_payload
from monata.measure.result import MeasureSet
from monata.sim.export_payload import _summaries_from_payload
from monata.sim.results import AnalysisResult, Waveform
from monata.sim.results import SimResult

__all__ = [
    "export_sim_result_hdf5",
    "load_sim_result_hdf5",
]


def export_sim_result_hdf5(
    result: SimResult,
    path: str | Path,
    *,
    compression: str | None = "gzip",
    compression_opts: Any = 9,
) -> Path:
    """Export a ``SimResult`` to an HDF5 file when ``h5py`` is installed."""

    h5py = _import_h5py("export_sim_result_hdf5")

    output_path = Path(path)
    analysis = result.analysis_result
    dataset_options = _hdf5_dataset_options(compression, compression_opts)
    with h5py.File(output_path, "w") as file:
        file.attrs["format"] = "monata.sim.result"
        file.attrs["version"] = 1
        simulation = file.create_group("simulation")
        simulation.attrs["status"] = str(result.status)
        simulation.attrs["metadata_json"] = json.dumps(_json_safe(result.metadata), sort_keys=True)
        simulation.attrs["corner_json"] = json.dumps(corner_to_payload(result.corner), sort_keys=True)
        simulation.attrs["measures_json"] = json.dumps(_json_safe(result.measures.to_dict()), sort_keys=True)
        simulation.attrs["summaries_json"] = json.dumps(_json_safe(result.summaries), sort_keys=True)
        if result.error_message is not None:
            simulation.attrs["error_message"] = result.error_message
        if result.sweep_var is not None:
            _write_hdf5_array(file.create_group("abscissas"), "sweep", result.sweep_var, dataset_options)
        waveforms = file.create_group("waveforms")
        for name, values in result.waveforms.items():
            _write_hdf5_array(waveforms, str(name), values, dataset_options)
        if analysis is not None:
            _write_analysis_hdf5(file.create_group("analysis"), analysis, dataset_options)
    return output_path


def load_sim_result_hdf5(path: str | Path) -> SimResult:
    """Load a ``SimResult`` previously written by ``export_sim_result_hdf5``."""

    h5py = _import_h5py("load_sim_result_hdf5")

    with h5py.File(path, "r") as file:
        if _hdf5_attr_text(file.attrs, "format") != "monata.sim.result":
            raise ValueError("unsupported HDF5 result format")
        simulation = file["simulation"]
        return SimResult(
            status=_hdf5_attr_text(simulation.attrs, "status"),
            waveforms=_read_hdf5_arrays(file.get("waveforms")),
            sweep_var=_read_hdf5_sweep(file.get("abscissas")),
            corner=corner_from_payload(_json_hdf5_attr(simulation.attrs, "corner_json", None)),
            metadata=dict(_json_hdf5_attr(simulation.attrs, "metadata_json", {})),
            error_message=_optional_hdf5_attr_text(simulation.attrs, "error_message"),
            analysis_result=_read_analysis_hdf5(file.get("analysis")),
            measures=MeasureSet(_json_hdf5_attr(simulation.attrs, "measures_json", {})),
            summaries=_summaries_from_payload(_json_hdf5_attr(simulation.attrs, "summaries_json", {})),
        )


def _import_h5py(caller: str) -> Any:
    try:
        return import_module("h5py")
    except ImportError as exc:
        raise RuntimeError(f"{caller} requires the optional 'hdf5' extra") from exc


def _write_analysis_hdf5(group: Any, analysis: AnalysisResult, dataset_options: Mapping[str, Any]) -> None:
    group.attrs["analysis"] = analysis.analysis or ""
    group.attrs["source"] = analysis.source or ""
    group.attrs["metadata_json"] = json.dumps(_json_safe(analysis.metadata), sort_keys=True)
    analysis_abscissa_dataset = None
    if analysis.abscissa is not None:
        analysis_abscissa_dataset = _write_hdf5_waveform(
            group.create_group("abscissas"),
            analysis.abscissa.name,
            analysis.abscissa,
            dataset_options,
        )
    waveforms = group.create_group("waveforms")
    waveform_abscissas = group.create_group("waveform_abscissas")
    waveform_datasets: dict[str, Any] = {}
    for name, waveform in analysis.waveforms.items():
        dataset = _write_hdf5_waveform(waveforms, name, waveform, dataset_options)
        waveform_datasets[name] = dataset
        if _waveform_uses_analysis_abscissa(waveform, analysis.abscissa):
            if analysis_abscissa_dataset is not None:
                _write_hdf5_abscissa_reference(dataset, analysis_abscissa_dataset, source="analysis")
        elif waveform.abscissa_data is not None:
            abscissa_dataset = _write_hdf5_array(waveform_abscissas, name, waveform.abscissa_data, dataset_options)
            _write_hdf5_abscissa_reference(dataset, abscissa_dataset, source="waveform")
    _write_analysis_entity_links(group, analysis, waveform_datasets)


def _write_hdf5_abscissa_reference(dataset: Any, abscissa_dataset: Any, *, source: str) -> None:
    dataset.attrs["abscissa_source"] = source
    dataset.attrs["abscissa_dataset"] = abscissa_dataset.name
    dataset.attrs["abscissa_ref"] = abscissa_dataset.ref


def _waveform_uses_analysis_abscissa(waveform: Waveform, abscissa: Waveform | None) -> bool:
    if abscissa is None or waveform.abscissa_data is None:
        return False
    if waveform.abscissa_name != abscissa.name:
        return False
    return bool(np.array_equal(np.asarray(waveform.abscissa_data), np.asarray(abscissa.data), equal_nan=True))


def _write_hdf5_waveform(group: Any, name: str, waveform: Waveform, dataset_options: Mapping[str, Any]) -> Any:
    dataset = _write_hdf5_array(group, name, waveform.data, dataset_options)
    dataset.attrs["name"] = waveform.name
    dataset.attrs["unit"] = "" if waveform.unit is None else str(waveform.unit)
    dataset.attrs["quantity"] = waveform.quantity or ""
    dataset.attrs["title"] = waveform.title or ""
    dataset.attrs["metadata_json"] = json.dumps(_json_safe(waveform.metadata), sort_keys=True)
    dataset.attrs["display_name"] = waveform.display_name or ""
    dataset.attrs["normalized_name"] = waveform.normalized_name or ""
    dataset.attrs["raw_vector_name"] = waveform.raw_vector_name or ""
    dataset.attrs["vector_kind"] = waveform.vector_kind
    dataset.attrs["source_vector"] = waveform.source_vector or ""
    dataset.attrs["abscissa_name"] = waveform.abscissa_name or ""
    dataset.attrs["analysis"] = waveform.analysis or ""
    dataset.attrs["source"] = waveform.source or ""
    dataset.attrs["extraction"] = waveform.extraction or ""
    dataset.attrs["plot_name"] = waveform.plot_name or ""
    return dataset


def _write_analysis_entity_links(group: Any, analysis: AnalysisResult, waveform_datasets: Mapping[str, Any]) -> None:
    datasets_by_waveform = {
        id(waveform): waveform_datasets[name]
        for name, waveform in analysis.waveforms.items()
        if name in waveform_datasets
    }
    entities = group.create_group("entities")
    _link_hdf5_waveforms(entities.create_group("node_voltages"), analysis.node_voltages_by_node, datasets_by_waveform)
    _link_hdf5_waveforms(
        entities.create_group("branch_currents"),
        analysis.branch_currents_by_element,
        datasets_by_waveform,
    )
    device_parameters = entities.create_group("device_parameters")
    for element, parameters in analysis.device_parameters_by_element.items():
        element_group = device_parameters.create_group(_unique_hdf5_name(device_parameters, element))
        element_group.attrs["source_name"] = element
        _link_hdf5_waveforms(element_group, parameters, datasets_by_waveform)
    _link_hdf5_waveforms(
        entities.create_group("internal_parameters"),
        analysis.internal_parameters,
        datasets_by_waveform,
    )


def _link_hdf5_waveforms(group: Any, waveforms: Mapping[str, Waveform], datasets_by_waveform: Mapping[int, Any]) -> None:
    for name, waveform in waveforms.items():
        dataset = datasets_by_waveform.get(id(waveform))
        if dataset is not None:
            group[_unique_hdf5_name(group, name)] = dataset


def _read_analysis_hdf5(group: Any | None) -> AnalysisResult | None:
    if group is None:
        return None
    return AnalysisResult(
        analysis=_optional_hdf5_attr_text(group.attrs, "analysis"),
        waveforms=_read_hdf5_waveforms(
            group.get("waveforms"),
            abscissa_data=_read_hdf5_arrays(group.get("waveform_abscissas")),
        ),
        abscissa=_read_hdf5_first_waveform(group.get("abscissas")),
        metadata=dict(_json_hdf5_attr(group.attrs, "metadata_json", {})),
        source=_optional_hdf5_attr_text(group.attrs, "source"),
    )


def _read_hdf5_sweep(group: Any | None) -> np.ndarray | None:
    if group is None or len(group) == 0:
        return None
    dataset = group["sweep"] if "sweep" in group else next(iter(group.values()))
    return _read_hdf5_array(dataset)


def _read_hdf5_arrays(group: Any | None) -> dict[str, np.ndarray]:
    if group is None:
        return {}
    return {_hdf5_source_name(dataset): _read_hdf5_array(dataset) for dataset in group.values()}


def _read_hdf5_waveforms(
    group: Any | None,
    *,
    abscissa_data: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Waveform]:
    if group is None:
        return {}
    return {
        _hdf5_source_name(dataset): _read_hdf5_waveform(
            dataset,
            abscissa_data=_read_hdf5_waveform_abscissa_data(
                dataset,
                (abscissa_data or {}).get(_hdf5_source_name(dataset)),
            ),
        )
        for dataset in group.values()
    }


def _read_hdf5_waveform_abscissa_data(dataset: Any, abscissa_data: np.ndarray | None) -> np.ndarray | None:
    if abscissa_data is not None:
        return abscissa_data
    reference = dataset.attrs.get("abscissa_ref")
    if reference is None:
        return None
    try:
        if not reference:
            return None
        return _read_hdf5_array(dataset.file[reference])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid HDF5 waveform abscissa reference: {dataset.name}") from exc


def _read_hdf5_first_waveform(group: Any | None) -> Waveform | None:
    if group is None or len(group) == 0:
        return None
    return _read_hdf5_waveform(next(iter(group.values())))


def _read_hdf5_waveform(dataset: Any, *, abscissa_data: np.ndarray | None = None) -> Waveform:
    attrs = dataset.attrs
    return Waveform(
        name=_hdf5_attr_text(attrs, "name", _hdf5_source_name(dataset)),
        data=_read_hdf5_array(dataset),
        unit=_optional_hdf5_attr_text(attrs, "unit"),
        quantity=_optional_hdf5_attr_text(attrs, "quantity"),
        title=_optional_hdf5_attr_text(attrs, "title"),
        abscissa_data=abscissa_data,
        metadata=dict(_json_hdf5_attr(attrs, "metadata_json", {})),
        display_name=_optional_hdf5_attr_text(attrs, "display_name"),
        normalized_name=_optional_hdf5_attr_text(attrs, "normalized_name"),
        raw_vector_name=_optional_hdf5_attr_text(attrs, "raw_vector_name"),
        vector_kind=_hdf5_attr_text(attrs, "vector_kind", "unknown"),
        source_vector=_optional_hdf5_attr_text(attrs, "source_vector"),
        abscissa=_optional_hdf5_attr_text(attrs, "abscissa_name"),
        analysis=_optional_hdf5_attr_text(attrs, "analysis"),
        source=_optional_hdf5_attr_text(attrs, "source"),
        extraction=_optional_hdf5_attr_text(attrs, "extraction"),
        plot_name=_optional_hdf5_attr_text(attrs, "plot_name"),
    )


def _read_hdf5_array(dataset: Any) -> np.ndarray:
    data = np.asarray(dataset[()])
    dtype = _optional_hdf5_attr_text(dataset.attrs, "dtype")
    shape_json = _optional_hdf5_attr_text(dataset.attrs, "shape_json")
    if dtype is not None:
        data = data.astype(np.dtype(dtype), copy=False)
    if shape_json is not None:
        data = data.reshape(tuple(json.loads(shape_json)))
    return data


def _write_hdf5_array(group: Any, name: str, values: Any, dataset_options: Mapping[str, Any]) -> Any:
    array = np.asarray(values)
    dataset = group.create_dataset(_unique_hdf5_name(group, name), data=array, **dict(dataset_options))
    dataset.attrs["source_name"] = str(name)
    dataset.attrs["dtype"] = str(array.dtype)
    dataset.attrs["shape_json"] = json.dumps(list(array.shape))
    return dataset


def _hdf5_dataset_options(compression: str | None, compression_opts: Any) -> dict[str, Any]:
    if compression is None:
        return {}
    options = {"compression": compression}
    if compression_opts is not None:
        options["compression_opts"] = compression_opts
    return options


def _unique_hdf5_name(group: Any, name: str) -> str:
    base_name = _hdf5_name(name)
    if base_name not in group:
        return base_name
    index = 1
    while f"{base_name}__{index}" in group:
        index += 1
    return f"{base_name}__{index}"


def _hdf5_name(name: str) -> str:
    return str(name).replace("/", "__")


def _hdf5_source_name(dataset: Any) -> str:
    return _hdf5_attr_text(dataset.attrs, "source_name", dataset.name.rsplit("/", 1)[-1])


def _json_hdf5_attr(attrs: Any, name: str, default: Any) -> Any:
    text = _optional_hdf5_attr_text(attrs, name)
    if text is None:
        return default
    return json.loads(text)


def _optional_hdf5_attr_text(attrs: Any, name: str) -> str | None:
    value = _hdf5_attr_text(attrs, name, "")
    return value or None


def _hdf5_attr_text(attrs: Any, name: str, default: str = "") -> str:
    value = attrs.get(name, default)
    if isinstance(value, bytes):
        return value.decode()
    return str(value)
