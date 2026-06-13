"""Parsers for ngspice stdout text emitted by Monata control blocks."""

from __future__ import annotations

import re
from typing import Mapping

import numpy as np

from monata.measure import MeasureResult, MeasureSet


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_MEASURE_RE = re.compile(
    rf"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_.$:-]*)\s*=\s*(?P<value>{_FLOAT_RE}|failed)\b(?P<tail>.*)$"
)
_FOURIER_OUTPUT_RE = re.compile(r"^\s*Fourier\s+analysis\s+for\s+(?P<vector>.+?):\s*$", re.IGNORECASE)
_FOURIER_SUMMARY_RE = re.compile(
    rf"No\.\s*Harmonics:\s*(?P<harmonics>\d+)\s*,\s*"
    rf"THD:\s*(?P<thd>{_NUMBER_PATTERN})\s*%\s*,\s*"
    r"Gridsize:\s*(?P<grid_size>\d+)\s*,\s*"
    r"Interpolation Degree:\s*(?P<interpolation_degree>\d+)\s*,\s*"
    r"No\.\s*Periods:\s*(?P<periods>\d+)",
    re.IGNORECASE,
)


def parse_measure_print(stdout: str, expected: Mapping[str, str]) -> MeasureSet:
    """Parse ngspice .measure scalar output for expected measure names."""

    found: dict[str, MeasureResult] = {}
    expected_lower = {name.lower(): (name, analysis) for name, analysis in expected.items()}
    for line in stdout.splitlines():
        match = _MEASURE_RE.match(line)
        if match is None:
            continue
        parsed_name = match.group("name")
        key = parsed_name.lower()
        if key not in expected_lower:
            continue
        name, analysis = expected_lower[key]
        raw_value = match.group("value")
        raw_line = line.strip()
        if raw_value.lower() == "failed":
            found[name] = MeasureResult(
                name=name,
                value=None,
                analysis=analysis,
                reason="measure_failed",
                raw=raw_line,
            )
        else:
            found[name] = MeasureResult(
                name=name,
                value=float(raw_value),
                analysis=analysis,
                raw=raw_line,
            )
    for name, analysis in expected.items():
        if name not in found:
            found[name] = MeasureResult(
                name=name,
                value=None,
                analysis=analysis,
                reason="measure_missing",
                raw=None,
            )
    return MeasureSet(found)


def parse_op_print(text: str, output_names: list[str]) -> dict[str, np.ndarray]:
    """Parse ngspice `print v(node)` scalar operating-point output."""

    observed = {
        match.group("vector").lower(): float(match.group("value"))
        for match in re.finditer(
            rf"^\s*(?P<vector>v\([^)]+\))\s*=\s*(?P<value>{_FLOAT_RE})",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
    }
    waveforms: dict[str, np.ndarray] = {}
    for name in output_names:
        value = observed.get(f"v({name})".lower())
        if value is None:
            raise ValueError(f"ngspice operating-point output missing v({name})")
        waveforms[name] = np.array([value])
    return waveforms


def parse_noise_print(text: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Parse ngspice stdout from `print` commands for noise1/noise2 plots."""

    rows: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        try:
            rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    if not rows:
        raise ValueError("ngspice noise output did not contain spectrum rows")

    totals: dict[str, float] = {}
    for name in ("onoise_total", "inoise_total"):
        match = re.search(rf"{name}\s*=\s*({_FLOAT_RE})", text, re.IGNORECASE)
        if match is None:
            raise ValueError(f"ngspice noise output missing {name}")
        totals[name] = float(match.group(1))

    data = np.array(rows)
    return data[:, 0], {
        "onoise_spectrum": data[:, 1],
        "inoise_spectrum": data[:, 2],
    }, totals


def noise_aborted(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    return (
        "noise simulation(s) aborted" in combined
        or "ac input not found" in combined
        or "no such vector" in combined
    )


def parse_tf_stdout(stdout: str) -> dict[str, np.ndarray]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        label, value = line.rsplit("=", 1)
        key = _tf_output_key(label)
        if key is None:
            continue
        try:
            values[key] = float(value.strip().split()[0])
        except (IndexError, ValueError):
            continue
    required = {"transfer_function", "input_resistance", "output_resistance"}
    if not required.issubset(values):
        raise ValueError("ngspice tf output did not contain transfer/input/output resistance values")
    return {
        "transfer_function": np.array([values["transfer_function"]]),
        "input_resistance": np.array([values["input_resistance"]]),
        "output_resistance": np.array([values["output_resistance"]]),
    }


def parse_fourier_stdout(stdout: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    harmonics: list[float] = []
    frequencies: list[float] = []
    magnitudes: list[float] = []
    phases: list[float] = []
    normalized_magnitudes: list[float] = []
    normalized_phases: list[float] = []
    metadata: dict[str, object] = {}
    for line in stdout.splitlines():
        metadata.update(_parse_fourier_metadata_line(line))
        columns = line.split()
        if len(columns) < 4:
            continue
        try:
            harmonic = float(columns[0])
            frequency = float(columns[1])
            magnitude = float(columns[2])
            phase = float(columns[3])
        except ValueError:
            continue
        harmonics.append(harmonic)
        frequencies.append(frequency)
        magnitudes.append(magnitude)
        phases.append(phase)
        if len(columns) >= 6:
            try:
                normalized_magnitudes.append(float(columns[4]))
                normalized_phases.append(float(columns[5]))
            except ValueError:
                pass
    if not harmonics:
        raise ValueError("ngspice fourier output did not contain a harmonic table")
    metadata.setdefault("fourier_harmonic_count", len(harmonics))
    waveforms = {
        "harmonic": np.asarray(harmonics),
        "frequency": np.asarray(frequencies),
        "fourier_magnitude": np.asarray(magnitudes),
        "fourier_phase": np.asarray(phases),
    }
    if len(normalized_magnitudes) == len(harmonics) and len(normalized_phases) == len(harmonics):
        waveforms["fourier_normalized_magnitude"] = np.asarray(normalized_magnitudes)
        waveforms["fourier_normalized_phase"] = np.asarray(normalized_phases)
    return waveforms, metadata


def _tf_output_key(label: str) -> str | None:
    key = _safe_label(label)
    if key == "transfer_function":
        return "transfer_function"
    if key == "output_impedance" or key.startswith("output_impedance_at_"):
        return "output_resistance"
    if key.endswith("input_impedance") or key == "input_impedance":
        return "input_resistance"
    return None


def _safe_label(label: str) -> str:
    return re.sub(r"\W+", "_", label.strip().lower()).strip("_")


def _parse_fourier_metadata_line(line: str) -> dict[str, object]:
    output_match = _FOURIER_OUTPUT_RE.match(line)
    if output_match is not None:
        return {"fourier_output_vector": output_match.group("vector").strip()}

    summary_match = _FOURIER_SUMMARY_RE.search(line)
    if summary_match is None:
        return {}
    return {
        "fourier_harmonic_count": int(summary_match.group("harmonics")),
        "fourier_thd_percent": float(summary_match.group("thd")),
        "fourier_grid_size": int(summary_match.group("grid_size")),
        "fourier_interpolation_degree": int(summary_match.group("interpolation_degree")),
        "fourier_period_count": int(summary_match.group("periods")),
    }


__all__ = [
    "noise_aborted",
    "parse_fourier_stdout",
    "parse_measure_print",
    "parse_noise_print",
    "parse_op_print",
    "parse_tf_stdout",
]
