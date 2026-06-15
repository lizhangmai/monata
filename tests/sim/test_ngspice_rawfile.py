import numpy as np
import pytest

from monata.sim.backends.ngspice_output import parse_rawfile


def test_parse_ascii_rawfile_real_vectors(tmp_path):
    rawfile = tmp_path / "real.raw"
    rawfile.write_text(
        """Title: real
Date: today
Plotname: DC transfer characteristic
Flags: real
No. Variables: 2
No. Points: 3
Variables:
    0   v-sweep voltage
    1   v(in)   voltage
Values:
    0   0.0
        0.0
    1   0.5
        0.5
    2   1.0
        1.0
"""
    )

    result = parse_rawfile(rawfile)

    assert result.title == "real"
    assert result.plotname == "DC transfer characteristic"
    assert result.flags == ("real",)
    assert result.scale.name == "v-sweep"
    np.testing.assert_allclose(result.scale.data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(result.vector("v(in)").data, np.array([0.0, 0.5, 1.0]))
    assert result.metadata["extraction"] == "rawfile"
    assert result.metadata["num_variables"] == 2


def test_parse_rawfile_keeps_header_metadata_and_data_columns(tmp_path):
    rawfile = tmp_path / "metadata.raw"
    rawfile.write_text(
        """Note: rawfile is version 14
Circuit: MixedCaseCircuit
Doing analysis at TEMP = 25.000000 and TNOM = 27.000000
Warning: source stepping failed
Title: transient
Date: today
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 2
Variables:
No. of Data Columns : 2
    0   time    time
    1   v(out)  voltage
Values:
    0   0.0
        1.0
    1   1e-9
        2.0
"""
    )

    result = parse_rawfile(rawfile)

    assert result.metadata["circuit"] == "MixedCaseCircuit"
    assert result.metadata["date"] == "today"
    assert result.metadata["warnings"] == ["source stepping failed"]
    assert result.metadata["temperature"] == pytest.approx(25.0)
    assert result.metadata["nominal_temperature"] == pytest.approx(27.0)
    assert result.vector("v(out)").kind == "voltage"
    np.testing.assert_allclose(result.vector("v(out)").data, np.array([1.0, 2.0]))


def test_parse_ascii_rawfile_complex_vectors(tmp_path):
    rawfile = tmp_path / "complex.raw"
    rawfile.write_text(
        """Title: complex
Plotname: AC Analysis
Flags: complex
No. Variables: 2
No. Points: 2
Variables:
    0   frequency frequency
    1   v(out) voltage
Values:
    0   1.0,0.0
        2.0,3.0
    1   10.0,0.0
        4.0,5.0
"""
    )

    result = parse_rawfile(rawfile)

    assert result.flags == ("complex",)
    assert np.iscomplexobj(result.vector("v(out)").data)
    np.testing.assert_allclose(result.scale.data, np.array([1.0 + 0j, 10.0 + 0j]))
    np.testing.assert_allclose(result.vector("v(out)").data, np.array([2.0 + 3.0j, 4.0 + 5.0j]))
    assert result.metadata["rawfile_format"] == "ascii"


def test_parse_binary_rawfile_real_vectors(tmp_path):
    rawfile = tmp_path / "binary-real.raw"
    payload = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ],
        dtype="<f8",
    ).tobytes()
    rawfile.write_bytes(
        b"""Title: binary real
Plotname: DC transfer characteristic
Flags: real
No. Variables: 2
No. Points: 3
Variables:
    0   v-sweep voltage
    1   v(in)   voltage
Binary:
"""
        + payload
    )

    result = parse_rawfile(rawfile)

    assert result.metadata["rawfile_format"] == "binary"
    assert result.scale.name == "v-sweep"
    np.testing.assert_allclose(result.scale.data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(result.vector("v(in)").data, np.array([0.0, 0.5, 1.0]))


def test_parse_binary_rawfile_complex_vectors(tmp_path):
    rawfile = tmp_path / "binary-complex.raw"
    payload = np.array(
        [
            [[1.0, 0.0], [2.0, 3.0]],
            [[10.0, 0.0], [4.0, 5.0]],
        ],
        dtype="<f8",
    ).tobytes()
    rawfile.write_bytes(
        b"""Title: binary complex
Plotname: AC Analysis
Flags: complex
No. Variables: 2
No. Points: 2
Variables:
    0   frequency frequency
    1   v(out) voltage
Binary:
"""
        + payload
    )

    result = parse_rawfile(rawfile)

    assert result.metadata["rawfile_format"] == "binary"
    assert np.iscomplexobj(result.vector("v(out)").data)
    np.testing.assert_allclose(result.scale.data, np.array([1.0 + 0j, 10.0 + 0j]))
    np.testing.assert_allclose(result.vector("v(out)").data, np.array([2.0 + 3.0j, 4.0 + 5.0j]))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            """Title: bad
Flags: real
No. Variables: 1
No. Points: 1
Variables:
    0 time time
""",
            "missing Values",
        ),
        (
            """Title: bad
Flags: real
No. Variables: 2
No. Points: 1
Variables:
    0 time time
Values:
    0 0.0
""",
            "vector count mismatch",
        ),
        (
            """Title: bad
Flags: real
No. Variables: 1
No. Points: 1
Variables:
    0 time time
Values:
    0 not-a-number
""",
            "malformed real",
        ),
        (
            """Title: bad
Flags: complex
No. Variables: 1
No. Points: 1
Variables:
    0 frequency frequency
Values:
    0 1.0
""",
            "malformed complex",
        ),
        (
            """Title: bad
Flags: real
No. Variables: 2
No. Points: 1
Variables:
    0 time time
    1 v(out) voltage
Values:
    0 0.0
""",
            "value width mismatch",
        ),
    ],
)
def test_parse_ascii_rawfile_rejects_malformed_input(tmp_path, content, message):
    rawfile = tmp_path / "bad.raw"
    rawfile.write_text(content)

    with pytest.raises(ValueError, match=message):
        parse_rawfile(rawfile)


def test_parse_binary_rawfile_rejects_truncated_payload(tmp_path):
    rawfile = tmp_path / "binary.raw"
    rawfile.write_bytes(
        b"""Title: binary
Flags: real
No. Variables: 1
No. Points: 2
Variables:
    0 time time
Binary:
\x00\x01"""
    )

    with pytest.raises(ValueError, match="payload too short"):
        parse_rawfile(rawfile)
