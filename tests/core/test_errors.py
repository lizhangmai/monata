

from monata.errors import (
    LibraryNotFoundError,
    CellNotFoundError,
    ViewNotFoundError,
    ViewNotGeneratedError,
    ViewAlreadyModifiedError,
)


def test_library_not_found_error():
    err = LibraryNotFoundError("mylib")
    assert "mylib" in str(err)
    assert isinstance(err, KeyError)


def test_cell_not_found_error():
    err = CellNotFoundError("inverter", "mylib")
    assert "inverter" in str(err)
    assert "mylib" in str(err)
    assert isinstance(err, KeyError)


def test_view_not_found_error():
    err = ViewNotFoundError("schematic", "inverter")
    assert "schematic" in str(err)
    assert "inverter" in str(err)
    assert isinstance(err, KeyError)


def test_view_not_generated_error():
    err = ViewNotGeneratedError("netlist", "inverter")
    assert "netlist" in str(err)
    assert "generate" in str(err).lower()
    assert isinstance(err, FileNotFoundError)


def test_view_already_modified_error():
    err = ViewAlreadyModifiedError("netlist", "inverter")
    assert "netlist" in str(err)
    assert "force" in str(err).lower()
    assert isinstance(err, RuntimeError)
