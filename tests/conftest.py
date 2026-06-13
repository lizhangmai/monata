import os
import sys

import pytest

# Add src/ to path so tests can import monata without pip install -e .
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

# Add tests/ to path so shared test support modules can be imported as support.*.
tests_root = os.path.dirname(__file__)
sys.path.insert(0, tests_root)


@pytest.fixture(autouse=True)
def ngspice_on_path(monkeypatch):
    from support.ngspice import put_ngspice_on_path

    put_ngspice_on_path(monkeypatch, __file__)


@pytest.fixture
def require_ngspice():
    from support.ngspice import skip_if_no_ngspice

    skip_if_no_ngspice()
