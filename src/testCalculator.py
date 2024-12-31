"""
This file contains the test cases for the square function
pytest src/testCalculator.py
"""

import pytest
from src.pyCS import square


def test_positive():
    assert square(7) == 49
    assert square(8) == 64


def test_negative():
    assert square(-7) == 49
    assert square(-8) == 64


def test_zero():
    assert square(0) == 0


def test_str():
    with pytest.raises(TypeError):
        square("cat")
