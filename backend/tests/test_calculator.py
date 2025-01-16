import pytest
from app.services.calculator import Calculator
from app.core.exceptions import CalculatorError


def test_basic_addition():
    assert Calculator.evaluate("2 + 2") == 4
    assert Calculator.evaluate("0 + 0") == 0
    assert Calculator.evaluate("10 + 20") == 30


def test_basic_subtraction():
    assert Calculator.evaluate("5 - 3") == 2
    assert Calculator.evaluate("0 - 0") == 0
    assert Calculator.evaluate("10 - 20") == -10


def test_basic_multiplication():
    assert Calculator.evaluate("3 * 4") == 12
    assert Calculator.evaluate("0 * 5") == 0
    assert Calculator.evaluate("10 * -2") == -20


def test_basic_division():
    assert Calculator.evaluate("8 / 2") == 4
    assert Calculator.evaluate("10 / 2") == 5
    assert Calculator.evaluate("-6 / 2") == -3


def test_floating_point_operations():
    assert Calculator.evaluate("3.5 + 2.1") == 5.6
    assert Calculator.evaluate("10.5 / 2") == 5.25
    assert abs(Calculator.evaluate("1 / 3") - 0.3333333333333333) < 1e-10


def test_unary_minus():
    assert Calculator.evaluate("-5") == -5
    assert Calculator.evaluate("-(2 + 3)") == -5
    assert Calculator.evaluate("-(-5)") == 5


def test_complex_expressions():
    assert Calculator.evaluate("2 + 3 * 4") == 14
    assert Calculator.evaluate("(2 + 3) * 4") == 20
    assert Calculator.evaluate("10 / 2 + 3") == 8


def test_division_by_zero():
    with pytest.raises(CalculatorError):
        Calculator.evaluate("1 / 0")


def test_invalid_expressions():
    with pytest.raises(CalculatorError):
        Calculator.evaluate("2 + ")

    with pytest.raises(CalculatorError):
        Calculator.evaluate("2 2")

    with pytest.raises(CalculatorError):
        Calculator.evaluate("")


def test_unsupported_operations():
    with pytest.raises(CalculatorError):
        Calculator.evaluate("2 ** 3")  # Power operation not supported

    with pytest.raises(CalculatorError):
        Calculator.evaluate("2 % 3")  # Modulo operation not supported


def test_non_mathematical_expressions():
    with pytest.raises(CalculatorError):
        Calculator.evaluate("print(2)")

    with pytest.raises(CalculatorError):
        Calculator.evaluate("'2' + '2'")

    with pytest.raises(CalculatorError):
        Calculator.evaluate("import os")
