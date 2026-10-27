
import pytest

from app.main import load_positive_int, load_positive_float


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_positive_int_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("TEST_CONFIG_VALUE", value)

    with pytest.raises(RuntimeError):
        load_positive_int("TEST_CONFIG_VALUE", "12000")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "abc"])
def test_positive_float_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("TEST_CONFIG_VALUE", value)

    with pytest.raises(RuntimeError):
        load_positive_float("TEST_CONFIG_VALUE", "30")


def test_positive_int_accepts_valid_value(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_VALUE", "5000")

    assert load_positive_int("TEST_CONFIG_VALUE", "12000") == 5000


def test_positive_float_accepts_valid_value(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_VALUE", "15.5")

    assert load_positive_float("TEST_CONFIG_VALUE", "30") == 15.5
