import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


def test_group_diff_real(any_real_dtype):
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [1, 2, 3, 4, 5]}, dtype=any_real_dtype)
    result = df.groupby("a")["b"].diff()
    exp_dtype = "float"
    if any_real_dtype in ["int8", "int16", "float32"]:
        exp_dtype = "float32"
    expected = Series([np.nan, np.nan, np.nan, 1.0, 3.0], dtype=exp_dtype, name="b")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-02"),
            Timestamp("2013-01-03"),
        ],
        [Timedelta("5 days"), Timedelta("6 days"), Timedelta("7 days")],
    ],
)
def test_group_diff_datetimelike(data):
    df = DataFrame({"a": [1, 2, 2], "b": data})
    result = df.groupby("a")["b"].diff()
    expected = Series([NaT, NaT, Timedelta("1 days")], name="b")
    tm.assert_series_equal(result, expected)


def test_group_diff_bool():
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [True, True, False, False, True]})
    result = df.groupby("a")["b"].diff()
    expected = Series([np.nan, np.nan, np.nan, False, False], name="b")
    tm.assert_series_equal(result, expected)


def test_group_diff_object_raises(object_dtype):
    df = DataFrame(
        {"a": ["foo", "bar", "bar"], "b": ["baz", "foo", "foo"]}, dtype=object_dtype
    )
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for -"):
        df.groupby("a")["b"].diff()
