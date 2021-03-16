import numpy as np

from pandas import (
    DataFrame,
    Series,
    Timestamp,
)
import pandas._testing as tm


def test_group_shift_with_null_key():
    # This test is designed to replicate the segfault in issue #13813.
    n_rows = 1200

    # Generate a moderately large dataframe with occasional missing
    # values in column `B`, and then group by [`A`, `B`]. This should
    # force `-1` in `labels` array of `g.grouper.group_info` exactly
    # at those places, where the group-by key is partially missing.
    df = DataFrame(
        [(i % 12, i % 3 if i % 3 else np.nan, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    expected = DataFrame(
        [(i + 12 if i % 3 and i < n_rows - 12 else np.nan) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    result = g.shift(-1)

    tm.assert_frame_equal(result, expected)


def test_group_shift_with_fill_value():
    # GH #24128
    n_rows = 24
    df = DataFrame(
        [(i % 12, i % 3, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    expected = DataFrame(
        [(i + 12 if i < n_rows - 12 else 0) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    result = g.shift(-1, fill_value=0)[["Z"]]

    tm.assert_frame_equal(result, expected)


def test_group_shift_lose_timezone():
    # GH 30134
    now_dt = Timestamp.utcnow()
    df = DataFrame({"a": [1, 1], "date": now_dt})
    result = df.groupby("a").shift(0).iloc[0]
    expected = Series({"date": now_dt}, name=result.name)
    tm.assert_series_equal(result, expected)


def test_group_shift_lose_index_1():
    # GH 13519 -- test with as_index=False
    df = DataFrame({'K': [1, 1, 1, 2, 2, 3], 'V': [1, 2, 3, 4, 5, 6]})
    g = df.groupby('K', as_index=False)

    expected = DataFrame({'K': [1, 1, 1, 2, 2, 3], 'V': [np.nan, 1, 2, np.nan, 4, np.nan]})
    result = g.shift(1)

    tm.assert_frame_equal(expected, result)


def test_group_shift_lose_index_2():
    # GH 13519 -- test with as_index=True
    df = DataFrame({'K': [1, 1, 1, 2, 2, 3], 'V': [1, 2, 3, 4, 5, 6]})
    g = df.groupby('K', as_index=True)

    expected = DataFrame({'K': [1, 1, 1, 2, 2, 3], 'V': [np.nan, 1, 2, np.nan, 4, np.nan]}).set_index('K')
    result = g.shift(1)

    tm.assert_frame_equal(expected, result)
