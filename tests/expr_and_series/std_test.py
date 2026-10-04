from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
data_with_nulls = {"a": [1, 3, 2, None], "b": [4, 4, 6, None], "z": [7.0, 8.0, 9.0, None]}

expected_results = {
    "a_ddof_1": [1.0],
    "a_ddof_0": [0.816497],
    "b_ddof_2": [1.632993],
    "z_ddof_0": [0.816497],
}


@pytest.mark.parametrize("input_data", [data, data_with_nulls])
def test_std(constructor: Constructor, input_data: dict[str, list[float | None]]) -> None:
    df = nw.from_native(constructor(input_data))
    result = df.select(
        nw.col("a").std(ddof=1).alias("a_ddof_1"),
        nw.col("a").std(ddof=0).alias("a_ddof_0"),
        nw.col("z").std(ddof=0).alias("z_ddof_0"),
    )
    expected_results = {"a_ddof_1": [1.0], "a_ddof_0": [0.816497], "z_ddof_0": [0.816497]}
    assert_equal_data(result, expected_results)

    result = df.select(nw.col("b").std(ddof=2).alias("b_ddof_2"))
    expected_results = {"b_ddof_2": [1.632993]}
    assert_equal_data(result, expected_results)


@pytest.mark.parametrize("input_data", [data, data_with_nulls])
def test_std_series(
    constructor_eager: ConstructorEager, input_data: dict[str, list[float | None]]
) -> None:
    df = nw.from_native(constructor_eager(input_data), eager_only=True)
    result = {
        "a_ddof_1": [df["a"].std(ddof=1)],
        "a_ddof_0": [df["a"].std(ddof=0)],
        "b_ddof_2": [df["b"].std(ddof=2)],
        "z_ddof_0": [df["z"].std(ddof=0)],
    }
    assert_equal_data(result, expected_results)


def test_std_ddof_gt_1_non_integer_scale(constructor: Constructor) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/4020
    # `(n - 1) / (n - ddof)` is not an integer here, which catches backends
    # that truncate the scaling factor instead of dividing (e.g. DuckDB's
    # integer `divide`).
    df = nw.from_native(constructor({"a": [1.0, 2.0, 3.0, 4.0]}))
    result = df.select(nw.col("a").std(ddof=2).alias("s"))
    assert_equal_data(result, {"s": [1.5811388300841898]})
