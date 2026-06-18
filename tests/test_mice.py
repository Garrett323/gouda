import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pytest
from gouda import Mice, SimpleImputer
import time
import pandas as pd
# from hyperimpute.plugins.imputers import Imputers


def test_nans():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = Mice(backend="linear", max_iter=2).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
    imputed = Mice(backend="ridge", max_iter=2).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
    imputed = Mice(backend="pmm", max_iter=2).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


def test_iterations_work():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = Mice(backend="linear", max_iter=2).fit(data).transform(data)
    imputed2 = Mice(backend="linear", max_iter=2).fit(data).transform(data)
    imputed3 = Mice(backend="linear", max_iter=4).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    print("imputed2:\n", imputed2)
    diff = np.abs(imputed - imputed2)
    print("diff:\n", diff)
    print("max diff:\n", diff.max())
    print(f"stats:\nmean: {imputed.mean()} min: {
          imputed.min()} max: {imputed.max()}")
    assert np.allclose(imputed, imputed2, atol=1e-3), \
        "Unexpected difference between different iterations"
    print("imputed3:\n", imputed3)
    assert not np.allclose(
        imputed, imputed3), "No difference between iterations"


def test_not_simple():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = Mice(backend="linear", max_iter=2).fit(data).transform(data)
    imputed_simple = SimpleImputer().fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    print("imputed simple:\n", imputed_simple)
    assert not np.allclose(imputed, imputed_simple), "Still at initial values"


@pytest.mark.heavy
def test_time():
    data = np.random.rand(500, 12)
    data[data < 0.78] = np.nan
    max_iter = 10
    # hyper_mice = Imputers().get("mice")

    N = 3
# Warmup
    for _ in range(3):
        _ = Mice(max_iter=max_iter).fit(data).transform(data)
        _ = IterativeImputer(max_iter=max_iter).fit(data).transform(data)

# # Benchmark Hyper
#     times_h = []
#     for _ in range(N):
#         start = time.perf_counter_ns()
#         _ = hyper_mice.fit_transform(data)
#         times_h.append(time.perf_counter_ns() - start)

# Benchmark Rust
    times_rs = []
    for _ in range(N):
        imputer = Mice(max_iter=max_iter).fit(data)
        start = time.perf_counter_ns()
        _ = imputer.transform(data)
        times_rs.append(time.perf_counter_ns() - start)

# Benchmark sklearn
    times_sk = []
    for _ in range(N):
        imputer = IterativeImputer(max_iter=max_iter).fit(data)
        start = time.perf_counter_ns()
        _ = imputer.transform(data)
        times_sk.append(time.perf_counter_ns() - start)

    elapsed_rs = sorted(times_rs)[N // 2]  # median
    elapsed_sk = sorted(times_sk)[N // 2]
    # elapsed_h = sorted(times_h)[N // 2]

    assert 4 * elapsed_rs < elapsed_sk, f"Rust: {
        elapsed_rs}ns  sklearn: {elapsed_sk}ns"
    # assert 4 * elapsed_rs < elapsed_h, f"Rust: {
    #     elapsed_rs}ns  hyper impute: {elapsed_h}ns"


def test_categoricals():
    data = pd.DataFrame({
        "a": np.random.rand(200),
        "b": ["a" for _ in range(150)] + ["b" for _ in range(50)]
    })
    mask = np.random.rand(200, 2) > 0.5
    data[mask] = pd.NA
    imputed = Mice(max_iter=2, encoding='label').fit(data).transform(data)
    print("data", data.iloc[:20])
    print("imputed", imputed[:20])
    assert isinstance(imputed, pd.DataFrame), "No DataFrame returned"
    for val in imputed.iloc[mask[:, 1], 1]:
        assert val == "a"


def test_shape_mismatch():
    data = pd.read_csv("tests/resources/test.csv")
    print(data)
    mask = np.random.rand(*data.shape) > 0.5
    data[mask] = pd.NA
    imputed = Mice(max_iter=2, encoding='label').fit(data).transform(data)
    print(imputed)
