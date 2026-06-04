import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
import pytest
from gouda import Mice, SimpleImputer
import time

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
    imputed2 = Mice(backend="linear",max_iter=2).fit(data).transform(data)
    imputed3 = Mice(backend="linear",max_iter=4).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert np.allclose(imputed, imputed2), "Unexpected difference between different iterations"
    print("imputed3:\n", imputed3)
    assert not np.allclose(imputed, imputed3), "No difference between iterations"

def test_not_simple():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = Mice(backend="linear",max_iter=2).fit(data).transform(data)
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

    N = 3
# Warmup
    for _ in range(3):
        _ = Mice(max_iter=max_iter).fit(data).transform(data)
        _ = IterativeImputer(max_iter=max_iter).fit(data).transform(data)

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

    assert elapsed_rs < elapsed_sk * 0.8, f"Rust: {
        elapsed_rs}ns  sklearn: {elapsed_sk}ns"
