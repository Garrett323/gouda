import numpy as np
import pandas as pd
from missforest import MissForest as mfpy
import pytest
from gouda import MissForest, SimpleImputer
import time


def test_nans():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = MissForest().fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


def test_not_simple():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = MissForest().fit(data).transform(data)
    imputed_simple = SimpleImputer().fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    print("imputed simple:\n", imputed_simple)
    assert not np.allclose(imputed, imputed_simple), "Still at initial values"


@pytest.mark.heavy
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_time():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    data = pd.DataFrame(data)
    N = 3
# Warmup
    for _ in range(3):
        _ = MissForest().fit(data.values).transform(data.values)
        _ = mfpy().fit(data).transform(data)

# Benchmark Rust
    times_rs = []
    for _ in range(N):
        imputer = MissForest().fit(data.values)
        start = time.perf_counter_ns()
        _ = imputer.transform(data.values)
        times_rs.append(time.perf_counter_ns() - start)

# Benchmark sklearn
    times_sk = []
    for _ in range(N):
        imputer = mfpy().fit(data)
        start = time.perf_counter_ns()
        _ = imputer.transform(data)
        times_sk.append(time.perf_counter_ns() - start)

    elapsed_rs = sorted(times_rs)[N // 2]  # median
    elapsed_sk = sorted(times_sk)[N // 2]

    assert elapsed_rs < elapsed_sk * 0.8, f"Rust: {elapsed_rs}ns  Python: {elapsed_sk}ns"
