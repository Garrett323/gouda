import numpy as np
import time
from sklearn.impute import KNNImputer
from gouda import KnnImputer


def test_time():
    data = np.random.rand(500, 50)
    data[data < 0.78] = np.nan

    N = 5
# Warmup
    for _ in range(3):
        KnnImputer().fit(data).transform(data)
        KNNImputer().fit(data).transform(data)

# Benchmark Rust
    times_rs = []
    for _ in range(N):
        imputer = KnnImputer()
        imputer.fit(data)
        start = time.perf_counter_ns()
        _ = imputer.transform(data)
        times_rs.append(time.perf_counter_ns() - start)

# Benchmark sklearn
    times_sk = []
    for _ in range(N):
        imputer = KNNImputer()
        imputer.fit(data)
        start = time.perf_counter_ns()
        _ = imputer.transform(data)
        times_sk.append(time.perf_counter_ns() - start)

    elapsed_rs = sorted(times_rs)[N // 2]  # median
    elapsed_sk = sorted(times_sk)[N // 2]

    assert elapsed_rs < elapsed_sk, f"Rust: {
        elapsed_rs}ns  sklearn: {elapsed_sk}ns"


def test_nans():
    data = np.random.rand(500, 50)
    data[data < 0.18] = np.nan
    imputed = KnnImputer().fit(data).transform(data)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
