import numpy as np
import time
from sklearn.impute import KNNImputer
from gouda import KnnImputer 

def test_t():
    data = np.random.rand(500, 5)
    data[data < 0.10] = np.nan
    # data[0,1] = np.nan
    # data[0,4] = np.nan
    # data[1,1] = np.nan
    # data[1,4] = np.nan

    N = 20

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

    assert elapsed_rs < elapsed_sk, f"Rust: {elapsed_rs}ns  sklearn: {elapsed_sk}ns"
