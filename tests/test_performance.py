import numpy as np
import time
from sklearn.impute import KNNImputer as SKKNN
from gouda import KnnImputer as RSKNN


def test_time():
    data = np.random.rand(500, 50)
    data[data < 0.78] = np.nan

    N = 5
# Warmup
    for _ in range(3):
        RSKNN().fit(data).transform(data)
        SKKNN().fit(data).transform(data)

# Benchmark Rust
    times_rs = []
    for _ in range(N):
        imputer = RSKNN()
        imputer.fit(data)
        start = time.perf_counter_ns()
        _ = imputer.transform(data)
        times_rs.append(time.perf_counter_ns() - start)

# Benchmark sklearn
    times_sk = []
    for _ in range(N):
        imputer = SKKNN()
        imputer.fit(data)
        start = time.perf_counter_ns()
        _ = imputer.transform(data)
        times_sk.append(time.perf_counter_ns() - start)

    elapsed_rs = sorted(times_rs)[N // 2]  # median
    elapsed_sk = sorted(times_sk)[N // 2]

    assert elapsed_rs < elapsed_sk, f"Rust: {
        elapsed_rs}ns  sklearn: {elapsed_sk}ns"


def test_nans():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = RSKNN().fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
    imputed = RSKNN(metric="expected_distance").fit(data).transform(data)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


def test_isclose():
    data = np.random.rand(500, 50)
    data[data < 0.38] = np.nan
    imputed_rs = RSKNN().fit(data).transform(data)
    imputed_sk = SKKNN().fit(data).transform(data)
    assert not np.isclose(imputed_rs, imputed_sk).all(), "wrong distances"


def test_ed():
    data = np.random.rand(500, 50)
    assert (data < 1.0).all()
    assert (data >= 0.0).all()
    data[data < 0.38] = np.nan
    imputed_rs = RSKNN(metric="expected_distance").fit(data).transform(data)
    imputed_sk = SKKNN().fit(data).transform(data)
    assert not np.isclose(imputed_rs, imputed_sk).all(), "wrong distances"
