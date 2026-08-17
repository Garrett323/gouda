# Gouda

Gouda is a Python library for missing-data imputation. Its core estimators are
implemented in Rust and exposed through scikit-learn-style `fit`, `transform`,
and `fit_transform` methods. It accepts NumPy arrays and pandas DataFrames and
can label-encode mixed numerical/categorical DataFrames.

## Requirements and installation

Gouda requires Python 3.11 or newer.

```bash
pip install gouda-cheese
uv add "gouda-cheese"
```

With the optional PyTorch-based GAIN imputer:

```bash
pip install "gouda-cheese[deep]"
uv add "gouda-cheese[deep]"
```

The distribution name is `gouda-cheese`; the Python import name is `gouda`.

## Quick start

```python
import numpy as np
from gouda import KnnImputer

X = np.array([
    [1.0, np.nan, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 5.0, 9.0],
])

imputer = KnnImputer(k=2, weights="distance")
X_imputed = imputer.fit_transform(X)
```

The estimators follow the usual scikit-learn transformer API, so they can also
be used in a pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gouda import SimpleImputer

pipeline = make_pipeline(SimpleImputer(), StandardScaler())
X_transformed = pipeline.fit_transform(X)
```

## Mixed-type DataFrames

Pass `encoding="label"` to encode string or categorical columns during 
imputation and decode them in the returned DataFrame. KNN should use the Gower
metric for mixed data.

```python
import numpy as np
import pandas as pd
from gouda import KnnImputer

df = pd.DataFrame({
    "age": [29.0, np.nan, 41.0, 35.0],
    "plan": ["basic", "plus", pd.NA, "plus"],
})

result = KnnImputer(
    k=2,
    metric="gower",
    encoding="label",
).fit_transform(df)
```

DataFrame input produces DataFrame output with the original column names.
Categorical values are learned from the data supplied to `fit`; transform data
must have the same feature layout and compatible categories.

## Imputers

| Estimator | Purpose | Main parameters |
| --- | --- | --- |
| `SimpleImputer` | Column mean for numerical data and mode for label-encoded categoricals | `encoding` |
| `ConstantImputer` | Replace missing values with a fixed value (default `0.0`) | `value`, `encoding` |
| `KnnImputer` | Nearest-neighbour imputation | `k`, `metric`, `weights`, `encoding` |
| `Mice` | Iterative chained-equation imputation | `max_iter`, `backend`, `alpha`, `pmm_backend`, `encoding` |
| `SVMImputer` | Per-feature SVM-based imputation | `kernel`, `encoding` |
| `GAIN` | Generative adversarial imputation implemented with PyTorch | training/network parameters, `encoding` |

KNN metrics are `"nan_euclid"` (default), `"expected_distance"`, and
`"gower"`; weights are `"uniform"` (default) and `"distance"`. MICE backends
are `"linear"`, `"ridge"`, and `"pmm"`; predictive mean matching can use a
`"linear"` or `"ridge"` backend. SVM kernels are `"linear"` (default),
`"rbf"`, `"polynomial"`, and `"sigmoid"`.

GAIN is imported lazily and is available only when the `deep` extra is
installed. It supports CPU and CUDA devices (selected automatically unless
`device` is set), deterministic seeding through `random_state`, and early
stopping. For example:

```python
from gouda import GAIN

imputer = GAIN(max_epochs=100, patience=15, random_state=42)
X_imputed = imputer.fit_transform(X)
```

## Input notes

- Missing numerical values should be represented by `numpy.nan`; pandas
  missing values are accepted in DataFrames.
- No training column may be entirely missing, because there is no value from
  which to learn an imputation.
- `transform` requires the same number and order of features used during
  `fit`.
- Use `encoding="label"` for categorical or mixed-type DataFrames. For mixed
  KNN data, use `metric="gower"`.
- `ConstantImputer` fills encoded categorical cells with the numeric constant;
  it is therefore mainly intended for numerical data.

## Development

The native extension is built with Maturin. From a clone of the repository:

```bash
uv sync
uv run maturin develop --release
uv run pytest -m "not heavy"
```

Run the complete Python suite, including timing-heavy tests, with
`uv run pytest`. Rust unit tests can be run with `cargo test --release`.

Benchmarking has its own reproducibility and plotting guide in
[`benchmarking/README.md`](benchmarking/README.md).
