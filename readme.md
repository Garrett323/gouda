# Gouda

Fast, scikit-learn compatible imputation for Python, implemented primarily in Rust.

Designed for mixed-type datasets with missing values, including categorical features.

## Features

- 🚀 Rust-powered performance
- ✅ Fully scikit-learn compatible (`fit`, `transform`, `fit_transform`)
- 🐼 Supports NumPy arrays and pandas DataFrames
- 🏷️ Native support for categorical features
- 💾 Pickle compatible

## Installation

```bash
pip install gouda-cheese
# or
uv add gouda-cheese
```

## Usage

```python
from gouda import KnnImputer

imputer = KnnImputer()
X_imputed = imputer.fit_transform(X)
```

## Available Imputers

- `KnnImputer`
- `Mice`
- `SimpleImputer`
- `ConstantImputer`
- `SVMImputer`

## Why Gouda?

Most existing imputation libraries either:
- only support numerical data,
- require cumbersome preprocessing for categoricals, or
- become slow on large datasets.
- limited selection of algorithms.

Gouda provides high-performance, sklearn-compatible imputers with first-class support for mixed numerical and categorical data.
