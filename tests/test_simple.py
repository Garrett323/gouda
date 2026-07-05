from gouda import SimpleImputer, ConstantImputer
from sklearn.utils.estimator_checks import check_estimator
import pytest
import pandas as pd
import numpy as np


def test_nans_simple():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = SimpleImputer().fit(data).transform(data)
    print("data:\n", data)
    print(f"imputed:\n{imputed}")
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


def test_simple_values():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = SimpleImputer().fit(data).transform(data)
    col_means: list[float] = np.nanmean(data, axis=0)
    mask = np.isnan(data)
    for col in range(data.shape[1]):
        imputed_col = imputed[mask[:, col], col]
        assert np.allclose(
            imputed_col,
            col_means[col]
        ), f"Expected: {col_means[col]}\n{imputed_col}"


def test_nans_constant():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = ConstantImputer(value=4.0).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


def test_constant_values():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    mask = np.isnan(data)
    imputed = ConstantImputer(value=4.0).fit(data).transform(data)
    assert np.isclose(imputed[mask], 4.0).all(), \
        "put wrong values in constant imputation"
    imputed = ConstantImputer().fit(data).transform(data)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
    assert np.isclose(imputed[mask], 0.0).all(), \
        "put wrong values in constant imputation"

def test_categoricals():
    data = pd.DataFrame({
        "a": np.random.rand(200),
        "b": ["a" for _ in range(150)] + ["b" for _ in range(50)]
    })
    mask = np.random.rand(200, 2) > 0.5
    data[mask] = pd.NA
    imputed = SimpleImputer(encoding='label').fit(data).transform(data)
    print("data", data.iloc[:20])
    print("imputed", imputed[:20])
    assert isinstance(imputed, pd.DataFrame), "No DataFrame returned"
    for l in imputed.iloc[mask[:, 1], 1]:
        assert l == "a"


def test_checksklearn_simple():
    check_estimator(SimpleImputer(encoding="label"))

def test_checksklearn_const():
    # this is needed since constant throws a warning when using categorical values but sklearn wants to check if it supports categorical data
    with pytest.warns(): 
        check_estimator(ConstantImputer(encoding="label"))
