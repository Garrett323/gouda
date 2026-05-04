from gouda import SimpleImputer, ConstantImputer
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
    col_means = np.nanmean(data, axis=0)
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
    imputed = ConstantImputer(4.0).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


def test_constant_values():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    mask = np.isnan(data)
    imputed = ConstantImputer(4.0).fit(data).transform(data)
    assert np.isclose(imputed[mask], 4.0).all(), \
        "put wrong values in constant imputation"
    imputed = ConstantImputer.zero().fit(data).transform(data)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
    assert np.isclose(imputed[mask], 0.0).all(), \
        "put wrong values in constant imputation"
