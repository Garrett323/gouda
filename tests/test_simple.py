from gouda import SimpleImputer, ConstantImputer
import numpy as np


def test_nans_simple():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = SimpleImputer().fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


def test_nans_constant():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = ConstantImputer(4.0).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
    imputed = ConstantImputer.zero().fit(data).transform(data)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


