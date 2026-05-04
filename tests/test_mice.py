import numpy as np
from gouda import Mice, SimpleImputer

def test_nans():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = Mice(backend="linear", n_iterations=2).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
    imputed = Mice(backend="ridge", n_iterations=2).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"
    imputed = Mice(backend="pmm", n_iterations=2).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"

def test_iterations_work():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = Mice(backend="linear", n_iterations=2).fit(data).transform(data)
    imputed2 = Mice(backend="linear",n_iterations=2).fit(data).transform(data)
    imputed3 = Mice(backend="linear",n_iterations=4).fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    assert np.allclose(imputed, imputed2), "Unexpected difference between different iterations"
    print("imputed3:\n", imputed3)
    assert not np.allclose(imputed, imputed3), "No difference between iterations"

def test_not_simple():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = Mice(backend="linear",n_iterations=2).fit(data).transform(data)
    imputed_simple = SimpleImputer().fit(data).transform(data)
    print("data:\n", data)
    print("imputed:\n", imputed)
    print("imputed simple:\n", imputed_simple)
    assert not np.allclose(imputed, imputed_simple), "Still at initial values"
