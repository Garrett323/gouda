from gouda import SVMImputer
import pytest
import pandas as pd
import numpy as np
from sklearn.utils.estimator_checks import check_estimator

def test_kernel_getter():
    model = SVMImputer()
    assert model.kernel == "linear"

def test_nans():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    print("Trying to fit model")
    model = SVMImputer().fit(data)
    print("predicting..")
    imputed = model.transform(data)
    print(imputed)
    assert not np.isnan(imputed).any(), "Imputed still has missing values"


@pytest.fixture
def simple_data():
    # 2 fully-observed columns to help SVM predict the 3rd (missing) column
    rng = np.random.RandomState(0)
    X = rng.rand(50, 3)
    X_missing = X.copy()
    mask = rng.rand(50) < 0.2
    X_missing[mask, 2] = np.nan
    return X_missing, X, mask
 
 
def test_no_missing_values_in_output(simple_data):
    """Core functionality: output must contain no NaNs where input had them."""
    X_missing, _, _ = simple_data
    imputer = SVMImputer()
    X_out = imputer.fit_transform(X_missing)
    assert not np.isnan(X_out).any()
 
 
def test_output_shape_preserved(simple_data):
    """Transform must not change the shape of the input."""
    X_missing, _, _ = simple_data
    imputer = SVMImputer()
    X_out = imputer.fit_transform(X_missing)
    assert X_out.shape == X_missing.shape
 
 
def test_fit_transform_equals_fit_then_transform(simple_data):
    """fit_transform(X) should match fit(X).transform(X)."""
    X_missing, _, _ = simple_data
    imp1 = SVMImputer()
    out1 = imp1.fit_transform(X_missing)
 
    imp2 = SVMImputer()
    imp2.fit(X_missing)
    out2 = imp2.transform(X_missing)
 
    np.testing.assert_allclose(out1, out2, rtol=1e-5)
 
 
def test_no_missing_values_passthrough():
    """If there are no missing values, output should equal input (unchanged)."""
    rng = np.random.RandomState(1)
    X = rng.rand(20, 4)
    imputer = SVMImputer()
    X_out = imputer.fit_transform(X)
    np.testing.assert_allclose(X_out, X, rtol=1e-5)
 
 
def test_all_missing_column_raises_or_handles():
    """
    Edge case: a column that is entirely missing has no signal to fit an SVM on.
    Should either raise an informative error or fall back gracefully
    (e.g., impute with a constant) -- must not silently crash or return NaN.
    """
    rng = np.random.RandomState(2)
    X = rng.rand(20, 3)
    X[:, 1] = np.nan
    imputer = SVMImputer()
    try:
        X_out = imputer.fit_transform(X)
        assert not np.isnan(X_out).any()
    except ValueError:
        pass  # explicit, informative failure is acceptable
 
 
def test_single_row_input():
    """Edge case: a single-sample input should not crash transform."""
    rng = np.random.RandomState(3)
    X_fit = rng.rand(30, 3)
    X_fit[::5, 0] = np.nan
    imputer = SVMImputer().fit(X_fit)
 
    X_single = rng.rand(1, 3)
    X_single[0, 0] = np.nan
    X_out = imputer.transform(X_single)
    assert X_out.shape == (1, 3)
    assert not np.isnan(X_out).any()
 
 
def test_dataframe_input_supported():
    """Should accept pandas DataFrame input, consistent with numpy behavior."""
    rng = np.random.RandomState(4)
    X = rng.rand(30, 3)
    X[::4, 1] = np.nan
    df = pd.DataFrame(X, columns=["a", "b", "c"])
 
    imp_np = SVMImputer().fit_transform(X)
    imp_df = SVMImputer(encoding=("label")).fit_transform(df)
 
    arr = imp_df.values if hasattr(imp_df, "values") else imp_df
    np.testing.assert_allclose(arr, imp_np, rtol=1e-5)
 
 
def test_transform_before_fit_raises():
    """Calling transform before fit should raise NotFittedError (sklearn convention)."""
    from sklearn.exceptions import NotFittedError
 
    imputer = SVMImputer()
    with pytest.raises(NotFittedError):
        imputer.transform(np.array([[1.0, np.nan], [2.0, 3.0]]))
 
 
def test_mismatched_feature_count_at_transform_raises(simple_data):
    """Transform with a different number of features than fit should raise."""
    X_missing, _, _ = simple_data
    imputer = SVMImputer().fit(X_missing)
    X_wrong = X_missing[:, :2]  # drop a column
    with pytest.raises(ValueError):
        imputer.transform(X_wrong)
 

def test_checksklearn():
    check_estimator(SVMImputer(encoding="label"))
