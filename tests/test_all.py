from gouda import Imputers
from sklearn.utils.estimator_checks import check_estimator
import pandas as pd
import numpy as np
import pytest
from tests.utily import missing_data, missing_cat_data, missing_mixed_data, expected_warning

param = {
    "KnnImputer": {
        "encoding": "label",
        "metric": "gower",
    },
    "SimpleImputer": {
        "encoding": "label",
    },
    "Mice": {
        "encoding": "label",
    },
    "SVMImputer": {
        "encoding": "label",
    },
    "ConstantImputer": {
        "encoding": "label",
    },
    "GAIN": {
        "encoding": "label",
    },
}


@pytest.mark.parametrize("model", Imputers)
def test_raises_on_all_nan_column(model):
    """A column that is entirely missing carries no signal -- the model
    should either raise a clear, informative error or fill with some
    documented default, but should not silently fail/crash uninformatively."""
    data = np.random.rand(500, 5)
    data[:, 0] = np.nan
    m = model(**param[model.__name__])
    try:
        out = m.fit_transform(data)
        assert not np.isnan(out).any()
    except ValueError:
        pass  # acceptable: explicit, informative failure


class TestInputValidity:
    @pytest.mark.parametrize("model", Imputers)
    def test_works_with_cat(self, missing_cat_data, model):
        data, missing, _ = missing_cat_data
        with expected_warning(model, param[model.__name__]):
            imputed = model(**param[model.__name__]).fit(missing).transform(missing)
        print("data:\n", data)
        print(f"imputed:\n{imputed}")
        assert not imputed.isna().any().any(), "Imputed still has missing values"

    @pytest.mark.parametrize("model", Imputers)
    def test_works_with_mixed(self, missing_mixed_data, model):
        data, missing, _ = missing_mixed_data
        with expected_warning(model, param[model.__name__]):
            imputed = model(**param[model.__name__]).fit(missing).transform(missing)
        print("data:\n", data)
        print(f"imputed:\n{imputed}")
        assert not imputed.isna().any().any(), "Imputed still has missing values"

    @pytest.mark.parametrize("model", Imputers)
    def test_observed_values_stay_unchanged_for_fortran_contiguous_dataframe(self, model):
        data = pd.DataFrame(
            np.asfortranarray([
                [5.1, 3.5, 1.4, 0.2],
                [4.9, 3.0, 1.4, 0.2],
                [4.7, 3.2, 1.3, 0.2],
                [4.6, 3.1, 1.5, 0.2],
                [5.0, 3.6, 1.4, 0.2],
                [5.4, 3.9, 1.7, 0.4],
            ]),
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )
        missing = data.copy()
        missing.iloc[1, 0] = np.nan
        missing.iloc[2, 1] = np.nan
        missing.iloc[4, 2] = np.nan
        observed = missing.notna().to_numpy()

        assert missing.to_numpy(copy=False).flags.f_contiguous

        imputed = model(
            **param[model.__name__]
        ).fit_transform(missing)

        np.testing.assert_allclose(
            imputed.to_numpy(dtype=float)[observed],
            data.to_numpy(dtype=float)[observed],
            rtol=1e-10,
            atol=1e-12,
            err_msg="MICE modified an observed value in a Fortran-contiguous DataFrame",
        )


class TestOutputValidity:
    @pytest.mark.parametrize("model", Imputers)
    def test_nans_simple(self, missing_data, model):
        data, missing, _ = missing_data
        imputed = model().fit(missing).transform(missing)
        print("data:\n", data)
        print(f"imputed:\n{imputed}")
        assert not np.isnan(imputed).any(), "Imputed still has missing values"

    @pytest.mark.parametrize("model", Imputers)
    def test_observed_values_unchanged(self, missing_data, model):
        """A well-behaved imputer should not alter values that were already
        observed — only fill in the missing ones."""
        X_full, X_missing, mask = missing_data
        m = model()
        out = m.fit_transform(X_missing)
        observed = ~mask
        np.testing.assert_allclose(
            out[observed], X_full[observed], rtol=1e-5, atol=1e-5,
            err_msg="Imputer modified originally-observed (non-missing) values",
        )

    @pytest.mark.parametrize("model", Imputers)
    def test_cat_correct_values(self, missing_cat_data, model):
        data, missing, _ = missing_cat_data
        with expected_warning(model, param[model.__name__]):
            imputed = model(**param[model.__name__]).fit(missing).transform(missing)
        print("data", data.iloc[:20])
        print("imputed", imputed[:20])
        assert isinstance(imputed, pd.DataFrame), "No DataFrame returned"
        d = {
            col: set(data[col].dropna().unique()) == set(
                missing[col].dropna().unique())
            for col in data.columns.intersection(missing.columns)
        }
        for key, b in d.items():
            assert b, f"value mismatch in {key}"

    @pytest.mark.parametrize("model", Imputers)
    def test_output_is_finite(self, missing_data, model):
        _, X_missing, _ = missing_data
        m = model()
        out = m.fit_transform(X_missing)
        assert np.isfinite(out).all(), "Output contains inf/-inf/NaN"

    @pytest.mark.parametrize("model", Imputers)
    def test_imputed_values_within_reasonable_range(self, missing_data, model):
        """Imputed values shouldn't wildly exceed the observed data's range
        (a common failure mode for a poorly-trained/unstable GAN)."""
        X_full, X_missing, mask = missing_data
        m = model()
        out = m.fit_transform(X_missing)

        lo, hi = X_full.min(), X_full.max()
        span = hi - lo
        buffer = 0.5 * span  # generous slack
        imputed_vals = out[mask]
        assert imputed_vals.min() >= lo - buffer
        assert imputed_vals.max() <= hi + buffer

    @pytest.mark.parametrize("model", Imputers)
    def test_accepts_pandas_dataframe(self, missing_data, model):
        _, X_missing, _ = missing_data
        df = pd.DataFrame(X_missing, columns=[
                          f"f{i}" for i in range(X_missing.shape[1])])
        m = model(**param[model.__name__])
        if model.__name__ == "ConstantImputer":
            with pytest.warns(UserWarning):
                out = m.fit_transform(df)
        else:
            out = m.fit_transform(df)
        assert not np.isnan(np.asarray(out)).any()


@pytest.mark.parametrize("model", Imputers)
def test_checksklearn(model):
    m = model(**param[model.__name__])
    if model.__name__ == "Mice":
        check_estimator(
            m,
            expected_failed_checks={
                "check_estimators_pickle": "known float drift in transform after pickle, tracked in #123",
            },
        )
    elif model.__name__ == "ConstantImputer":
        with pytest.warns(UserWarning):
            check_estimator(m)
    else:
        check_estimator(m)
