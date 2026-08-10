from gouda import GAIN, SimpleImputer
import pytest
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import pandas as pd

RANDOM_STATE = 42

def apply_mcar_mask(X, missing_rate=0.2, seed=RANDOM_STATE):
    """Missing Completely At Random mask. Returns (X_missing, mask) where
    mask[i, j] == True means the value is MISSING (i.e. was removed)."""
    rng = np.random.RandomState(seed)
    mask = rng.rand(*X.shape) < missing_rate
    # guarantee no fully-missing row/column, which is undefined behavior
    # for basically any imputer
    for i in range(X.shape[0]):
        if mask[i].all():
            mask[i, rng.randint(X.shape[1])] = False
    for j in range(X.shape[1]):
        if mask[:, j].all():
            mask[rng.randint(X.shape[0]), j] = False
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing, mask
 
 
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))
 
 
@pytest.fixture
def complete_data():
    """Fully-observed ground-truth data, optionally with correlated features
    so that a good imputer should be able to beat mean imputation."""
    seed = RANDOM_STATE
    n_samples = 500
    n_features = 5
    correlated = True
    rng = np.random.RandomState(seed)
    if correlated:
        # latent factors driving correlated observed features
        n_latent = max(2, n_features // 3)
        latent = rng.normal(size=(n_samples, n_latent))
        loadings = rng.normal(size=(n_latent, n_features))
        noise = rng.normal(scale=0.3, size=(n_samples, n_features))
        X = latent @ loadings + noise
    else:
        X = rng.normal(size=(n_samples, n_features))
    return X.astype(np.float64)
 
 
@pytest.fixture
def missing_data(complete_data):
    X_missing, mask = apply_mcar_mask(complete_data, missing_rate=0.2)
    return complete_data, X_missing, mask


class TestEdgeCases:
    def test_no_missing_values(self, complete_data):
        """If there's nothing to impute, output should equal input (up to
        numerical tolerance) and definitely shouldn't error."""
        model = GAIN(random_state=RANDOM_STATE)
        out = model.fit_transform(complete_data)
        np.testing.assert_allclose(out, complete_data, rtol=1e-4, atol=1e-4)
 
    def test_single_column(self):
        X = np.random.RandomState(0).normal(size=(200, 1))
        X_missing, mask = apply_mcar_mask(X, missing_rate=0.3)
        model = GAIN(random_state=RANDOM_STATE)
        out = model.fit_transform(X_missing)
        assert not np.isnan(out).any()
 
    def test_small_number_of_samples(self):
        X = np.random.RandomState(0).normal(size=(20, 5))
        X_missing, _ = apply_mcar_mask(X, missing_rate=0.2)
        model = GAIN(random_state=RANDOM_STATE)
        out = model.fit_transform(X_missing)  # should not crash
        assert not np.isnan(out).any()
 
    def test_column_with_high_missingness(self, complete_data):
        """One column mostly missing (e.g. 90%) -- stress test."""
        rng = np.random.RandomState(0)
        mask = np.zeros_like(complete_data, dtype=bool)
        mask[:, 0] = rng.rand(complete_data.shape[0]) < 0.9
        # keep at least one observed value in that column
        mask[0, 0] = False
        X_missing = complete_data.copy()
        X_missing[mask] = np.nan
        model = GAIN(random_state=RANDOM_STATE)
        out = model.fit_transform(X_missing)
        assert not np.isnan(out).any()
 
    def test_transform_on_new_unseen_data(self, missing_data):
        """If the API implies fit/transform separation, transform() should
        work on new data with the same schema as fit(), not just the
        training data (this may legitimately raise NotImplementedError for
        pure transductive GAIN variants -- adjust/skip if that's the design)."""
        _, X_missing, _ = missing_data
        model = GAIN(random_state=RANDOM_STATE)
        model.fit(X_missing[:400])
 
        X_new = X_missing[400:]
        try:
            out_new = model.transform(X_new)
        except NotImplementedError:
            pytest.skip("Model is transductive-only; transform on new data not supported")
        assert out_new.shape == X_new.shape
        assert not np.isnan(out_new).any()
 
    def test_raises_on_all_nan_column(self, complete_data):
        """A column that is entirely missing carries no signal -- the model
        should either raise a clear, informative error or fill with some
        documented default, but should not silently fail/crash uninformatively."""
        X_missing = complete_data.copy()
        X_missing[:, 0] = np.nan
        model = GAIN(random_state=RANDOM_STATE)
        try:
            out = model.fit_transform(X_missing)
            assert not np.isnan(out).any()
        except ValueError:
            pass  # acceptable: explicit, informative failure
 

 
class TestImputationQuality:
 
    def test_beats_or_matches_mean_imputation_on_correlated_data(self, missing_data):
        """On data with real feature correlation, GAIN should be competitive
        with (ideally better than) simple mean imputation. We allow some
        slack since this is stochastic and dataset-dependent."""
        X_full, X_missing, mask = missing_data
 
        gain = GAIN(random_state=RANDOM_STATE)
        gain_out = gain.fit_transform(X_missing)
        gain_rmse = rmse(gain_out[mask], X_full[mask])
 
        baseline = SimpleImputer()
        baseline_out = baseline.fit_transform(X_missing)
        baseline_rmse = rmse(baseline_out[mask], X_full[mask])
 
        assert gain_rmse <= baseline_rmse * 1.25, (
            f"GAIN RMSE ({gain_rmse:.4f}) is much worse than mean-imputation "
            f"baseline ({baseline_rmse:.4f}) on correlated data"
        )
 
    def test_reasonable_absolute_error(self, missing_data):
        X_full, X_missing, mask = missing_data
        model = GAIN(random_state=RANDOM_STATE)
        out = model.fit_transform(X_missing)
        mae = float(np.mean(np.abs(out[mask] - X_full[mask])))
        data_scale = float(np.std(X_full))
        assert mae < 2.0 * data_scale, (
            f"Mean absolute imputation error ({mae:.4f}) is very large "
            f"relative to the data scale ({data_scale:.4f})"
        )
 
    @pytest.mark.parametrize("missing_rate", [0.05, 0.2, 0.5])
    def test_quality_degrades_gracefully_with_missing_rate(self, complete_data, missing_rate):
        """Error should increase (or stay flat) as missingness increases,
        not blow up erratically -- a sanity check on stability."""
        X_missing, mask = apply_mcar_mask(complete_data, missing_rate=missing_rate)
        model = GAIN(random_state=RANDOM_STATE)
        out = model.fit_transform(X_missing)
        err = rmse(out[mask], complete_data[mask])
        data_scale = float(np.std(complete_data))
        assert err < 3.0 * data_scale, (
            f"RMSE exploded at missing_rate={missing_rate}: {err:.4f}"
        )
 
    def test_reproducibility_with_fixed_random_state(self, missing_data):
        """Same random_state should give (near-)deterministic results."""
        _, X_missing, _ = missing_data
        out1 = GAIN(random_state=123).fit_transform(X_missing)
        out2 = GAIN(random_state=123).fit_transform(X_missing)
        np.testing.assert_allclose(out1, out2, rtol=1e-3, atol=1e-3)
 
    def test_different_seeds_give_similar_quality(self, missing_data):
        """Different random seeds shouldn't cause wildly different quality --
        checks the training procedure isn't unstable."""
        X_full, X_missing, mask = missing_data
        rmses = []
        for seed in (1, 2, 3):
            out = GAIN(random_state=seed).fit_transform(X_missing)
            rmses.append(rmse(out[mask], X_full[mask]))
        rmses = np.array(rmses)
        assert rmses.std() < rmses.mean(), (
            f"High variance in imputation quality across seeds: {rmses}"
        )
 
