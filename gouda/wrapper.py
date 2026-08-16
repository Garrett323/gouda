from typing import Literal
from gouda.gouda import KnnImputerRS, SimpleImputerRS, ConstantImputerRS, SVMImputerRS, MiceRS
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import validate_data
import numpy as np
import pandas as pd


def imputer_tags(tags):
    tags.input_tags.allow_nan = True
    tags.input_tags.string = True   # declares intentional string/categorical support
    return tags


class KnnImputer(TransformerMixin, BaseEstimator):
    def __init__(self, *,
                 k: int = 5,
                 metric: Literal["nan_euclid", "gower",
                                 "expected_distance"] = "nan_euclid",
                 weights: Literal["uniform", "distance"] = "uniform",
                 encoding: None | Literal["label"] = None,
                 ) -> None:
        self.encoding = encoding
        self.metric = metric
        self.weights = weights
        self.k = k
        self._model = None

    def fit(self, X, y=None):
        if self.k <= 0:
            raise ValueError("To use knn imputation please pass k > 0!")
        self._model = KnnImputerRS(
            self.k, metric=self.metric, weights=self.weights, encoding=self.encoding
        )
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None,
                          ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return imputer_tags(super().__sklearn_tags__())


class SimpleImputer(TransformerMixin, BaseEstimator):
    def __init__(self, *,
                 encoding: None | Literal["label"] = None,
                 ) -> None:
        self.encoding = encoding
        self._model = None

    def fit(self, X, y=None):
        self._model = SimpleImputerRS(encoding=self.encoding)
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None,
                          ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return imputer_tags(super().__sklearn_tags__())


class ConstantImputer(TransformerMixin, BaseEstimator):
    def __init__(self, *,
                 value=0.0,
                 encoding: None | Literal["label"] = None,
                 ) -> None:
        self.value = value
        self.encoding = encoding
        self._model = None

    def fit(self, X, y=None):
        self._model = ConstantImputerRS(self.value, encoding=self.encoding)
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None,
                          ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return imputer_tags(super().__sklearn_tags__())


class SVMImputer(TransformerMixin, BaseEstimator):
    def __init__(self, *,
                 kernel: Literal["linear", "rbf", "polynomial", "sigmoid"] = "linear",
                 encoding: None | Literal["label"] = None,
                 ) -> None:
        self.kernel = kernel
        self.encoding = encoding
        self._model = None

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        self._model = SVMImputerRS(kernel=self.kernel, encoding=self.encoding)
        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None,
                          ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return imputer_tags(super().__sklearn_tags__())


class Mice(TransformerMixin, BaseEstimator):
    def __init__(self, *,
                 max_iter: int = 10,
                 backend: Literal["linear", "ridge", "pmm"] = "linear",
                 alpha: float = 1.0,
                 pmm_backend: Literal["linear", "ridge"] = "linear",
                 encoding: None | Literal["label"] = None,
                 ) -> None:
        self.max_iter = max_iter
        self.backend = backend
        self.alpha = alpha
        self.pmm_backend = pmm_backend
        self.encoding = encoding
        self._model = None

    def fit(self, X, y=None):
        self._model = MiceRS(
            max_iter=self.max_iter,
            backend=self.backend,
            pmm_backend=self.pmm_backend,
            alpha=self.alpha,
            encoding=self.encoding,
        )
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        self._model.fit(X)
        self.n_iter_ = self._model._n_iter
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None,
                          ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return imputer_tags(super().__sklearn_tags__())
