from typing import Literal
from gouda.gouda import KnnImputerRS, SimpleImputerRS, ConstantImputerRS, SVMImputerRS, MiceRS
from sklearn.impute._base import _BaseImputer
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import validate_data
from sklearn.utils import Tags
import numpy as np
import pandas as pd

def __tags(tags):
    tags.input_tags.allow_nan = True
    tags.input_tags.string = True   # declares intentional string/categorical support
    return tags


class KnnImputer(_BaseImputer, TransformerMixin):
    def __init__(self, *, 
                 k: int = 5,
                 metric: Literal["nan_euclid", "gower", "expected_distance"] = "nan_euclid",
                 weights: Literal["uniform", "distance"] = "uniform",
                 encoding: None | Literal["label"] = None,
                 missing_values=np.nan, 
                 add_indicator: bool = False, 
                 keep_empty_features: bool = False
                 ) -> None:
        super().__init__(missing_values=missing_values, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        self.encoding = encoding
        self.metric = metric
        self.weights= weights
        self.k = k
        self._model = None

    def fit(self, X, y=None):
        self._model = KnnImputerRS(
            self.k,metric=self.metric, weights=self.weights, encoding=self.encoding 
        )
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")  
        X = np.asarray(X)  

        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return __tags(super().__sklearn_tags__())


class SimpleImputer(_BaseImputer, TransformerMixin):
    def __init__(self, *, 
                 encoding: None | Literal["label"] = None,
                 missing_values=np.nan, 
                 add_indicator: bool = False, 
                 keep_empty_features: bool = False
                 ) -> None:
        super().__init__(missing_values=missing_values, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        self.encoding = encoding
        self._model = None

    def fit(self, X, y=None):
        self._model = SimpleImputerRS(encoding=self.encoding)
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")  
        X = np.asarray(X)  

        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return __tags(super().__sklearn_tags__())


class ConstantImputer(_BaseImputer, TransformerMixin):
    def __init__(self, *, 
                 value=0.0,
                 encoding: None | Literal["label"] = None,
                 missing_values=np.nan, 
                 add_indicator: bool = False, 
                 keep_empty_features: bool = False
                 ) -> None:
        super().__init__(missing_values=missing_values, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        self.value = value
        self.encoding = encoding
        self._model = None

    def fit(self, X, y=None):
        self._model = ConstantImputerRS(self.value, encoding=self.encoding)
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")  
        X = np.asarray(X)  

        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return __tags(super().__sklearn_tags__())


class SVMImputer(_BaseImputer, TransformerMixin):
    def __init__(self, *, 
                 kernel: Literal["linear", "rbf", "polynomial", "sigmoid", "precomputed"] = "linear",
                 encoding: None | Literal["label"] = None,
                 missing_values=np.nan, 
                 add_indicator: bool = False, 
                 keep_empty_features: bool = False
                 ) -> None:
        super().__init__(missing_values=missing_values, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        self.kernel= kernel 
        self.encoding = encoding
        self._model = None

    def fit(self, X, y=None):
        self._model = SVMImputerRS(kernel=self.kernel, encoding=self.encoding)
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")  
        X = np.asarray(X)  

        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return __tags(super().__sklearn_tags__())


class Mice(_BaseImputer, TransformerMixin):
    def __init__(self, *, 
                 max_iter: int = 10,
                 backend: Literal["linear", "ridge", "pmm"] = "linear",
                 alpha: float = 1.0,
                 pmm_backend: Literal["linear", "ridge"] = "linear",
                 encoding: None | Literal["label"] = None,
                 missing_values=np.nan, 
                 add_indicator: bool = False, 
                 keep_empty_features: bool = False
                 ) -> None:
        super().__init__(missing_values=missing_values, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        self.max_iter= max_iter
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
            alpha = self.alpha,
            encoding=self.encoding,
        )
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")  
        X = np.asarray(X)  

        self._model.fit(X)
        self.n_iter_ = self._model._n_iter
        return self

    def transform(self, X):
        if self._model is None:
            raise NotFittedError
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan", reset=False)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        else:
            X = np.asarray(X)
        return self._model.transform(X)

    def __sklearn_tags__(self):
        return __tags(super().__sklearn_tags__())
