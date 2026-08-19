from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


@dataclass
class _RModules:
    conversion: Any
    default_converter: Any
    pandas2ri: Any
    packages: Any


class RMiceImputer(BaseEstimator, TransformerMixin):
    """Thin sklearn-style wrapper around the R `mice` package.

    This adapter is benchmark-oriented:
    - `fit` runs the R imputation once and caches the completed training data.
    - `transform` returns the cached result for the training matrix and reruns
      the R imputation only when called on different data.

    That keeps benchmark `fit + transform` totals correct while still exposing a
    Python-side API that matches the existing harness.
    """

    def __init__(
        self,
        *,
        max_iter: int = 10,
        m: int = 1,
        method: str | None = None,
        seed: int | None = 42,
        print_flag: bool = False,
    ) -> None:
        self.max_iter = max_iter
        self.m = m
        self.method = method
        self.seed = seed
        self.print_flag = print_flag
        self._fitted = False

    def fit(self, X, y=None):
        df = self._to_frame(X)
        self._is_dataframe = isinstance(X, pd.DataFrame)
        self._fit_columns = list(df.columns)
        self._fit_dtypes = df.dtypes.to_dict()
        self._fit_input = df.copy(deep=True)
        self._completed_fit = self._run_mice(df)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise NotFittedError("Call fit before transform.")

        df = self._to_frame(X)
        if self._matches_fit_input(df):
            completed = self._completed_fit.copy(deep=True)
        else:
            warnings.warn(
                "RMiceImputer.transform received data different from the fitted "
                "matrix; rerunning R mice on the new data.",
                RuntimeWarning,
                stacklevel=2,
            )
            completed = self._run_mice(df)

        completed = self._restore_dtypes(completed)
        if self._is_dataframe:
            completed.index = X.index
            return completed
        return completed.to_numpy()

    def _run_mice(self, df: pd.DataFrame) -> pd.DataFrame:
        rmods = self._import_r_modules()
        mice = rmods.packages.importr("mice")
        frame = self._prepare_frame(df)

        kwargs: dict[str, Any] = {
            "data": frame,
            "m": self.m,
            "maxit": self.max_iter,
            "printFlag": self.print_flag,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.method is not None:
            kwargs["method"] = self.method

        with rmods.conversion.localconverter(
            rmods.default_converter + rmods.pandas2ri.converter
        ):
            mids = mice.mice(**kwargs)
            completed = mice.complete(mids, action=1)
            return rmods.conversion.get_conversion().rpy2py(completed)

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy(deep=True)
        for column in frame.columns:
            if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(
                frame[column]
            ):
                frame[column] = frame[column].astype("category")
        return frame

    def _restore_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        restored = df.copy(deep=True)
        for column, dtype in self._fit_dtypes.items():
            try:
                restored[column] = restored[column].astype(dtype)
            except (TypeError, ValueError):
                if pd.api.types.is_numeric_dtype(dtype):
                    restored[column] = pd.to_numeric(restored[column], errors="coerce")
        return restored

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy(deep=True)
        array = np.asarray(X)
        columns = [f"x{i}" for i in range(array.shape[1])]
        return pd.DataFrame(array, columns=columns)

    def _matches_fit_input(self, df: pd.DataFrame) -> bool:
        return (
            list(df.columns) == self._fit_columns
            and df.shape == self._fit_input.shape
            and df.equals(self._fit_input)
        )

    def _import_r_modules(self) -> _RModules:
        try:
            from rpy2.robjects import conversion, default_converter, pandas2ri, packages
        except ImportError as exc:
            raise ImportError(
                "RMiceImputer requires an R installation plus the Python package "
                "`rpy2`, and the R package `mice`."
            ) from exc
        return _RModules(
            conversion=conversion,
            default_converter=default_converter,
            pandas2ri=pandas2ri,
            packages=packages,
        )
