import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Literal, Self


class ConstantImputer:
    def __init__(self, constant: float,
                 encoding: None | Literal["label"] = None
                 ) -> None: ...

    @staticmethod
    def zero() -> "ConstantImputer":
        ...

    def fit(self, _data: npt.NDArray[np.float64] | pd.DataFrame) -> Self:
        ...

    def transform(self, data: npt.NDArray[np.float64] | pd.DataFrame) \
            -> npt.NDArray[np.float64]:
        ...


class KnnImputer:
    k: int

    def __init__(self, k: int = 5, metric: str = "nan_euclid",
                 weights: str = "uniform",
                 encoding: None | Literal["label"] = None
                 ) -> None: ...

    def fit(self, data: npt.NDArray[np.float64] | pd.DataFrame) -> Self:
        ...

    def transform(self, data: npt.NDArray[np.float64] | pd.DataFrame) \
            -> npt.NDArray[np.float64]:
        ...


class SimpleImputer:
    def __init__(self,
                 encoding: None | Literal["label"] = None
                 ) -> None: ...

    def fit(self, data: npt.NDArray[np.float64] | pd.DataFrame) -> Self:
        ...

    def transform(self, data: npt.NDArray[np.float64] | pd.DataFrame) \
            -> npt.NDArray[np.float64]:
        ...


class Mice:
    n_iterations: int = 15
    backend: Literal["linear", "ridge", "pmm"] = "linear"
    alpha: float = 1.0

    def __init__(self, max_iter: int = 10,
                 backend: Literal["linear", "ridge", "pmm"] = "linear",
                 alpha: float = 1.0,
                 pmm_backend: Literal["linear", "ridge"] = "linear",
                 encoding: None | Literal["label"] = None
                 ) -> None: ...

    def fit(self, data: npt.NDArray[np.float64] | pd.DataFrame) -> Self:
        ...

    def transform(self, data: npt.NDArray[np.float64] | pd.DataFrame) \
            -> npt.NDArray[np.float64]:
        ...


class MissForest:
    is_fitted: bool
    n_trees: int
    max_depth: int
    seed: int | None
    min_samples_leaf: int

    def __init__(self, n_trees: int = 15, max_depth: int = 15,
                 min_samples_leaf: int = 5, seed: int | None = None,
                 encoding: None | Literal["label"] = None
                 ) -> None: ...

    def fit(self, data: npt.NDArray[np.float64] | pd.DataFrame) \
            -> Self:
        ...

    def transform(self, data: npt.NDArray[np.float64] | pd.DataFrame) \
            -> npt.NDArray[np.float64]:
        ...
