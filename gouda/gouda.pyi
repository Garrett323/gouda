import numpy
import numpy.typing
from typing import Literal

class ConstantImputer:
    def __init__(self, constant: float) -> None: ...
    @staticmethod
    def zero() -> ConstantImputer:
        ...
    def fit(self, _data:numpy.typing.NDArray[numpy.float64]) -> ConstantImputer:
        ...

    def transform(self, data:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        ...


class KnnImputer:
    k: int
    def __init__(self, k:int=5, metric:str="nan_euclid", weights:str="uniform") -> None: ...
    def fit(self, data:numpy.typing.NDArray[numpy.float64]) -> KnnImputer:
        ...

    def transform(self, data:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        ...


class SimpleImputer:
    def fit(self, data:numpy.typing.NDArray[numpy.float64]) -> SimpleImputer:
        ...

    def transform(self, data:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        ...


class Mice:
    n_iterations: int = 15
    backend: Literal["linear", "ridge", "pmm"] = "linear"
    alpha: float = 1.0
    def __init__(self, max_iter: int = 10, backend: Literal["linear", "ridge", "pmm"]= "linear", alpha: float = 1.0, pmm_backend: Literal["linear", "ridge"] = "linear") -> None: ...
    def fit(self, data:numpy.typing.NDArray[numpy.float64]) -> Mice:
        ...

    def transform(self, data:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        ...


class MissForest:
    is_fitted: bool
    n_trees: int
    max_depth: int
    seed: int | None
    min_samples_leaf: int

    def __init__(self, n_trees: int = 15, max_depth: int = 15, min_samples_leaf: int = 5, seed: int | None = None) -> None: ...
    def fit(self, data:numpy.typing.NDArray[numpy.float64]) -> Mice:
        ...

    def transform(self, data:numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        ...
