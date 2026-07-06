from gouda.gouda import  MissForest
from gouda.wrapper import KnnImputer, SimpleImputer, ConstantImputer, SVMImputer, Mice
# gouda_cheese/__init__.py

_lazy_imports = {
    "GAIN": "gouda.deeplearning.gain",
}

def __getattr__(name):
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
# from gouda.deeplearning.gain import GAIN
#
Imputers = [KnnImputer, SVMImputer, SimpleImputer, ConstantImputer, Mice]

__all__ = ["KnnImputer", "SimpleImputer", "ConstantImputer", "Mice", "MissForest", "SVMImputer", "GAIN", "Imputers"]
