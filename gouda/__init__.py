from gouda.gouda import MissForest
from gouda.wrapper import KnnImputer, SimpleImputer, ConstantImputer, SVMImputer, Mice

_lazy_imports = {
    "GAIN": "gouda.deeplearning.gain",
}

def __getattr__(name):
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

Imputers = [KnnImputer, SVMImputer, SimpleImputer, ConstantImputer, Mice]
# Add GAIN if the optional dependency is installed
try:
    GAIN = __getattr__("GAIN")
    Imputers.append(GAIN)
except (ImportError, ModuleNotFoundError):
    pass


__all__ = ["KnnImputer", "SimpleImputer", "ConstantImputer", "Mice", "MissForest", "SVMImputer", "GAIN", "Imputers"]
