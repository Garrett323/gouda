"""Reproducible benchmark runner for Gouda and reference imputers.

The runner writes one raw row per seed/repetition and a tidy summary.  Raw
measurements are deliberately retained so journal figures and statistical
analyses do not have to reconstruct observations from means and standard
deviations.
"""

from __future__ import annotations

import argparse
from ast import arg
import gc
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Generator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd
import yaml
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer as KNNsk, SimpleImputer
from ucimlrepo import fetch_ucirepo

from gouda import KnnImputer, Mice, SVMImputer, SimpleImputer


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_OUTPUT = HERE / "results"
# Nones will be imported on the fly; due to dependencies
MODEL_REGISTRY = {
    "mice": Mice,
    "knn": KnnImputer,
    "svm": SVMImputer,
    "simple": SimpleImputer,
    "gain": None,
    # Gouda's MissForest is intentionally excluded
    # BASELINES
    "iterative": IterativeImputer,
    "knn-sk": KNNsk,
    "simple-sk": SimpleImputer,
    "missforest-py": None
}
LOGGER = logging.getLogger(__name__)


def _deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def make_experiments(path: Path) -> Generator[dict[str, Any], None, None]:
    with path.open(encoding="utf-8") as handle:
        documents = list(yaml.safe_load_all(handle))
    if len(documents) < 2:
        raise ValueError(
            "Configuration needs one defaults document and at least one experiment")
    defaults = documents[0]
    for experiment in documents[1:]:
        if experiment:
            yield _deep_update(defaults, experiment)


def get_model(name: str):
    match name:
        case "gain":
            try:
                from gouda import GAIN
            except ImportError as exc:
                raise RuntimeError(
                    "GAIN benchmarks require: uv sync --extra deep") from exc
            return GAIN
        case "missforest-py":
            try:
                from missforest import MissForest as MissForestPy
            except ImportError as exc:
                raise RuntimeError(
                    "missforest-py benchmarks require the benchmark development dependencies") from exc
            return MissForestPy
        case _:
            try:
                return MODEL_REGISTRY[name]
            except KeyError as exc:
                choices = ", ".join(
                    sorted([*MODEL_REGISTRY, "gain", "missforest-py"]))
                raise ValueError(f"Unknown model {name!r}; choose one of: {
                                 choices}") from exc


def _safe_std(values: pd.Series) -> float:
    if len(values) == 0:
        return np.nan
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarise independent seeds; repetitions are first reduced per seed."""
    keys = [
        "experiment", "model", "dataset_id", "dataset", "n_rows", "n_features",
        "mechanism", "missing_rate",
    ]
    per_seed = (
        raw.groupby(keys + ["seed"], as_index=False, dropna=False)
        .agg(
            numerical_nrmse=("numerical_nrmse", "first"),
            categorical_pfc=("categorical_pfc", "first"),
            missing_fraction=("missing_fraction", "first"),
            fit_seconds=("fit_seconds", "median"),
            transform_seconds=("transform_seconds", "median"),
            total_seconds=("total_seconds", "median"),
        )
    )
    rows: list[dict[str, Any]] = []
    metrics = [
        "numerical_nrmse",
        "categorical_pfc",
        "missing_fraction",
        "fit_seconds",
        "transform_seconds",
        "total_seconds",
    ]
    for group_values, group in per_seed.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, group_values))
        base["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            values = group[metric].dropna()
            base[f"{metric}_mean"] = float(
                values.mean()) if len(values) else np.nan
            base[f"{metric}_std"] = _safe_std(values)
            base[f"{metric}_median"] = float(
                values.median()) if len(values) else np.nan
        rows.append(base)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


class Experiment:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.name = str(params["name"])
        self.model_name = str(params["model"])
        self.model = get_model(self.model_name)
        if params.get("missing_mechanism") not in {"mcar", "mar", "mnar"}:
            raise ValueError(
                f"{self.name}: missing_mechanism must be mcar, mar, or mnar")
        self.model_params = params.get("model_params") or {}

    def run(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for dataset_id in self.params["datasets"]:
            ground_truth, dataset_name = self.fetch_data(int(dataset_id))
            if self.params.get("no_cat", False) and self._has_categoricals(ground_truth):
                LOGGER.info(f"[{self.name}] skipping mixed dataset {
                            dataset_name!r}")
                continue
            LOGGER.info(f"[{self.name}] {dataset_name}")
            for rate in self.params["missing_rates"]:
                LOGGER.info(f"[{self.name}] Missing Rate: {rate}")
                total_seeds = len(self.params["seeds"])
                # LOGGER.info(f"[{self.name}] Number of total Seed: {total_seeds}")
                for seed_count, seed in enumerate(self.params["seeds"]):
                    LOGGER.info(f"[{self.name}] Seed: {seed_count}/{total_seeds}")
                    missing, mask = self.make_missing(
                        ground_truth, float(rate), int(seed))
                    for _ in range(int(self.params.get("n_warmups", 1))):
                        self._new_model(int(seed)).fit(
                            missing).transform(missing)

                    seed_timings: list[tuple[float, float]] = []
                    imputed = None
                    for repetition in range(int(self.params["n_repetitions"])):
                        gc.collect()
                        gc.disable()
                        try:
                            start = time.perf_counter_ns()
                            fitted = self._new_model(int(seed)).fit(missing)
                            fit_ns = time.perf_counter_ns() - start
                            start = time.perf_counter_ns()
                            candidate = fitted.transform(missing)
                            transform_ns = time.perf_counter_ns() - start
                        finally:
                            gc.enable()
                        self.validate_output(ground_truth, candidate, mask)
                        if imputed is None:
                            imputed = candidate
                        seed_timings.append((fit_ns / 1e9, transform_ns / 1e9))

                    metrics = self.compute_metrics(ground_truth, imputed, mask)
                    for repetition, (fit_s, transform_s) in enumerate(seed_timings):
                        rows.append({
                            "experiment": self.name,
                            "model": self.model_name,
                            "dataset_id": int(dataset_id),
                            "dataset": dataset_name,
                            "n_rows": int(ground_truth.shape[0]),
                            "n_features": int(ground_truth.shape[1]),
                            "mechanism": self.params["missing_mechanism"],
                            "missing_rate": float(rate),
                            "missing_fraction": float(mask.to_numpy().mean()),
                            "seed": int(seed),
                            "repetition": repetition,
                            "fit_seconds": fit_s,
                            "transform_seconds": transform_s,
                            "total_seconds": fit_s + transform_s,
                            **metrics,
                        })
        return pd.DataFrame(rows)

    def _new_model(self, seed: int):
        """Construct a model and seed it when its sklearn API exposes a seed."""
        model = self.model(**self.model_params)
        if hasattr(model, "get_params") and hasattr(model, "set_params"):
            available = model.get_params(deep=False)
            if "random_state" in available and "random_state" not in self.model_params:
                model.set_params(random_state=seed)
        return model

    def fetch_data(self, dataset_id: int) -> tuple[pd.DataFrame, str]:
        data = fetch_ucirepo(id=dataset_id)
        frame = data["data"]["features"].copy()
        return frame, str(data["metadata"]["name"])

    def make_missing(self, data: pd.DataFrame, rate: float, seed: int):
        mechanism = self.params["missing_mechanism"]
        mechanism_params = self.params.get("missing_params") or {}
        try:
            from swiss_cheese import MAR, MNAR, MCAR
        except (ImportError, SyntaxError) as exc:
            raise RuntimeError(
                "Missing value generation does require a swiss-cheese version compatible with this Python; "
            ) from exc
        factory = {"mar": MAR, "mnar": MNAR, "mcar": MCAR}[mechanism]
        missing = factory(**mechanism_params,
                          random_seed=seed)(data.copy(), rate)
        mask = missing.isna() & data.notna()
        if not mask.to_numpy().any():
            raise RuntimeError(f"No values were removed for rate={
                               rate}, seed={seed}")
        return missing, mask

    @staticmethod
    def _has_categoricals(data: pd.DataFrame) -> bool:
        return len(data.select_dtypes(exclude="number").columns) > 0

    @staticmethod
    def validate_output(ground_truth: pd.DataFrame, imputed: Any, mask: pd.DataFrame) -> None:
        output = pd.DataFrame(
            imputed, index=ground_truth.index, columns=ground_truth.columns)
        if output.shape != ground_truth.shape:
            raise ValueError(f"Imputer changed shape from {
                             ground_truth.shape} to {output.shape}")
        observed = ~mask
        numeric = ground_truth.select_dtypes(include="number").columns
        if len(numeric):
            obs = observed[numeric].to_numpy(
            ) & ground_truth[numeric].notna().to_numpy()
            if not np.allclose(
                output[numeric].to_numpy(dtype=float)[obs],
                ground_truth[numeric].to_numpy(dtype=float)[obs],
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError("Imputer modified observed numerical values")
        categorical = ground_truth.select_dtypes(exclude="number").columns
        if len(categorical):
            obs = observed[categorical].to_numpy(
            ) & ground_truth[categorical].notna().to_numpy()
            if not np.all(
                output[categorical].to_numpy()[obs]
                == ground_truth[categorical].to_numpy()[obs]
            ):
                raise ValueError(
                    "Imputer modified observed categorical values")

    @staticmethod
    def compute_metrics(ground_truth: pd.DataFrame, imputed: Any, mask: pd.DataFrame):
        output = pd.DataFrame(
            imputed, index=ground_truth.index, columns=ground_truth.columns)
        numerical = ground_truth.select_dtypes(include="number").columns
        categorical = ground_truth.select_dtypes(exclude="number").columns
        nrmse = np.nan
        if len(numerical):
            truth = ground_truth[numerical].to_numpy(dtype=float)
            estimate = output[numerical].to_numpy(dtype=float)
            selected = mask[numerical].to_numpy()
            scale = np.nanmax(truth, axis=0) - np.nanmin(truth, axis=0)
            valid = selected & np.broadcast_to(scale > 0, truth.shape)
            if valid.any():
                normalised_error = (estimate - truth) / \
                    np.where(scale > 0, scale, 1.0)
                nrmse = float(
                    np.sqrt(np.mean(np.square(normalised_error[valid]))))
        pfc = np.nan
        if len(categorical):
            selected = mask[categorical].to_numpy()
            if selected.any():
                equal = output[categorical].to_numpy(
                ) == ground_truth[categorical].to_numpy()
                pfc = float(1.0 - equal[selected].mean())
        return {"numerical_nrmse": nrmse, "categorical_pfc": pfc}


def environment_metadata(config: Path, selected: list[str] | None) -> dict[str, Any]:
    packages = ["gouda-cheese", "numpy", "pandas",
                "scikit-learn", "swiss-cheese", "torch"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parent, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=HERE.parent, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": str(config.resolve()),
        "selected_experiments": selected,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": versions,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiments", nargs="+",
                        help="Experiment names to run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for params in make_experiments(args.config):
        if args.experiments and params["name"] not in args.experiments:
            continue
        exp = Experiment(params)
        df = exp.run()
        df.to_csv(f"{args.output}/{exp.name}.csv")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        raise RuntimeError("No benchmark observations were produced")
    raw.to_csv(args.output / "raw_results.csv", index=False)
    _summary(raw).to_csv(args.output / "summary.csv", index=False)
    with (args.output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(environment_metadata(
            args.config, args.experiments), handle, indent=2)
    LOGGER.info(f"Wrote {len(raw)} observations to {args.output.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{HERE}/bench.log"),
        ],
        force=True,
    )
    try:
        LOGGER.info(
            "@@@@@@@@@@@@@@@@@@ Starting benchmark run @@@@@@@@@@@@@@@@@@@@@"
        )
        main()
    except Exception as e:
        LOGGER.error(f"Pipeline FAILED: {e}")
