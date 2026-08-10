import yaml
import numpy as np
import os
import pandas as pd
import argparse
from ucimlrepo import fetch_ucirepo
from collections.abc import Generator
from gouda import *
from swiss_cheese import MCAR, MAR, MNAR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer as KNNsk
from missforest import MissForest as MissForestPy
import time


class Experiment:
    def __init__(self, params, defaults) -> None:
        self.process(params, defaults)
        self.name = params["name"]
        self.model = get_model(params["model"])
        assert self.params["missing_mechanism"] is not None, \
            "provide a missingness mechanism"
        if self.params.get("model_params") is None:
            self.params["model_params"] = {}

    def process(self, params, defaults):
        self.params = _deep_update(defaults, params)

    def run(self):
        error = {}
        times = {}
        for ds in self.params["datasets"]:
            print(f"Processing {ds}")
            df = self.fetch_data(ds)
            if not self.supports_cat():
                print(f"skipping.. {self.current_dataset}")
                continue
            error[self.current_dataset] = {}
            times[self.current_dataset] = {}
            for p in self.params["missing_rates"]:
                errors = []
                fits = []
                imputes = []
                for seed in self.params["seeds"]:
                    missing, mask = self.make_missing(df, p, seed)
                    _warmup = self.model(**self.params["model_params"]).fit(missing).transform(missing)
                    for _ in range(20):
                        start = time.perf_counter_ns()
                        model = self.model(
                            **self.params["model_params"]).fit(missing)
                        fits.append(time.perf_counter_ns() - start)
                        start = time.perf_counter_ns()
                        imputed = model.transform(missing)
                        imputes.append(time.perf_counter_ns() - start)
                    errors.append(self.compute_error(df, imputed, mask))
                self.add_metrics(p, error, times, errors, fits, imputes)
        self.to_disk(error, times)

    def make_missing(self, data, missing_rate, seed=None):
        match self.params["missing_mechanism"]:
            case "mcar":
                missing = MCAR(random_seed=seed)(data, missing_rate)
            case "mnar":
                missing = MNAR(**self.params["missing_params"], random_seed=seed)(data, missing_rate)
            case "mar":
                missing = MAR(**self.params["missing_params"], random_seed=seed)(data, missing_rate)
            case _:
                raise NotImplementedError
        return missing, missing.isna()

    def compute_error(self, ground_truth, imputed, missing_mask) -> float:
        if self.only_num:
            col_min = ground_truth.min()
            col_max = ground_truth.max()
            gt = (ground_truth - col_min) / (col_max - col_min)
            imputed = pd.DataFrame(imputed, columns=gt.columns, index=gt.index)
            imputed = (imputed - col_min) / (col_max - col_min)
            nmse_error = ((gt[missing_mask] - imputed[missing_mask]) ** 2).mean().mean()
            return nmse_error

        if not self.num_cols.empty:
            # compute numerical error
            num_gt = ground_truth[self.num_cols]
            num_imputed = imputed[self.num_cols]
            # range normalize
            col_min = num_gt.min()
            col_max = num_gt.max()
            num_gt = ground_truth[missing_mask][self.num_cols]
            num_imputed = imputed[missing_mask][self.num_cols]
            num_gt = (num_gt - col_min) / (col_max - col_min)
            num_imputed = (num_imputed - col_min) / (col_max - col_min)
            nmse_error = ((num_gt - num_imputed) ** 2).mean().mean()
        else:
            nmse_error = 0.0

        if not self.cat_cols.empty:
            # compute categorical error
            cat_gt = ground_truth[missing_mask][self.cat_cols]
            cat_imputed = imputed[missing_mask][self.cat_cols]
            mask = cat_gt == cat_imputed
            cat_error = 1.0 - (mask.sum().sum() / mask.size)
        else:
            cat_error = 0.0

        return cat_error + nmse_error

    def to_disk(self, error, times):
        os.makedirs(path := f"Results/{self.name}", exist_ok=True)
        with open(f"{path}/error.yaml", "w") as f:
            yaml.dump(error, f)
        with open(f"{path}/timing.yaml", "w") as f:
            yaml.dump(times, f)

    def fetch_data(self, id: int):
        data = fetch_ucirepo(id=id)
        self.current_dataset = data["metadata"]["name"]
        df: pd.DataFrame = data["data"]["features"]
        self.num_cols = df.select_dtypes(include="number").columns
        self.cat_cols = df.select_dtypes(exclude="number").columns
        self.only_num = self.cat_cols.empty
        return df

    def add_metrics(self, p, error, times, seed_errors, fit_times, impute_times):
        error[self.current_dataset][p] = {
            "mean": float(np.mean(seed_errors)),
            "std": float(np.std(seed_errors, ddof=1))
        }

        total_time = np.array(fit_times) + np.array(impute_times)
        times[self.current_dataset][p] = {
            "fit": {
                "mean": float(np.mean(fit_times)),
                "median": float(np.median(fit_times)),
                "std": float(np.std(fit_times, ddof=1))
            },
            "impute": {
                "mean": float(np.mean(impute_times)),
                "median": float(np.median(impute_times)),
                "std": float(np.std(impute_times, ddof=1))
            },
            "total": {
                "mean": float(np.mean(total_time)),
                "median": float(np.median(total_time)),
                "std": float(np.std(total_time, ddof=1))
            }
        }


    def supports_cat(self):
        if self.params["no_cat"] and not self.only_num:
            return False
        return True


def _deep_update(base, override):
    result = base.copy()
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def make_experiments(path: str) -> Generator[Experiment]:
    with open(path, "r") as f:
        conf = yaml.safe_load_all(f)
        default = next(conf)
    return (Experiment(defaults=default, params=c) for c in conf)


def get_model(model):
    match model:
        case "mice":
            return Mice
        case "knn":
            return KnnImputer
        case "svm":
            return SVMImputer
        case "gain":
            return GAIN
        case "knn-sk":
            return KNNsk
        case "simple":
            return SimpleImputer
        case "missforest":
            return MissForest
        case "missforest-py":
            return MissForestPy
        case "iterative":
            return IterativeImputer
        case _:
            raise ValueError


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-e",
        nargs="+",        # accepts one or more values
        metavar="ARG",
        help="List of experiments"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.e)  # args.e is a list
    if args.e is not None:
        print("Running only selected experiments..")
    for e in make_experiments("config.yaml"):
        if args.e is not None:
            if e.name not in args.e:
                print(f"skipping {e.name}")
                continue
        print(f"Starting {e.name}..")
        e.run()
        print("Done..")
