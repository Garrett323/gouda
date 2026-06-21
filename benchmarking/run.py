import yaml
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from collections.abc import Generator
from gouda import *
from swiss_cheese import MCAR, MAR, MNARrs


class Experiment:
    def __init__(self, params, defaults)-> None:
        self.process(params, defaults)
        self.name = params["name"]
        self.model = get_model(params["model"])
        assert self.params["missing_mechanism"] is not None, \
            "provide a missingness mechanism"

    def process(self, params, defaults):
        self.params = _deep_update(defaults, params)

    def run(self):
        error = {}
        for ds in self.params["datasets"]:
            df = self.fetch_data(ds)
            error[ds] = {}
            for p in self.params["missing_rates"]:
                error[ds][p] = 0.0
                for seed in self.params["seeds"]:
                    missing = self.make_missing(df, p, seed)
                    model = self.model(**self.params["model_params"], encoding="label").fit(missing)
                    imputed = model.transform(missing)
                    error[ds][p] += self.compute_error(df, imputed)
                error[ds][p] /= len(self.params["seeds"])
        self.to_disk(error)

    def make_missing(self, data, missing_rate, seed=None):
        match self.params["missing_mechanism"]:
            case "mcar":
                return MCAR(random_seed=seed)(data,missing_rate)
            case "mnar":
                return MNARrs(**self.params["missing_params"], seed=seed)(data,missing_rate)
            case "mar":
                return MAR(**self.params["missing_params"], seed=seed)(data,missing_rate)
            case _:
                raise NotImplementedError

    def compute_error(self, ground_truth, imputed) -> float:
        # compute numerical error
        num_gt = ground_truth[self.num_cols]
        num_imputed = imputed[self.num_cols]
        # range normalize
        min = num_gt.min()
        max = num_gt.max()
        num_gt = (num_gt - min) / max
        num_imputed = (num_imputed - min) / max
        nmse_error = ((num_gt - num_imputed) ** 2).sum().sum()

        # compute categorical error
        cat_gt = ground_truth[self.num_cols]
        cat_imputed = imputed[self.num_cols]
        mask = cat_gt == cat_imputed
        cat_acc = mask.sum().sum() / mask.shape[0]

        return cat_acc + nmse_error

    def to_disk(self, error):
        os.makedirs(path := f"Results/{self.name}")
        with open(f"{path}/error.yaml") as f:
            yaml.dump(error, f)



    def fetch_data(self, id: int):
        data = fetch_ucirepo(id=id)
        self.current_dataset = data["metadata"]["name"]
        df: pd.DataFrame = data["data"]["features"]
        self.num_cols = df.select_dtypes(include="number").columns
        self.cat_cols = df.select_dtypes(exclude="number").columns
        return df


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
        case "simple":
            return SimpleImputer
        case "missforest":
            return MissForest
        case _:
            raise ValueError


if __name__ == "__main__":
    for e in make_experiments("config.yaml"):
        e.run()

