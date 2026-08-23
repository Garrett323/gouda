"""Measure runtime scaling across dataset sizes and native thread counts.

Each thread count runs in a fresh process because Rayon and BLAS thread pools
are process-global. The benchmark compares Gouda's KNN, MICE, and simple
imputers on identical synthetic numerical data and plots both absolute runtime
and speedup over the single-thread result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODEL_NAMES = ("knn", "mice", "simple")
MODEL_LABELS = {"knn": "KNN", "mice": "MICE", "simple": "Simple"}


def parse_positive_counts(value: str, name: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges such as ``1-16``."""
    counts: set[int] = set()
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_text, end_text = item.split("-", maxsplit=1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError
                counts.update(range(start, end + 1))
            else:
                counts.add(int(item))
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain positive integers or inclusive ranges"
        ) from exc
    ordered = sorted(counts)
    if not ordered or ordered[0] < 1:
        raise ValueError(f"{name} must contain positive integers")
    return ordered


def parse_thread_counts(value: str | None) -> list[int]:
    if value:
        counts = parse_positive_counts(value, "--threads")
    else:
        counts = list(range(1, 17))
    if 1 not in counts:
        raise ValueError("Thread counts must include 1 to calculate speedup")
    return counts


def parse_row_counts(value: str) -> list[int]:
    return parse_positive_counts(value, "--rows")


def parse_models(value: str) -> list[str]:
    models = list(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    invalid = sorted(set(models) - set(MODEL_NAMES))
    if invalid:
        raise ValueError(
            f"Unknown models: {', '.join(invalid)}; "
            f"choose from {', '.join(MODEL_NAMES)}"
        )
    if not models:
        raise ValueError("At least one model is required")
    return models


def worker(args: argparse.Namespace) -> None:
    # Imports intentionally happen here, after the parent has set the native
    # thread environment for this fresh worker process.
    import numpy as np

    from gouda import KnnImputer, Mice, SimpleImputer

    row_counts = parse_row_counts(args.rows)
    selected_models = parse_models(args.models)
    factories = {
        "knn": lambda: KnnImputer(k=args.k),
        "mice": lambda: Mice(max_iter=args.mice_iterations),
        "simple": SimpleImputer,
    }

    rng = np.random.default_rng(args.seed)
    complete = rng.normal(size=(row_counts[-1], args.features))
    missing_mask = rng.random(complete.shape) < args.missing_rate

    # Ensure every tested prefix has an observed value in every column.
    for column in range(args.features):
        if missing_mask[: row_counts[0], column].all():
            missing_mask[0, column] = False
    # Avoid rows without a usable observed value.
    missing_mask[missing_mask.all(axis=1), 0] = False

    missing = complete.copy()
    missing[missing_mask] = np.nan

    def execute(model_name: str, rows: int) -> tuple[float, float]:
        model_input = missing[:rows]
        expected = complete[:rows]
        mask = missing_mask[:rows]

        start = time.perf_counter_ns()
        model = factories[model_name]().fit(model_input)
        fit_seconds = (time.perf_counter_ns() - start) / 1e9
        start = time.perf_counter_ns()
        output = np.asarray(model.transform(model_input))
        transform_seconds = (time.perf_counter_ns() - start) / 1e9

        if output.shape != expected.shape:
            raise RuntimeError(
                f"{model_name} changed shape from {expected.shape} to {output.shape}"
            )
        if not np.isfinite(output).all():
            raise RuntimeError(f"{model_name} produced non-finite output")
        if not np.allclose(
            output[~mask], expected[~mask], rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(f"{model_name} modified observed values")
        return fit_seconds, transform_seconds

    observations = []
    for rows in row_counts:
        for model_name in selected_models:
            for _ in range(args.warmups):
                execute(model_name, rows)
            for repetition in range(args.repetitions):
                fit_seconds, transform_seconds = execute(model_name, rows)
                observations.append({
                    "model": model_name,
                    "rows": rows,
                    "repetition": repetition,
                    "fit_seconds": fit_seconds,
                    "transform_seconds": transform_seconds,
                    "total_seconds": fit_seconds + transform_seconds,
                })
    print("GOUDASCALING=" + json.dumps(observations))


def run_parent(args: argparse.Namespace) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    thread_counts = parse_thread_counts(args.threads)
    row_counts = parse_row_counts(args.rows)
    selected_models = parse_models(args.models)
    rows = []

    for threads in thread_counts:
        env = os.environ.copy()
        for variable in (
            "RAYON_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            env[variable] = str(threads)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--threads",
            str(threads),
            "--rows",
            args.rows,
            "--features",
            str(args.features),
            "--missing-rate",
            str(args.missing_rate),
            "--seed",
            str(args.seed),
            "--k",
            str(args.k),
            "--mice-iterations",
            str(args.mice_iterations),
            "--models",
            args.models,
            "--repetitions",
            str(args.repetitions),
            "--warmups",
            str(args.warmups),
        ]
        print(f"Measuring {threads} thread(s)...", flush=True)
        completed = subprocess.run(
            command, env=env, check=False, capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Worker for {threads} thread(s) failed:\n{completed.stdout}\n"
                f"{completed.stderr}"
            )
        marker = next(
            (
                line
                for line in reversed(completed.stdout.splitlines())
                if line.startswith("GOUDASCALING=")
            ),
            None,
        )
        if marker is None:
            raise RuntimeError(
                f"Worker returned no measurements:\n{completed.stdout}\n"
                f"{completed.stderr}"
            )
        for observation in json.loads(marker.removeprefix("GOUDASCALING=")):
            rows.append({
                "threads": threads,
                "features": args.features,
                "missing_rate": args.missing_rate,
                "seed": args.seed,
                **observation,
            })

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["model", "threads", "rows"], as_index=False)
        .agg(
            repetitions=("repetition", "count"),
            median_seconds=("total_seconds", "median"),
            mean_seconds=("total_seconds", "mean"),
            std_seconds=("total_seconds", "std"),
        )
        .sort_values(["model", "threads", "rows"])
        .reset_index(drop=True)
    )
    summary["std_seconds"] = summary["std_seconds"].fillna(0.0)

    baselines = summary[summary["threads"] == 1].set_index(
        ["model", "rows"]
    )["median_seconds"]
    summary["speedup"] = [
        float(baselines.loc[(model, row_count)]) / runtime
        for model, row_count, runtime in zip(
            summary["model"], summary["rows"], summary["median_seconds"]
        )
    ]
    summary["parallel_efficiency"] = summary["speedup"] / summary["threads"]

    args.output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output / "thread_scaling_raw.csv", index=False)
    summary.to_csv(args.output / "thread_scaling_summary.csv", index=False)

    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(
        2,
        len(selected_models),
        figsize=(3.6 * len(selected_models), 6.0),
        squeeze=False,
        sharex="col",
    )
    normalizer = mpl.colors.Normalize(
        vmin=min(thread_counts), vmax=max(thread_counts)
    )
    color_map = mpl.colormaps["viridis"]

    for column, model_name in enumerate(selected_models):
        model_data = summary[summary["model"] == model_name]
        for threads in thread_counts:
            values = model_data[model_data["threads"] == threads].sort_values("rows")
            color = color_map(normalizer(threads))
            axes[0, column].plot(
                values["rows"],
                values["median_seconds"],
                "o-",
                color=color,
                markersize=3,
                linewidth=1.2,
            )
            axes[1, column].plot(
                values["rows"],
                values["speedup"],
                "o-",
                color=color,
                markersize=3,
                linewidth=1.2,
            )

        axes[0, column].set_title(MODEL_LABELS[model_name])
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel("Median fit + transform time (s)")
        axes[1, column].set_ylabel("Speedup vs. one thread")
        axes[1, column].set_xlabel("Number of rows")
        axes[1, column].axhline(
            1.0, linestyle="--", color="#888888", linewidth=0.8
        )
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.set_xticks(row_counts)
            axis.xaxis.set_major_formatter(
                mpl.ticker.StrMethodFormatter("{x:,.0f}")
            )
            axis.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            axis.grid(color="#d9d9d9", linewidth=0.6, which="both")

    scalar_mappable = mpl.cm.ScalarMappable(norm=normalizer, cmap=color_map)
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axes,
        ticks=thread_counts,
        fraction=0.025,
        pad=0.03,
    )
    colorbar.set_label("Native threads")
    figure.suptitle(
        f"Size and thread scaling: {args.features} features, "
        f"{args.missing_rate:.0%} missing"
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.9,
        bottom=0.09,
        top=0.9,
        wspace=0.35,
        hspace=0.12,
    )
    for extension in ("pdf", "png"):
        figure.savefig(
            args.output / f"thread_scaling.{extension}", bbox_inches="tight"
        )
    plt.close(figure)

    print(summary.to_string(index=False))
    print(f"Wrote scaling results and figures to {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threads",
        help="Thread counts/ranges (default: every count from 1 through 16)",
    )
    parser.add_argument(
        "--rows",
        default="100,250,500,1000,2000",
        help="Row counts/ranges (default: 100,250,500,1000,2000)",
    )
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mice-iterations", type=int, default=10)
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help=f"Comma-separated models (default: {','.join(MODEL_NAMES)})",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE / "scaling_results")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.worker:
        worker_threads = parse_positive_counts(args.threads or "", "--threads")
        if len(worker_threads) != 1:
            raise ValueError("A worker requires exactly one thread count")
    else:
        parse_thread_counts(args.threads)
    parse_row_counts(args.rows)
    parse_models(args.models)
    if args.features < 1:
        raise ValueError("--features must be positive")
    if not 0.0 < args.missing_rate < 1.0:
        raise ValueError("--missing-rate must be between 0 and 1")
    if args.k < 1:
        raise ValueError("--k must be positive")
    if args.mice_iterations < 1:
        raise ValueError("--mice-iterations must be positive")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.worker:
        worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
