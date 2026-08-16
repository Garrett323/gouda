"""Measure Gouda's strong scaling across native thread counts.

Each measurement runs in a fresh process because Rayon and BLAS thread pools
are process-global and may be initialized before Python code can resize them.
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


def parse_thread_counts(value: str | None) -> list[int]:
    if value:
        counts = sorted({int(item) for item in value.split(",")})
        if not counts or counts[0] < 1:
            raise ValueError("Thread counts must be positive integers")
        return counts
    available = os.cpu_count() or 1
    counts = [1]
    while counts[-1] * 2 <= available:
        counts.append(counts[-1] * 2)
    if counts[-1] != available:
        counts.append(available)
    return counts


def worker(args: argparse.Namespace) -> None:
    # Imports intentionally happen here. The parent sets native thread
    # environment variables before this worker process starts.
    import numpy as np

    from gouda import KnnImputer

    rng = np.random.default_rng(args.seed)
    complete = rng.normal(size=(args.rows, args.features))
    missing_mask = rng.random(complete.shape) < args.missing_rate
    missing = complete.copy()
    missing[missing_mask] = np.nan
    # Avoid unsupported all-missing columns without changing the random mask
    # elsewhere.
    for column in range(args.features):
        if missing_mask[:, column].all():
            missing[0, column] = complete[0, column]
            missing_mask[0, column] = False

    def execute():
        start = time.perf_counter_ns()
        model = KnnImputer(k=args.k).fit(missing)
        fit_seconds = (time.perf_counter_ns() - start) / 1e9
        start = time.perf_counter_ns()
        output = model.transform(missing)
        transform_seconds = (time.perf_counter_ns() - start) / 1e9
        if not np.isfinite(output).all():
            raise RuntimeError("KNN scaling benchmark produced non-finite output")
        if not np.array_equal(output[~missing_mask], complete[~missing_mask]):
            raise RuntimeError("KNN scaling benchmark modified observed values")
        return fit_seconds, transform_seconds

    for _ in range(args.warmups):
        execute()
    observations = []
    for repetition in range(args.repetitions):
        fit_seconds, transform_seconds = execute()
        observations.append({
            "repetition": repetition,
            "fit_seconds": fit_seconds,
            "transform_seconds": transform_seconds,
            "total_seconds": fit_seconds + transform_seconds,
        })
    print("GOUDASCALING=" + json.dumps(observations))


def run_parent(args: argparse.Namespace) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    counts = parse_thread_counts(args.threads)
    rows = []
    for threads in counts:
        env = os.environ.copy()
        for variable in ("RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env[variable] = str(threads)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--rows", str(args.rows), "--features", str(args.features),
            "--missing-rate", str(args.missing_rate), "--seed", str(args.seed),
            "--k", str(args.k), "--repetitions", str(args.repetitions),
            "--warmups", str(args.warmups),
        ]
        print(f"Measuring {threads} thread(s)...", flush=True)
        completed = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        marker = next(
            (line for line in reversed(completed.stdout.splitlines()) if line.startswith("GOUDASCALING=")),
            None,
        )
        if marker is None:
            raise RuntimeError(f"Worker returned no measurements:\n{completed.stdout}\n{completed.stderr}")
        for observation in json.loads(marker.removeprefix("GOUDASCALING=")):
            rows.append({
                "threads": threads,
                "rows": args.rows,
                "features": args.features,
                "missing_rate": args.missing_rate,
                "seed": args.seed,
                **observation,
            })

    raw = pd.DataFrame(rows)
    summary = raw.groupby("threads", as_index=False).agg(
        median_seconds=("total_seconds", "median"),
        mean_seconds=("total_seconds", "mean"),
        std_seconds=("total_seconds", "std"),
    )
    baseline = float(summary.loc[summary["threads"] == 1, "median_seconds"].iloc[0])
    summary["speedup"] = baseline / summary["median_seconds"]
    summary["parallel_efficiency"] = summary["speedup"] / summary["threads"]

    args.output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output / "thread_scaling_raw.csv", index=False)
    summary.to_csv(args.output / "thread_scaling_summary.csv", index=False)

    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    threads = summary["threads"].to_numpy()
    speedup = summary["speedup"].to_numpy()
    efficiency = summary["parallel_efficiency"].to_numpy() * 100
    axes[0].plot(threads, threads, "--", color="#777777", label="Ideal")
    axes[0].plot(threads, speedup, "o-", color="#0072B2", linewidth=1.8, label="Gouda KNN")
    axes[0].set(xlabel="Native threads", ylabel="Speedup vs. one thread", title="Strong scaling")
    axes[0].legend(frameon=False)
    axes[1].plot(threads, efficiency, "s-", color="#D55E00", linewidth=1.8)
    axes[1].axhline(100, linestyle="--", color="#777777")
    axes[1].set(xlabel="Native threads", ylabel="Parallel efficiency (%)", title="Scaling efficiency")
    for axis in axes:
        axis.set_xticks(threads)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    fig.suptitle(f"KNN: {args.rows:,} rows × {args.features} features, {args.missing_rate:.0%} missing")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(args.output / f"thread_scaling.{extension}", bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Wrote scaling results and figures to {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", help="Comma-separated counts; default: powers of two through CPU count")
    parser.add_argument("--rows", type=int, default=2_000)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE / "scaling_results")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
