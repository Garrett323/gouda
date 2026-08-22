"""Combine per-experiment benchmark files into raw and summary CSV files.

By default, this script reads ``results/*.csv`` (and legacy extensionless
files), excluding files that it generates itself. Repetitions are reduced to
their median within each seed before statistics are calculated across seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"

GROUP_COLUMNS = [
    "experiment",
    "model",
    "dataset_id",
    "dataset",
    "n_rows",
    "n_features",
    "mechanism",
    "missing_rate",
]
METRICS = [
    "numerical_nrmse",
    "categorical_pfc",
    "missing_fraction",
    "fit_seconds",
    "transform_seconds",
    "total_seconds",
]
REQUIRED_COLUMNS = GROUP_COLUMNS + ["seed", "repetition"] + METRICS
GENERATED_NAMES = {
    "raw_results.csv",
    "summary.csv",
    "thread_scaling_raw.csv",
    "thread_scaling_summary.csv",
}


def _read_result(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    # Individual files written by older versions of run.py contain the
    # DataFrame index as an unnamed CSV column.
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)

    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path}: missing required columns: {names}")
    if frame.empty:
        raise ValueError(f"{path}: contains no benchmark rows")
    return frame


def discover_inputs(results_dir: Path) -> list[Path]:
    """Find experiment CSVs, including legacy files without a suffix."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    paths = [
        path
        for path in results_dir.iterdir()
        if path.is_file()
        and path.name not in GENERATED_NAMES
        and (path.suffix.lower() == ".csv" or not path.suffix)
    ]
    return sorted(paths, key=lambda path: path.name)


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarise independent seeds after reducing timing repetitions."""
    per_seed = (
        raw.groupby(GROUP_COLUMNS + ["seed"], as_index=False, dropna=False)
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
    groups = per_seed.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    for group_values, group in groups:
        row = dict(zip(GROUP_COLUMNS, group_values))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in METRICS:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else (0.0 if len(values) == 1 else float("nan"))
            )
            row[f"{metric}_median"] = (
                float(values.median()) if len(values) else float("nan")
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(GROUP_COLUMNS).reset_index(drop=True)


def combine(inputs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not inputs:
        raise ValueError("No per-experiment result files were found")

    raw = pd.concat([_read_result(path) for path in inputs], ignore_index=True)
    order = GROUP_COLUMNS + ["seed", "repetition"]
    raw = raw.sort_values(order, kind="stable").reset_index(drop=True)
    return raw, summarise(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Experiment result files; default: discover them in --results-dir",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Input/output directory (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Raw output path (default: RESULTS_DIR/raw_results.csv)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Summary output path (default: RESULTS_DIR/summary.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs or discover_inputs(args.results_dir)
    raw_output = args.raw_output or args.results_dir / "raw_results.csv"
    summary_output = args.summary_output or args.results_dir / "summary.csv"

    raw, summary = combine(inputs)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_output, index=False)
    summary.to_csv(summary_output, index=False)

    print(f"Combined {len(inputs)} files and {len(raw)} rows")
    print(f"Wrote {raw_output.resolve()}")
    print(f"Wrote {summary_output.resolve()}")


if __name__ == "__main__":
    main()
