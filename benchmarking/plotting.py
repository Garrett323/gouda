"""Create publication-quality benchmark figures from ``summary.csv``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v"]


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _methods(data: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(data["experiment"].astype(str)))


def _plot_metric(ax, data: pd.DataFrame, mean: str, std: str, ylabel: str, log=False) -> None:
    for index, method in enumerate(_methods(data)):
        subset = data[data["experiment"] == method].sort_values("missing_rate")
        x = subset["missing_rate"].to_numpy(dtype=float) * 100
        y = subset[mean].to_numpy(dtype=float)
        spread = subset[std].fillna(0).to_numpy(dtype=float)
        color = PALETTE[index % len(PALETTE)]
        ax.plot(x, y, color=color, marker=MARKERS[index % len(MARKERS)], label=method, linewidth=1.8)
        ax.fill_between(x, np.maximum(y - spread, 0), y + spread, color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    if log:
        ax.set_yscale("log")


def save_figure(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_dataset(data: pd.DataFrame, dataset: str, output: Path) -> None:
    subset = data[data["dataset"] == dataset]
    panels = []
    if subset["numerical_nrmse_mean"].notna().any():
        panels.append(("numerical_nrmse_mean", "numerical_nrmse_std", "Numerical NRMSE", "Numerical error", False))
    if "categorical_pfc_mean" in subset and subset["categorical_pfc_mean"].notna().any():
        panels.append(("categorical_pfc_mean", "categorical_pfc_std", "Categorical PFC", "Categorical error", False))
    panels.append(("total_seconds_median", "total_seconds_std", "Fit + transform time (s)", "Runtime (log scale)", True))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.0), squeeze=False)
    axes = axes[0]
    for ax, (mean, std, ylabel, title, log) in zip(axes, panels):
        _plot_metric(ax, subset, mean, std, ylabel, log=log)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=min(4, len(labels)))
    fig.suptitle(dataset, y=1.16, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output, f"{dataset.lower().replace(' ', '_')}_benchmark")


def plot_overview(data: pd.DataFrame, output: Path) -> None:
    """Show accuracy/runtime trade-off, averaged by method and dataset."""
    view = data.dropna(subset=["numerical_nrmse_mean", "total_seconds_median"]).copy()
    if view.empty:
        return
    aggregated = view.groupby("experiment", as_index=False).agg(
        numerical_nrmse=("numerical_nrmse_mean", "mean"),
        runtime=("total_seconds_median", "median"),
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for index, row in aggregated.iterrows():
        color = PALETTE[index % len(PALETTE)]
        ax.scatter(row["runtime"], row["numerical_nrmse"], s=48, color=color, marker=MARKERS[index % len(MARKERS)])
        ax.annotate(str(row["experiment"]), (row["runtime"], row["numerical_nrmse"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Median fit + transform time (s, log scale)")
    ax.set_ylabel("Mean numerical NRMSE")
    ax.set_title("Accuracy–runtime trade-off")
    ax.grid(color="#d9d9d9", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, output, "accuracy_runtime_tradeoff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path, default=HERE / "results" / "summary.csv")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    parser.add_argument("--dataset", action="append", help="Dataset to plot; default: all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    data = pd.read_csv(args.summary)
    required = {"experiment", "dataset", "missing_rate", "numerical_nrmse_mean", "total_seconds_median"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {', '.join(sorted(missing))}")
    datasets = args.dataset or sorted(data["dataset"].unique())
    for dataset in datasets:
        plot_dataset(data, dataset, args.output)
    plot_overview(data, args.output)
    print(f"Wrote figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
