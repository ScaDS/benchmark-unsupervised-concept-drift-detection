#!/usr/bin/env python3
"""Analyze metric stability across seeds from multi-seed experiment results.

The multi-seed runner writes one ``per_seed_metrics.csv`` file per
 detector/dataset combination.  This script summarizes the variation across
 seeds for accuracy, runtime, and Pareto hypervolume, and also summarizes the
 cross-seed correlation matrices written by the runner.

Usage:
    python analyze_multiseed_reproducibility.py \
        --input-dir multi_seed_results --plot plots/multi_seed
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


METRICS = ("accuracy_mean", "runtime_mean", "hypervolume")
METRIC_LABELS = {
    "accuracy_mean": "Accuracy",
    "runtime_mean": "Runtime",
    "hypervolume": "Hypervolume",
}


def finite_values(values: Iterable[object]) -> List[float]:
    """Return numeric, finite values."""
    result = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def percentile(values: List[float], q: float) -> float:
    """Return a percentile or NaN for an empty collection."""
    return float(np.percentile(values, q)) if values else float("nan")


def metric_statistics(values: Iterable[object]) -> Dict[str, float]:
    """Compute the statistics used for each metric across seeds."""
    valid = finite_values(values)
    if not valid:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "cv": float("nan"),
        }

    mean = float(np.mean(valid))
    std = float(np.std(valid))
    return {
        "n": len(valid),
        "mean": mean,
        "std": std,
        "median": float(np.median(valid)),
        "p10": percentile(valid, 10),
        "p25": percentile(valid, 25),
        "p75": percentile(valid, 75),
        "p90": percentile(valid, 90),
        "min": min(valid),
        "max": max(valid),
        "cv": std / abs(mean) if mean != 0 else float("nan"),
    }


KNOWN_DETECTORS = {
    "BNDM", "CDBD", "CDLEEDS", "CSDDM", "D3", "DAWIDD", "DDAL", "EDFS",
    "HDDDM", "IBDD", "IKS", "NNDVI", "OCDD", "PCACD", "SlidShaps", "SPLL",
    "STUDD", "UCDD", "UDetect", "WindowKDE",
}


def split_combination_name(name: str) -> tuple[str, str]:
    """Split a 'Detector_Dataset' directory name into (detector, dataset).

    Uses the known set of detector names to find the correct split point.
    Falls back to splitting on the first underscore if no known detector matches.
    """
    for det in sorted(KNOWN_DETECTORS, key=len, reverse=True):
        prefix = det + "_"
        if name.startswith(prefix):
            return det, name[len(prefix):]
    # Fallback: split on first underscore
    if "_" in name:
        idx = name.index("_")
        return name[:idx], name[idx + 1:]
    return name, name


def load_results(input_dir: Path) -> pd.DataFrame:
    """Load all per-seed metric files and add detector/dataset identifiers."""
    files = sorted(input_dir.glob("*/per_seed_metrics.csv"))
    if not files:
        print(f"Warning: no */per_seed_metrics.csv files found in {input_dir}")
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
            combination = path.parent.name
            frame["combination"] = combination
            if "detector" in frame.columns:
                frame["detector"] = frame["detector"].fillna(combination)
            else:
                det, _ = split_combination_name(combination)
                frame["detector"] = det
            if "dataset" in frame.columns:
                frame["dataset"] = frame["dataset"].fillna(combination)
            else:
                _, ds = split_combination_name(combination)
                frame["dataset"] = ds
            frames.append(frame)
        except Exception as exc:
            print(f"Error reading {path}: {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def correlation_summary(input_dir: Path, combination_dir: str) -> Dict[str, float]:
    """Summarize off-diagonal values from each saved correlation matrix."""
    result = {}
    for metric, filename in (
        ("accuracy", "accuracy_correlation.csv"),
        ("runtime", "runtime_correlation.csv"),
        ("pareto", "pareto_correlation.csv"),
        ("accuracy_pearson", "accuracy_correlation_pearson.csv"),
        ("runtime_pearson", "runtime_correlation_pearson.csv"),
        ("pareto_pearson", "pareto_correlation_pearson.csv"),
    ):
        path = input_dir / combination_dir / filename
        try:
            matrix = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce").to_numpy()
            values = matrix[np.triu_indices_from(matrix, k=1)]
            stats = metric_statistics(values)
            result[f"{metric}_correlation_mean"] = stats["mean"]
            result[f"{metric}_correlation_std"] = stats["std"]
        except (FileNotFoundError, OSError, ValueError):
            result[f"{metric}_correlation_mean"] = float("nan")
            result[f"{metric}_correlation_std"] = float("nan")
    return result


def analyze_combination(group: pd.DataFrame, input_dir: Path) -> Dict[str, object]:
    """Compute per-combination cross-seed summaries."""
    first = group.iloc[0]
    record: Dict[str, object] = {
        "combination": first["combination"],
        "detector": first["detector"],
        "dataset": first["dataset"],
        "n_seeds": int(group["seed"].nunique()) if "seed" in group else len(group),
    }

    for metric in METRICS:
        stats = metric_statistics(group[metric] if metric in group else [])
        for name, value in stats.items():
            record[f"{metric}_{name}"] = value

    record.update(correlation_summary(input_dir, str(first["combination"])))
    return record


def print_metric_section(summary: pd.DataFrame, metric: str) -> None:
    """Print a compact table for one metric, grouped by detector/dataset."""
    label = METRIC_LABELS[metric]
    print("\n" + "=" * 120)
    print(f"{label.upper()} STABILITY ACROSS SEEDS")
    print("=" * 120)
    columns = [
        "detector", "dataset", "n_seeds", f"{metric}_mean", f"{metric}_std",
        f"{metric}_median", f"{metric}_p10", f"{metric}_p25", f"{metric}_p75",
        f"{metric}_p90", f"{metric}_min", f"{metric}_max", f"{metric}_cv",
    ]
    available = [column for column in columns if column in summary]
    display = summary[available].copy()
    numeric_columns = [column for column in available if column not in {"detector", "dataset"}]
    display[numeric_columns] = display[numeric_columns].round(4)
    print(display.to_string(index=False))


def print_grouped_summary(summary: pd.DataFrame, group_by: str) -> None:
    """Print aggregate metric summaries by detector or dataset."""
    print("\n" + "=" * 100)
    print(f"AGGREGATED BY {group_by.upper()}")
    print("=" * 100)
    rows = []
    for name, group in summary.groupby(group_by, dropna=False):
        row = {group_by: name, "n_combinations": len(group)}
        for metric in METRICS:
            values = finite_values(group[f"{metric}_mean"])
            stats = metric_statistics(values)
            row[f"{metric}_mean_of_means"] = stats["mean"]
            row[f"{metric}_std_across_combinations"] = stats["std"]
            row[f"{metric}_cv_of_means"] = stats["cv"]
        rows.append(row)
    if rows:
        print(pd.DataFrame(rows).round(4).to_string(index=False))


def create_plots(results: pd.DataFrame, output_dir: Path) -> None:
    """Create box plots showing the distribution of each metric over seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations = sorted(results["combination"].unique())
    for metric in METRICS:
        if metric not in results:
            continue
        data = [finite_values(results.loc[results["combination"] == combo, metric]) for combo in combinations]
        keep = [(combo, values) for combo, values in zip(combinations, data) if values]
        if not keep:
            continue
        labels, values = zip(*keep)
        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.2), 6))
        ax.boxplot(values, labels=labels, showfliers=True, flierprops={"markersize": 3, "alpha": 0.4})
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} across random seeds")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        path = output_dir / f"{metric}_by_combination.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multi-seed metric stability")
    parser.add_argument("--input-dir", default="multi_seed_results", help="Directory written by multi_seed_analysis.py")
    parser.add_argument("--output-csv", default=None, help="Summary CSV path (default: <input-dir>/multiseed_summary.csv)")
    parser.add_argument("--plot", default=None, help="Directory in which to save metric box plots")
    parser.add_argument("--detector", default=None, help="Analyze only this detector")
    parser.add_argument("--dataset", default=None, help="Analyze only this dataset")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        parser.error(f"input directory does not exist: {input_dir}")

    results = load_results(input_dir)
    if results.empty:
        parser.error("no multi-seed metric data was loaded")

    if args.detector:
        results = results[results["detector"] == args.detector]
    if args.dataset:
        results = results[results["dataset"] == args.dataset]
    if results.empty:
        parser.error("filters matched no multi-seed results")

    summary = pd.DataFrame(
        analyze_combination(group, input_dir)
        for _, group in results.groupby("combination", sort=True)
    )
    output_csv = Path(args.output_csv) if args.output_csv else input_dir / "multiseed_summary.csv"
    summary.to_csv(output_csv, index=False)
    print(f"Loaded {len(results)} per-seed rows across {len(summary)} combinations")
    print(f"Summary saved to {output_csv}")

    for metric in METRICS:
        print_metric_section(summary, metric)
    print_grouped_summary(summary, "dataset")
    print_grouped_summary(summary, "detector")

    correlation_columns = [column for column in summary if column.endswith("_correlation_mean")]
    if correlation_columns:
        print("\n" + "=" * 100)
        print("CROSS-SEED CORRELATION SUMMARY")
        print("=" * 100)
        print(summary[["detector", "dataset"] + correlation_columns].round(4).to_string(index=False))

    if args.plot:
        create_plots(results, Path(args.plot))


if __name__ == "__main__":
    main()
