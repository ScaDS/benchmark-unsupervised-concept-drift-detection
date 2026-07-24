#!/usr/bin/env python3
"""
Generate Pareto-front plots (accuracy vs. runtime) per dataset.

For each dataset that appears under ``results/all_benchmark_results/<DETECTOR>/<DATASET>/``,
this script:
  1. Loads every detector's ``<DETECTOR>_<DATASET>_ACC_RT.csv`` file.
  2. Plots every completed configuration as a scatter point colored by
     drift detector.
  3. Highlights the global Pareto front (maximize ACCURACY, minimize RUNTIME)
     across all detectors with a connecting line and bold markers.
  4. Writes one PNG per dataset to ``pareto_fronts/<DATASET>.png``.

Usage:
    python plot_pareto_fronts.py
    python plot_pareto_fronts.py --results-dir results/all_benchmark_results \
                                 --output-dir pareto_fronts
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXCLUDED_DATASETS = {"SineClusters", "WaveformDrift2"}


def load_detector_dataset_csv(csv_path: Path):
    """Return (accuracies, runtimes) lists for completed rows."""
    accs, rts = [], []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("Status", "").strip()
            if status not in ("Completed", "COMPLETED"):
                continue
            acc_raw = row.get("ACCURACY", "").strip()
            rt_raw = row.get("RUNTIME", "").strip()
            if not acc_raw or not rt_raw:
                continue
            try:
                acc = float(acc_raw)
                rt = float(rt_raw)
            except ValueError:
                continue
            accs.append(acc)
            rts.append(rt)
    return accs, rts


def discover_datasets(base_dir: Path):
    """Return sorted set of dataset names across all detectors."""
    datasets = set()
    for det_dir in base_dir.iterdir():
        if not det_dir.is_dir():
            continue
        for ds_dir in det_dir.iterdir():
            if ds_dir.is_dir() and ds_dir.name not in EXCLUDED_DATASETS:
                datasets.add(ds_dir.name)
    return sorted(datasets)


def discover_detectors(base_dir: Path):
    """Return sorted list of detector names."""
    return sorted(d.name for d in base_dir.iterdir() if d.is_dir())


def pareto_front(points):
    """Return indices of Pareto-optimal points.

    Objective: maximize accuracy (x[0]), minimize runtime (x[1]).
    """
    n = len(points)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        ai, ri = points[i]
        for j in range(n):
            if i == j or not keep[j]:
                continue
            aj, rj = points[j]
            # j dominates i if j is >= in acc and <= in rt, with at least one strict.
            if aj >= ai and rj <= ri and (aj > ai or rj < ri):
                keep[i] = False
                break
    return [i for i, k in enumerate(keep) if k]


DETECTOR_COLORS = {
    "BNDM":      "#1f77b4",
    "CDBD":      "#aec7e8",
    "CDLEEDS":   "#f7b6d2",
    "CSDDM":     "#1f4e9c",
    "D3":        "#2ca02c",
    "DAWIDD":    "#c5b0d5",
    "DDAL":      "#ff9896",
    "EDFS":      "#7f7f7f",
    "HDDDM":     "#f4b8b0",
    "IBDD":      "#d62728",
    "IKS":       "#b8896a",
    "NNDVI":     "#bcbd22",
    "OCDD":      "#9467bd",
    "PCACD":     "#dbdb8d",
    "SlidShaps": "#efe6dc",
    "SPLL":      "#8c564b",
    "STUDD":     "#ff7f0e",
    "UCDD":      "#17becf",
    "UDetect":   "#e377c2",
    "WindowKDE": "#c7c7c7",
}


def _build_color_map(detectors):
    """Return a stable color dict for the given detector list."""
    fallback_cmap = plt.get_cmap("tab20")
    color_map = {}
    fb_idx = 0
    for d in detectors:
        if d in DETECTOR_COLORS:
            color_map[d] = DETECTOR_COLORS[d]
        else:
            color_map[d] = fallback_cmap(fb_idx % 20)
            fb_idx += 1
    return color_map


def _render_scatter(per_det_data, color_map, title, out_path,
                    xlim=None, ylim=None):
    """Draw the full scatter (cloud + per-detector Pareto) for one dataset.

    ``per_det_data`` is a list of tuples
    ``(detector, accs, rts, front_idx)`` already computed by the caller.
    ``xlim``/``ylim`` optionally zoom the axes.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    for det, accs, rts, front_idx in per_det_data:
        front_set = set(front_idx)
        front_acc = [accs[i] for i in front_idx]
        front_rt = [rts[i] for i in front_idx]
        non_front_acc = [accs[i] for i in range(len(accs))
                         if i not in front_set]
        non_front_rt = [rts[i] for i in range(len(rts))
                        if i not in front_set]

        if non_front_acc:
            ax.scatter(non_front_acc, non_front_rt,
                       s=14, alpha=0.12,
                       color=color_map[det],
                       edgecolors="none",
                       zorder=2)
        ax.scatter(front_acc, front_rt,
                   s=45, alpha=0.85,
                   color=color_map[det],
                   edgecolor="black", linewidths=0.5,
                   label=det, zorder=3)

    ax.set_xlabel("Accuracy", fontsize=24)
    ax.set_ylabel("Runtime (s)", fontsize=24)
    ax.set_title(title, fontsize=28)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    legend = ax.legend(fontsize=18, ncol=2, loc="upper left", framealpha=0.85)
    # Handle matplotlib API compatibility
    handles = legend.legendHandles if hasattr(legend, 'legendHandles') else legend.legend_handles
    for handle in handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dataset(dataset: str, detectors, base_dir: Path,
                 output_dir: Path):
    """Build and save the Pareto-front PNGs (full + elbow zoom) for a dataset."""
    color_map = _build_color_map(detectors)

    per_det_data = []          # (detector, accs, rts, front_idx)
    global_front_pts = []      # (acc, rt) of every per-detector Pareto point

    for det in detectors:
        csv_path = base_dir / det / dataset / f"{det}_{dataset}_ACC_RT.csv"
        if not csv_path.is_file():
            # Fall back to the *_REQL.csv variant (e.g. fixed DDAL data).
            csv_path = base_dir / det / dataset / f"{det}_{dataset}_ACC_RT_REQL.csv"
        if not csv_path.is_file():
            continue
        accs, rts = load_detector_dataset_csv(csv_path)
        if not accs:
            continue
        pts = list(zip(accs, rts))
        front_idx = pareto_front(pts)
        per_det_data.append((det, accs, rts, front_idx))
        for i in front_idx:
            global_front_pts.append((accs[i], rts[i]))

    if not per_det_data:
        print(f"[WARN] No data for dataset {dataset}; skipping.")
        return False

    total_front_points = sum(len(f) for *_, f in per_det_data)
    n_detectors = len(per_det_data)

    # ------------------------------------------------------------------
    # 1) Full overview plot
    # ------------------------------------------------------------------
    full_path = output_dir / f"{dataset}.png"
    _render_scatter(
        per_det_data, color_map,
        title=f"Per-detector Pareto-optimal configurations — {dataset}",
        out_path=full_path,
    )
    print(f"[OK] {dataset}: {total_front_points} Pareto points from "
          f"{n_detectors} detectors -> {full_path}")

    # ------------------------------------------------------------------
    # 2) Elbow / zoom plot
    # ------------------------------------------------------------------
    # The elbow is the bounding box of the *global* Pareto front (i.e. the
    # set of points that are not dominated by any other detector's point).
    # This naturally highlights the trade-off region between accuracy and
    # runtime and trims away the cloud of clearly-dominated configurations.
    global_idx = pareto_front(global_front_pts)
    elbow_pts = [global_front_pts[i] for i in global_idx]
    elbow_acc = [p[0] for p in elbow_pts]
    elbow_rt = [p[1] for p in elbow_pts]

    acc_lo, acc_hi = min(elbow_acc), max(elbow_acc)
    rt_lo, rt_hi = min(elbow_rt), max(elbow_rt)
    # Add a small margin around the elbow bounding box.
    acc_pad = max((acc_hi - acc_lo) * 0.10, 0.01)
    rt_pad = max((rt_hi - rt_lo) * 0.10, 1.0)
    xlim = (acc_lo - acc_pad, min(1.0, acc_hi + acc_pad))
    ylim = (max(0.0, rt_lo - rt_pad), rt_hi + rt_pad)

    elbow_path = output_dir / f"{dataset}_elbow.png"
    _render_scatter(
        per_det_data, color_map,
        title=f"Pareto elbow (zoom on global trade-off region) — {dataset}",
        out_path=elbow_path,
        xlim=xlim, ylim=ylim,
    )
    print(f"      elbow zoom ({len(elbow_pts)} global Pareto points) "
          f"-> {elbow_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-dataset Pareto fronts (accuracy vs. runtime).")
    parser.add_argument("--results-dir", default="results/all_benchmark_results",
                        help="Directory containing the OmniOpt result CSVs.")
    parser.add_argument("--output-dir", default="pareto_fronts",
                        help="Directory where the PNGs are written.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional subset of datasets to plot.")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    base_dir = (project_dir / args.results_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.is_dir():
        raise SystemExit(f"Results directory not found: {base_dir}")

    detectors = discover_detectors(base_dir)
    datasets = discover_datasets(base_dir)
    if args.datasets:
        requested = set(args.datasets)
        datasets = [d for d in datasets if d in requested]
        missing = requested - set(datasets)
        if missing:
            print(f"[WARN] Requested datasets not found: {sorted(missing)}")
    print(f"[INFO] Detectors ({len(detectors)}): {detectors}")
    print(f"[INFO] Datasets  ({len(datasets)}): {datasets}")
    print(f"[INFO] Writing PNGs to: {output_dir}")

    n_ok = 0
    for ds in datasets:
        if plot_dataset(ds, detectors, base_dir, output_dir):
            n_ok += 1

    print(f"\n[DONE] {n_ok}/{len(datasets)} datasets plotted.")


if __name__ == "__main__":
    main()
