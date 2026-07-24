#!/usr/bin/env python3
"""
Summarise the (relative) runtime / accuracy stability across all
detector/dataset/configuration combinations.

Input
-----
A directory containing the per-pair CSVs produced by
``runtime_stability_study.py`` or ``runtime_accuracy_stability_study.py``.
Each row corresponds to one (detector, dataset, config_id) combination and
must contain at least the columns:

    detector, dataset, config_id,
    n_runs, n_successful_runs,
    mean_runtime, std_runtime

If ``mean_accuracy`` and ``std_accuracy`` are present, accuracy stability is
also reported.

Output
------
For every available metric (runtime, accuracy) the script reports:

  * mean and std of the absolute per-config standard deviation
    (the quantity the user asked for)
  * mean and std of the coefficient of variation  CV = std / mean
    (a properly *relative* stability measure; see notes at the bottom)

Results are printed to stdout and -- if ``--output`` is given -- also
written as a CSV (one row per (scope, metric) combination, where
``scope`` is either ``overall``, ``per_detector`` or ``per_dataset``).

Usage
-----
    python summarize_stability.py --input runtime_stability_results
    python summarize_stability.py --input runtime_accuracy_stability_results \
        --output stability_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


# Known metrics. For each metric we keep:
#   (name, mean_col, std_col, values_col)
# ``values_col`` is the JSON-list column with the raw per-run values, used for
# per-run outlier filtering. If that column is absent we fall back to the
# precomputed mean/std.
METRIC_PAIRS = [
    ("runtime", "mean_runtime", "std_runtime", "runtimes"),
    ("accuracy", "mean_accuracy", "std_accuracy", "accuracies"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rows(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    for path in csv_files:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if math.isnan(x):
        return None
    return x


def parse_values(raw: str | None) -> list[float]:
    """Parse a JSON-encoded list of floats stored in a CSV cell."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    out: list[float] = []
    for v in data:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(x):
            out.append(x)
    return out


def mad_filter(values: list[float], k: float) -> tuple[list[float], int]:
    """Drop entries outside median +/- k * MAD (scaled to sigma).

    Returns (kept_values, n_dropped). With ``k <= 0`` or fewer than 3
    values nothing is filtered. MAD is scaled by 1.4826 so that ``k`` is
    expressed in Gaussian-sigma units (so k=3 ~ the classic 3-sigma rule).
    Ties at zero MAD (e.g. integer-second runtimes that are all identical)
    are left untouched.
    """
    if k is None or k <= 0 or len(values) < 3:
        return list(values), 0
    med = statistics.median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = statistics.median(abs_dev)
    if mad == 0:
        return list(values), 0
    sigma = 1.4826 * mad
    cutoff = k * sigma
    kept = [v for v in values if abs(v - med) <= cutoff]
    return kept, len(values) - len(kept)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(values: Iterable[float]) -> dict:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": n,
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if n > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def per_row_stats(rows: list[dict], mean_col: str, std_col: str,
                  values_col: str | None, mad_k: float
                  ) -> tuple[list[tuple[dict, float, float]], dict]:
    """Return ``(stats, info)`` where ``stats`` is a list of (row, std, cv)
    tuples and ``info`` reports how many runs were dropped by the MAD
    outlier filter.

    Per-run filtering is only applied when ``values_col`` is present in the
    row and ``mad_k > 0``; otherwise the precomputed ``mean_col`` /
    ``std_col`` values are used as-is.

    A row is skipped if fewer than 2 (filtered) runs remain.
    """
    out: list[tuple[dict, float, float]] = []
    info = {"n_runs_total": 0, "n_runs_dropped": 0,
            "n_rows_filtered": 0, "used_raw_values": False}
    for row in rows:
        values = parse_values(row.get(values_col)) if values_col else []
        if values and mad_k and mad_k > 0:
            info["used_raw_values"] = True
            kept, dropped = mad_filter(values, mad_k)
            info["n_runs_total"] += len(values)
            info["n_runs_dropped"] += dropped
            if dropped:
                info["n_rows_filtered"] += 1
            if len(kept) < 2:
                continue
            mean_v = statistics.mean(kept)
            std_v = statistics.stdev(kept)
        else:
            n_ok = to_float(row.get("n_successful_runs"))
            if n_ok is None or n_ok < 2:
                continue
            mean_v = to_float(row.get(mean_col))
            std_v = to_float(row.get(std_col))
            if mean_v is None or std_v is None:
                continue
        cv = std_v / mean_v if mean_v != 0 else float("nan")
        out.append((row, std_v, cv))
    return out, info


def aggregate(rows: list[dict], metric: str, mean_col: str, std_col: str,
              values_col: str | None, mad_k: float,
              group_key: str | None = None,
              cv_mad_k: float = 0.0) -> list[dict]:
    """Aggregate per-row std and CV. If ``group_key`` is given, one summary
    is produced per unique value of that column; otherwise a single overall
    summary is returned.

    ``cv_mad_k`` (>0) enables a second-stage MAD outlier filter applied to
    the per-config CV (and std) values *within each group* before computing
    the group summary. This removes entire configurations whose stability
    is itself an outlier among its peers.
    """
    stats, _info = per_row_stats(rows, mean_col, std_col, values_col, mad_k)
    if not stats:
        return []

    if group_key is None:
        groups: dict[str, list] = {"<overall>": stats}
    else:
        groups = {}
        for entry in stats:
            key = entry[0].get(group_key, "<unknown>")
            groups.setdefault(key, []).append(entry)

    results = []
    for key, entries in sorted(groups.items()):
        n_before = len(entries)
        n_dropped = 0
        if cv_mad_k and cv_mad_k > 0 and n_before >= 3:
            cvs_all = [e[2] for e in entries
                       if not math.isnan(e[2])]
            _kept, n_dropped = mad_filter(cvs_all, cv_mad_k)
            if n_dropped:
                med = statistics.median(cvs_all)
                mad = statistics.median([abs(v - med) for v in cvs_all])
                cutoff = cv_mad_k * 1.4826 * mad
                entries = [e for e in entries
                           if math.isnan(e[2])
                           or abs(e[2] - med) <= cutoff]
        stds = [e[1] for e in entries]
        cvs = [e[2] for e in entries]
        std_summary = summarise(stds)
        cv_summary = summarise(cvs)
        results.append({
            "metric": metric,
            "group": key,
            "n_configs": len(entries),
            "n_configs_dropped": n_dropped,
            "mean_std": std_summary["mean"],
            "std_of_std": std_summary["std"],
            "min_std": std_summary["min"],
            "max_std": std_summary["max"],
            "mean_cv": cv_summary["mean"],
            "std_of_cv": cv_summary["std"],
            "min_cv": cv_summary["min"],
            "max_cv": cv_summary["max"],
        })
    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_section(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n=== {title} ===")
    header = (
        f"{'group':<25} {'n(-drop)':>8} "
        f"{'mean_std':>10} {'std_of_std':>10} "
        f"{'std_range':>20} "
        f"{'mean_cv':>10} {'std_of_cv':>10} "
        f"{'cv_range':>20}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        std_range = f"[{r['min_std']:.3f},{r['max_std']:.3f}]"
        cv_range = f"[{r['min_cv']:.3f},{r['max_cv']:.3f}]"
        n_label = f"{r['n_configs']}"
        if r.get("n_configs_dropped"):
            n_label = f"{r['n_configs']}(-{r['n_configs_dropped']})"
        print(
            f"{str(r['group']):<25} {n_label:>8} "
            f"{r['mean_std']:>10.4f} {r['std_of_std']:>10.4f} "
            f"{std_range:>20} "
            f"{r['mean_cv']:>10.4f} {r['std_of_cv']:>10.4f} "
            f"{cv_range:>20}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise runtime/accuracy stability across all pairs.")
    parser.add_argument("--input", required=True,
                        help="Directory containing the per-pair stability CSVs.")
    parser.add_argument("--output", default=None,
                        help="Optional CSV file to write the summary to.")
    parser.add_argument("--no-breakdowns", action="store_true",
                        help="Skip per-detector / per-dataset breakdowns.")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Optional whitelist of datasets to include (substring match, "
             "case-insensitive). Default: ForestCovertype, GasSensor, Poker, "
             "Rialto, Sensor. Pass 'all' to disable filtering.")
    parser.add_argument(
        "--mad-k", type=float, default=3.0,
        help="Per-run outlier threshold expressed in Gaussian-sigma units "
             "(median +/- k * 1.4826 * MAD). Applied to the raw 'runtimes' / "
             "'accuracies' lists in the input CSVs; mean/std/CV are then "
             "recomputed on the filtered values. Use 0 to disable.")
    parser.add_argument(
        "--cv-mad-k", type=float, default=3.0,
        help="Second-stage MAD outlier threshold applied to the per-config "
             "CV values within each group (overall / per-detector / "
             "per-dataset) before computing the group summary. Same units "
             "as --mad-k. Use 0 to disable.")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    rows = load_rows(input_dir)
    print(f"[INFO] Loaded {len(rows)} configuration rows from "
          f"{input_dir}.", flush=True)

    # ------------------------------------------------------------------
    # Optional dataset filter
    # ------------------------------------------------------------------
    if args.datasets is None:
        dataset_filter = ["forest", "gas", "poker", "rialto", "sensor"]
    elif len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        dataset_filter = None
    else:
        dataset_filter = [d.lower() for d in args.datasets]

    if dataset_filter is not None:
        before = len(rows)
        rows = [r for r in rows
                if any(tok in r.get("dataset", "").lower()
                       for tok in dataset_filter)]
        kept = sorted({r.get("dataset", "") for r in rows})
        print(f"[INFO] Dataset filter {dataset_filter} -> "
              f"{len(rows)}/{before} rows; datasets kept: {kept}",
              flush=True)
        if not rows:
            print("[ERROR] No rows left after dataset filtering.", flush=True)
            return 1

    # Discover which metrics are present in the data.
    available_cols = set().union(*(r.keys() for r in rows)) if rows else set()
    active_metrics = []
    for name, mean_c, std_c, values_c in METRIC_PAIRS:
        if mean_c in available_cols and std_c in available_cols:
            vc = values_c if values_c in available_cols else None
            active_metrics.append((name, mean_c, std_c, vc))
    if not active_metrics:
        print("[ERROR] No (mean_*, std_*) metric pairs found in input.",
              flush=True)
        return 1
    print(f"[INFO] Metrics found: {[m[0] for m in active_metrics]}",
          flush=True)
    if args.mad_k and args.mad_k > 0:
        print(f"[INFO] Per-run MAD outlier filter active: k={args.mad_k} "
              f"(median +/- k*1.4826*MAD).", flush=True)
    else:
        print("[INFO] Per-run outlier filtering disabled.", flush=True)
    if args.cv_mad_k and args.cv_mad_k > 0:
        print(f"[INFO] Per-config CV MAD outlier filter active: "
              f"k={args.cv_mad_k}.", flush=True)
    else:
        print("[INFO] Per-config CV outlier filtering disabled.", flush=True)

    all_summary_rows: list[dict] = []
    for metric, mean_c, std_c, values_c in active_metrics:
        if values_c is None and args.mad_k and args.mad_k > 0:
            print(f"[WARN] '{metric}': raw per-run values column not present; "
                  f"using precomputed mean/std without outlier filtering.",
                  flush=True)
        # Report outlier-removal stats once per metric (on the full set).
        _stats, info = per_row_stats(rows, mean_c, std_c, values_c,
                                     args.mad_k)
        if info["used_raw_values"]:
            total = info["n_runs_total"]
            dropped = info["n_runs_dropped"]
            pct = (100.0 * dropped / total) if total else 0.0
            print(f"[INFO] {metric}: dropped {dropped}/{total} runs "
                  f"({pct:.2f}%) across {info['n_rows_filtered']} configs.",
                  flush=True)

        overall = aggregate(rows, metric, mean_c, std_c, values_c,
                            args.mad_k, group_key=None,
                            cv_mad_k=args.cv_mad_k)
        print_section(f"{metric.upper()} -- overall", overall)
        for r in overall:
            r2 = dict(r); r2["scope"] = "overall"; all_summary_rows.append(r2)

        if not args.no_breakdowns:
            per_det = aggregate(rows, metric, mean_c, std_c, values_c,
                                args.mad_k, group_key="detector",
                                cv_mad_k=args.cv_mad_k)
            print_section(f"{metric.upper()} -- per detector", per_det)
            for r in per_det:
                r2 = dict(r); r2["scope"] = "per_detector"
                all_summary_rows.append(r2)

            per_ds = aggregate(rows, metric, mean_c, std_c, values_c,
                               args.mad_k, group_key="dataset",
                               cv_mad_k=args.cv_mad_k)
            print_section(f"{metric.upper()} -- per dataset", per_ds)
            for r in per_ds:
                r2 = dict(r); r2["scope"] = "per_dataset"
                all_summary_rows.append(r2)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["scope", "metric", "group",
                      "n_configs", "n_configs_dropped",
                      "mean_std", "std_of_std", "min_std", "max_std",
                      "mean_cv", "std_of_cv", "min_cv", "max_cv"]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_summary_rows:
                writer.writerow({k: r.get(k) for k in fieldnames})
        print(f"\n[INFO] Summary written to: {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
