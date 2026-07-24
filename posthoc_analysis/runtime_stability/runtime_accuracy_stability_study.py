#!/usr/bin/env python3
"""
Runtime + Accuracy Stability Study.

For a given (drift detector, dataset) pair, this script:
  1. Loads the historical OmniOpt result CSV from
     ``results/all_benchmark_results/<DETECTOR>/<DATASET>/`` (preferring the
     ``*_ACC_RT.csv`` file).
  2. Randomly samples N (default 10) completed configurations.
  3. Re-executes ``main.py`` M (default 20) times for each configuration.
  4. Records the runtime AND accuracy of every run, then computes the mean
     and standard deviation per configuration for both metrics.
  5. Writes the aggregated per-configuration statistics to a CSV file.

Each row in the output CSV represents one detector/dataset/configuration
combination together with its runtime and accuracy statistics. Rows from
multiple invocations of this script (one per detector/dataset) can later
be concatenated to build the full study table.

Usage:
    python runtime_accuracy_stability_study.py --detector DDAL --dataset Electricity
    python runtime_accuracy_stability_study.py --detector DDAL --dataset Electricity \
        --n-configs 10 --n-runs 20 --output-dir runtime_accuracy_stability_results

The script keeps no global state: it can be safely launched in parallel for
different detector/dataset pairs (each invocation writes to its own file).
"""

import argparse
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Datasets that should never be considered (per study spec).
EXCLUDED_DATASETS = {"SineClusters", "WaveformDrift2"}

# Default benchmark settings (mirrors run_stream_detector_optimization.sh).
DEFAULT_CLASSIFIER = "HoeffdingTreeClassifier"
DEFAULT_N_TRAINING_SAMPLES = 2000

# Columns that are NOT detector parameters in the OmniOpt result CSVs.
NON_PARAM_COLUMNS = {
    "Status",
    "ACCURACY",
    "RUNTIME",
    "REQLABELS",
    "MTR",
    "trial_index",
    "arm_name",
    "generation_method",
}

# Regex used to extract runtime from main.py output.
RUNTIME_RE = re.compile(r"^RUNTIME:\s*([0-9.+\-eE]+)\s*$", re.MULTILINE)
OO_RUNTIME_RE = re.compile(r"^OO-Info:\s*runtime:\s*([0-9.+\-eE]+)\s*$",
                           re.MULTILINE)

# Regex used to extract accuracy from main.py output.
ACCURACY_RE = re.compile(r"^ACCURACY:\s*([0-9.+\-eE]+)\s*$", re.MULTILINE)
OO_ACCURACY_RE = re.compile(r"^OO-Info:\s*accuracy:\s*([0-9.+\-eE]+)\s*$",
                            re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_results_csv(detector: str, dataset: str, base_dir: Path) -> Path:
    """Return the path to the OmniOpt result CSV for the given combination.

    Prefers ``<DETECTOR>_<DATASET>_ACC_RT.csv`` over the ``_REQL`` variant,
    but falls back to whichever CSV is present.
    """
    folder = base_dir / detector / dataset
    if not folder.is_dir():
        raise FileNotFoundError(f"Result folder not found: {folder}")

    preferred = folder / f"{detector}_{dataset}_ACC_RT.csv"
    if preferred.is_file():
        return preferred

    candidates = sorted(folder.glob(f"{detector}_{dataset}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No result CSV found in {folder}")
    # Prefer files without REQL if possible.
    non_reql = [c for c in candidates if "REQL" not in c.name]
    return non_reql[0] if non_reql else candidates[0]


def normalise_value(raw: str) -> str:
    """Normalise a CSV value so it becomes a valid Python literal/expression."""
    val = raw.strip()
    if val == "":
        return val
    upper = val.upper()
    if upper == "TRUE":
        return "True"
    if upper == "FALSE":
        return "False"
    return val


def load_configurations(csv_path: Path):
    """Return (param_columns, list_of_param_dicts) parsed from the CSV."""
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        param_columns = [c for c in reader.fieldnames
                         if c not in NON_PARAM_COLUMNS]
        configs = []
        for row in reader:
            if row.get("Status", "").strip() != "Completed":
                continue
            cfg = {col: normalise_value(row[col]) for col in param_columns}
            if any(v == "" for v in cfg.values()):
                continue
            configs.append(cfg)
    return param_columns, configs


def sample_unique_configs(configs, n_configs, rng):
    """Pick ``n_configs`` unique configurations (by parameter tuple)."""
    seen = {}
    for cfg in configs:
        key = tuple(sorted(cfg.items()))
        seen.setdefault(key, cfg)
    unique = list(seen.values())
    if len(unique) <= n_configs:
        return unique
    return rng.sample(unique, n_configs)


def build_detector_expr(detector: str, params: dict) -> str:
    """Build the Python expression passed as the detector argument to main.py."""
    parts = [f"{name}={value}" for name, value in params.items()]
    return f"{detector}({','.join(parts)})"


def parse_runtime_from_output(stdout: str):
    """Extract the runtime value from ``main.py`` stdout, if present."""
    match = RUNTIME_RE.search(stdout) or OO_RUNTIME_RE.search(stdout)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_accuracy_from_output(stdout: str):
    """Extract the accuracy value from ``main.py`` stdout, if present."""
    match = ACCURACY_RE.search(stdout) or OO_ACCURACY_RE.search(stdout)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def run_single(detector_expr: str, dataset: str, classifier: str,
               n_training_samples: int, project_dir: Path,
               python_exe: str, timeout: int):
    """Execute ``main.py`` once and return (runtime, accuracy, error)."""
    cmd = [
        python_exe, "main.py",
        "True", "True", "False",            # accuracy, runtime, reqlabels
        dataset, str(n_training_samples),
        classifier, detector_expr,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"

    if result.returncode != 0:
        err = (result.stderr.strip().splitlines()[-1]
               if result.stderr else f"exit={result.returncode}")
        return None, None, err

    runtime = parse_runtime_from_output(result.stdout)
    accuracy = parse_accuracy_from_output(result.stdout)
    if runtime is None and accuracy is None:
        return None, None, "METRICS_NOT_FOUND"
    if runtime is None:
        return None, accuracy, "RUNTIME_NOT_FOUND"
    if accuracy is None:
        return runtime, None, "ACCURACY_NOT_FOUND"
    return runtime, accuracy, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Runtime + accuracy stability study for a "
                    "detector/dataset pair.")
    parser.add_argument("--detector", required=True,
                        help="Drift detector class name (e.g. DDAL).")
    parser.add_argument("--dataset", required=True,
                        help="Dataset class name (e.g. Electricity).")
    parser.add_argument("--n-configs", type=int, default=10,
                        help="Number of random configurations to sample.")
    parser.add_argument("--n-runs", type=int, default=20,
                        help="Number of repeated runs per configuration.")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                        help="Classifier class name.")
    parser.add_argument("--n-training-samples", type=int,
                        default=DEFAULT_N_TRAINING_SAMPLES,
                        help="Number of training samples passed to main.py.")
    parser.add_argument("--results-dir", default="results/all_benchmark_results",
                        help="Directory containing prior OmniOpt CSV results.")
    parser.add_argument("--output-dir",
                        default="runtime_accuracy_stability_results",
                        help="Directory where the per-pair CSV is written.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for configuration sampling.")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-run timeout in seconds.")
    parser.add_argument("--python-exe", default=sys.executable,
                        help="Python executable used to invoke main.py.")
    args = parser.parse_args()

    if args.dataset in EXCLUDED_DATASETS:
        print(f"Dataset '{args.dataset}' is excluded from the study. "
              f"Nothing to do.", flush=True)
        return 0

    project_dir = Path(__file__).resolve().parent
    base_dir = (project_dir / args.results_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and sample configurations
    # ------------------------------------------------------------------
    csv_path = find_results_csv(args.detector, args.dataset, base_dir)
    print(f"[INFO] Loading configurations from: {csv_path}", flush=True)

    param_columns, configs = load_configurations(csv_path)
    print(f"[INFO] {len(configs)} completed configurations available "
          f"(params: {param_columns})", flush=True)

    if not configs:
        print(f"[ERROR] No completed configurations for "
              f"{args.detector}/{args.dataset}. Aborting.", flush=True)
        return 1

    rng = random.Random(args.seed)
    selected = sample_unique_configs(configs, args.n_configs, rng)
    print(f"[INFO] Selected {len(selected)} configurations for the study.",
          flush=True)

    # ------------------------------------------------------------------
    # Execute each configuration n_runs times
    # ------------------------------------------------------------------
    out_path = output_dir / f"{args.detector}_{args.dataset}.csv"
    fieldnames = [
        "detector", "dataset", "config_id", "params",
        "n_runs", "n_successful_runs",
        "mean_runtime", "std_runtime", "min_runtime", "max_runtime",
        "mean_accuracy", "std_accuracy", "min_accuracy", "max_accuracy",
        "runtimes", "accuracies", "errors",
    ]

    with out_path.open("w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for config_id, params in enumerate(selected):
            detector_expr = build_detector_expr(args.detector, params)
            print(f"\n[CONFIG {config_id + 1}/{len(selected)}] "
                  f"{detector_expr}", flush=True)

            runtimes = []
            accuracies = []
            errors = []
            for run_idx in range(args.n_runs):
                start = time.time()
                runtime, accuracy, err = run_single(
                    detector_expr=detector_expr,
                    dataset=args.dataset,
                    classifier=args.classifier,
                    n_training_samples=args.n_training_samples,
                    project_dir=project_dir,
                    python_exe=args.python_exe,
                    timeout=args.timeout,
                )
                wall = time.time() - start
                # Only count a run as successful if BOTH metrics were parsed.
                if runtime is not None and accuracy is not None:
                    runtimes.append(runtime)
                    accuracies.append(accuracy)
                    print(f"  run {run_idx + 1:>2}/{args.n_runs}: "
                          f"runtime={runtime:.2f} "
                          f"accuracy={accuracy:.4f} "
                          f"(wall={wall:.1f}s)",
                          flush=True)
                else:
                    errors.append(err or "UNKNOWN")
                    print(f"  run {run_idx + 1:>2}/{args.n_runs}: "
                          f"FAILED ({err})", flush=True)

            if runtimes:
                mean_rt = statistics.mean(runtimes)
                std_rt = statistics.stdev(runtimes) if len(runtimes) > 1 \
                    else 0.0
                min_rt = min(runtimes)
                max_rt = max(runtimes)
            else:
                mean_rt = std_rt = min_rt = max_rt = float("nan")

            if accuracies:
                mean_acc = statistics.mean(accuracies)
                std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 \
                    else 0.0
                min_acc = min(accuracies)
                max_acc = max(accuracies)
            else:
                mean_acc = std_acc = min_acc = max_acc = float("nan")

            writer.writerow({
                "detector": args.detector,
                "dataset": args.dataset,
                "config_id": config_id,
                "params": json.dumps(params),
                "n_runs": args.n_runs,
                "n_successful_runs": len(runtimes),
                "mean_runtime": mean_rt,
                "std_runtime": std_rt,
                "min_runtime": min_rt,
                "max_runtime": max_rt,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "min_accuracy": min_acc,
                "max_accuracy": max_acc,
                "runtimes": json.dumps(runtimes),
                "accuracies": json.dumps(accuracies),
                "errors": json.dumps(errors),
            })
            out_f.flush()

    print(f"\n[INFO] Results written to: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
