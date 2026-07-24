#!/usr/bin/env python3
"""
Analyze reproducibility results from individual CSV files.

This script reads all detector_dataset_reproducibility.csv files from the
reproducibility_results directory and computes:
- Overall accuracy success rate (percentage)
- Runtime variance statistics (mean, std, min, max)
- Breakdown by detector and dataset

Usage:
    python analyze_reproducibility.py --input-dir reproducibility_results
"""

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv_files(input_dir: Path) -> List[Dict[str, Any]]:
    """Load all reproducibility CSV files from the directory."""
    rows = []
    csv_files = sorted(input_dir.glob("*_reproducibility.csv"))
    
    if not csv_files:
        print(f"Warning: No *_reproducibility.csv files found in {input_dir}")
        return rows
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            with csv_file.open('r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return rows


def load_available_configurations_with_all_params(benchmark_results_dir: Path) -> Tuple[Dict[str, Set[tuple]], List[str]]:
    """Load all available configurations and return all parameter columns.
    
    Returns a tuple of (available_configs, all_param_columns).
    """
    available_configs = {}
    all_param_columns = set()
    
    if not benchmark_results_dir.exists():
        print(f"Warning: Benchmark results directory {benchmark_results_dir} does not exist")
        return available_configs, []
    
    # Find all CSV files in the benchmark results directory
    csv_files = list(benchmark_results_dir.glob('*/*/*.csv'))
    
    for csv_file in csv_files:
        try:
            # Extract detector and dataset from path
            parts = csv_file.parts
            detector = parts[-3]
            dataset = parts[-2]
            key = (detector, dataset)
            
            if key not in available_configs:
                available_configs[key] = set()
            
            # Read CSV and extract configurations
            df = pd.read_csv(csv_file)
            
            # Filter out non-completed rows
            if 'Status' in df.columns:
                df = df[df['Status'] == 'Completed']
            
            # Get parameter columns (exclude metric columns)
            metric_columns = {'Status', 'ACCURACY', 'RUNTIME', 'REQLABELS', 'MTR', 'OO-Info'}
            param_columns = [col for col in df.columns if col not in metric_columns]
            
            # Collect all parameter columns across all files
            all_param_columns.update(param_columns)
                
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return available_configs, sorted(all_param_columns)


def load_available_configurations(benchmark_results_dir: Path, common_param_columns: List[str] = None) -> Dict[str, List[Dict[str, str]]]:
    """Load all available configurations from all_benchmark_results directory.
    
    Returns a dictionary mapping (detector, dataset) to a list of configuration dictionaries.
    Each configuration dictionary maps parameter names to string values.
    
    If common_param_columns is provided, only those columns are stored.
    """
    available_configs = {}
    all_param_columns = set()
    empty_keys_debug = []  # Track pairs with no completed rows for debugging
    
    if not benchmark_results_dir.exists():
        print(f"Warning: Benchmark results directory {benchmark_results_dir} does not exist")
        return available_configs
    
    # Find all CSV files in the benchmark results directory
    csv_files = list(benchmark_results_dir.glob('*/*/*.csv'))
    
    print(f"Loading configurations from {len(csv_files)} benchmark result files...")
    
    for csv_file in csv_files:
        try:
            # Extract detector and dataset from path
            parts = csv_file.parts
            detector = parts[-3]
            dataset = parts[-2]
            key = (detector, dataset)
            
            if key not in available_configs:
                available_configs[key] = []
            
            # Read CSV and extract configurations
            df = pd.read_csv(csv_file)
            
            # Filter out non-completed rows (case-insensitive)
            if 'Status' in df.columns:
                before_filter = len(df)
                df = df[df['Status'].str.upper() == 'COMPLETED']
                after_filter = len(df)
                if before_filter > 0 and after_filter == 0:
                    # Log files that have rows but no completed ones
                    if len(empty_keys_debug) < 10:  # Only log first 10
                        print(f"Debug: {detector}/{dataset} has {before_filter} rows but 0 completed rows")
                        empty_keys_debug.append(key)
            
            # Get parameter columns (exclude metric columns)
            metric_columns = {'Status', 'ACCURACY', 'RUNTIME', 'REQLABELS', 'MTR', 'OO-Info'}
            param_columns = [col for col in df.columns if col not in metric_columns]
            
            # Collect all parameter columns across all files
            all_param_columns.update(param_columns)
            
            # Always use all available parameter columns for each file
            # Don't pre-filter to common columns - do that during matching instead
            use_columns = sorted(param_columns)
            
            # Extract each configuration as a dictionary
            for _, row in df.iterrows():
                # Convert all values to strings for consistent comparison (normalized)
                config_dict = {col: normalize_value(row[col]) for col in use_columns}
                available_configs[key].append(config_dict)
                
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Print summary
    total_configs = sum(len(configs) for configs in available_configs.values())
    print(f"Loaded {total_configs} configurations across {len(available_configs)} detector/dataset pairs")
    print(f"Parameter columns found in benchmark results: {sorted(all_param_columns)}")
    
    # Debug: show sample keys
    print(f"Sample detector/dataset keys in available_configs:")
    for key in sorted(list(available_configs.keys()))[:10]:
        print(f"  {key[0]}/{key[1]}")
    if len(available_configs) > 10:
        print(f"  ... and {len(available_configs) - 10} more")
    
    # Remove empty configuration lists
    empty_keys = [key for key, configs in available_configs.items() if len(configs) == 0]
    if empty_keys:
        print(f"Removing {len(empty_keys)} detector/dataset pairs with no valid configurations:")
        for key in sorted(empty_keys)[:20]:  # Show first 20
            print(f"  {key[0]}/{key[1]}")
        if len(empty_keys) > 20:
            print(f"  ... and {len(empty_keys) - 20} more")
    
    for key in empty_keys:
        del available_configs[key]
    
    if empty_keys:
        print(f"Final: {sum(len(configs) for configs in available_configs.values())} configurations across {len(available_configs)} detector/dataset pairs")
    
    return available_configs


def normalize_value(v) -> str:
    """Normalize a parameter value for consistent comparison.

    Handles the mismatch between csv.DictReader (which returns '' for empty
    fields) and pandas (which returns np.nan, converted to 'nan' by str()).
    """
    s = str(v).strip()
    if s == '' or s.lower() == 'nan' or s.lower() == 'none':
        return ''
    return s


def configuration_exists(row: Dict[str, Any], available_configs: Dict[str, List[Dict[str, str]]], 
                          param_columns: List[str] = None) -> bool:
    """Check if a configuration from reproducibility results exists in available_configs.
    
    Args:
        row: A row from the reproducibility results CSV
        available_configs: Dictionary of available configurations by (detector, dataset)
        param_columns: List of parameter column names to check (unused, kept for compatibility)
        
    Returns:
        True if the configuration exists in the benchmark results, False otherwise
    """
    detector = row.get('detector', 'Unknown')
    dataset = row.get('dataset', 'Unknown')
    key = (detector, dataset)
    
    if key not in available_configs:
        return False
    
    # Build configuration dictionary from the row - use all parameter columns in the row
    try:
        # Exclude result columns and metadata columns to get actual parameter columns
        result_columns = {'detector', 'dataset', 'mode', 'accuracy_success', 'run_success', 
                        'reqlabels_success', 'runtime_ratio', 'error', 'index',
                        'original_accuracy', 'original_runtime', 'original_reqlabels', 'original_mtr',
                        'reproduced_accuracy', 'reproduced_runtime', 'reproduced_reqlabels', 'reproduced_mtr',
                        '_original_accuracy', '_original_runtime', '_original_reqlabels', '_original_mtr'}
        
        # Get all parameter columns from the row (normalized for consistent comparison)
        repro_config = {col: normalize_value(row.get(col)) for col in row.keys() if col not in result_columns}
        
        # Check if any benchmark configuration matches on common parameters
        for bench_config in available_configs[key]:
            # Only match on parameters that exist in BOTH configs
            common_params = set(repro_config.keys()) & set(bench_config.keys())
            
            # Check if all common parameters match (with normalization)
            all_match = True
            for param in common_params:
                if repro_config[param] != normalize_value(bench_config[param]):
                    all_match = False
                    break
            if all_match:
                return True
        return False
    except Exception:
        return False


def parse_float(value: str) -> Optional[float]:
    """Safely parse a float value."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_percentile(data: List[float], p: int) -> float:
    """Return the p-th percentile (1-99) of data, robust for small samples."""
    if not data:
        return float('nan')
    if len(data) == 1:
        return data[0]
    qs = statistics.quantiles(data, n=100, method='inclusive')
    return qs[p - 1]


def mad_filter(values: List[float], k: float) -> tuple[List[float], int]:
    """Drop entries outside median +/- k * MAD (scaled to sigma).

    Returns (kept_values, n_dropped). With ``k <= 0`` or fewer than 3
    values nothing is filtered. MAD is scaled by 1.4826 so that ``k`` is
    expressed in Gaussian-sigma units (so k=3 ~ the classic 3-sigma rule).
    """
    if k <= 0 or len(values) < 3:
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


def pareto_front_2d(points: List[Tuple[float, float]]) -> List[int]:
    """Return indices of Pareto-optimal points (maximize acc, minimize runtime)."""
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
            if aj >= ai and rj <= ri and (aj > ai or rj < ri):
                keep[i] = False
                break
    return [i for i, k in enumerate(keep) if k]


def compute_hypervolume_2d(points: List[Tuple[float, float]], ref_acc: float, ref_rt: float) -> float:
    """Compute 2D hypervolume for points (accuracy, runtime) with reference point.

    Maximizes accuracy, minimizes runtime. Reference point is (ref_acc, ref_rt)
    — typically the worst observed values. Only Pareto-optimal points contribute.
    """
    if not points:
        return 0.0
    front_idx = pareto_front_2d(points)
    front = [points[i] for i in front_idx]
    front = [(a, r) for a, r in front if a >= ref_acc and r <= ref_rt]
    if not front:
        return 0.0
    front.sort(key=lambda p: p[0])
    hv = 0.0
    prev_acc = ref_acc
    for acc, rt in front:
        hv += (acc - prev_acc) * (ref_rt - rt)
        prev_acc = acc
    return hv


def compute_runtime_stats(runtime_ratios: List[float], mad_k: float = 0.0) -> Dict[str, float]:
    """Compute statistics for runtime ratios with optional MAD outlier filtering."""
    valid_ratios = [r for r in runtime_ratios if r is not None and not (r != r)]  # Filter NaN
    
    if not valid_ratios:
        return {
            'count': 0,
            'mean': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'median': float('nan'),
            'n_dropped': 0
        }
    
    # Apply MAD outlier filtering if requested
    filtered_ratios, n_dropped = mad_filter(valid_ratios, mad_k)
    
    return {
        'count': len(filtered_ratios),
        'mean': statistics.mean(filtered_ratios),
        'std': statistics.stdev(filtered_ratios) if len(filtered_ratios) > 1 else 0.0,
        'min': min(filtered_ratios),
        'max': max(filtered_ratios),
        'median': statistics.median(filtered_ratios),
        'n_dropped': n_dropped
    }


def analyze_rows(rows: List[Dict[str, Any]], mad_k: float = 0.0,
                 repro_threshold: float = 90.0, use_mad_outlier: bool = False,
                 outlier_mad_k: float = 3.0, available_configs: Optional[Dict[str, List[Dict[str, str]]]] = None,
                 ignore_timeouts: bool = False) -> None:
    """Analyze and print statistics from the loaded rows."""
    if not rows:
        print("No data to analyze")
        return
    
    # Filter out timeout runs if requested
    if ignore_timeouts:
        before_timeout_filter = len(rows)
        rows = [r for r in rows if 'timeout' not in str(r.get('error', '')).lower()]
        n_timeouts = before_timeout_filter - len(rows)
        if n_timeouts > 0:
            print(f"Ignoring {n_timeouts} timeout run(s) (--ignore-timeouts)")
    
    # Filter rows based on available configurations if provided
    if available_configs is not None:
        before_filter = len(rows)
        filtered_rows = []
        not_found_rows = []
        
        # Debug: show sample configurations
        if rows:
            sample_row = rows[0]
            detector = sample_row.get('detector', 'Unknown')
            dataset = sample_row.get('dataset', 'Unknown')
            key = (detector, dataset)
            
            print(f"\n{'='*70}")
            print("DEBUG: Sample Configuration Comparison")
            print(f"{'='*70}")
            print(f"Sample row from reproducibility results: {detector}/{dataset}")
            print(f"Key being looked up: {repr(key)}")
            print(f"Detector repr: {repr(detector)}, Dataset repr: {repr(dataset)}")
            
            # Get parameter columns from sample row
            result_columns = {'detector', 'dataset', 'mode', 'accuracy_success', 'run_success', 
                            'reqlabels_success', 'runtime_ratio', 'error', 'index',
                            'original_accuracy', 'original_runtime', 'original_reqlabels', 'original_mtr',
                            'reproduced_accuracy', 'reproduced_runtime', 'reproduced_reqlabels', 'reproduced_mtr',
                            '_original_accuracy', '_original_runtime', '_original_reqlabels', '_original_mtr'}
            sample_config = {col: normalize_value(sample_row.get(col)) for col in sample_row.keys() if col not in result_columns}
            print(f"Reproducibility config: {sample_config}")
            
            print(f"Key in available_configs: {key in available_configs}")
            if key in available_configs:
                print(f"Number of configs for this key: {len(available_configs[key])}")
            
            if key in available_configs and available_configs[key]:
                sample_benchmark_config = available_configs[key][0]
                print(f"Sample config from benchmark results: {sample_benchmark_config}")
                
                # Check if the specific reproducibility config exists
                if configuration_exists(sample_row, available_configs):
                    print(f"✓ Reproducibility config MATCHES benchmark results")
                else:
                    print(f"✗ Reproducibility config does NOT match benchmark results")
                    print(f"  Total benchmark configs for this key: {len(available_configs[key])}")
            else:
                print(f"No benchmark configurations found for {detector}/{dataset}")
                print(f"Checking if '{detector}' exists in benchmark results keys...")
                detectors_in_benchmark = set(det for det, _ in available_configs.keys())
                print(f"Detector '{detector}' in detectors_in_benchmark: {detector in detectors_in_benchmark}")
                if detector in detectors_in_benchmark:
                    print(f"  Yes, detector '{detector}' found. Datasets for this detector:")
                    datasets_for_detector = sorted([ds for det, ds in available_configs.keys() if det == detector])
                    for ds in datasets_for_detector[:10]:
                        print(f"    {detector}/{ds}")
                    # Check if the exact key exists
                    exact_keys = [(det, ds) for det, ds in available_configs.keys() if det == detector and ds == dataset]
                    print(f"  Exact matches for {detector}/{dataset}: {exact_keys}")
                else:
                    print(f"  No, detector '{detector}' NOT found in benchmark results")
                    print(f"  Available detectors in benchmark results (first 20): {sorted(list(detectors_in_benchmark))[:20]}")
            print(f"{'='*70}\n")
        
        for row in rows:
            if configuration_exists(row, available_configs):
                filtered_rows.append(row)
            else:
                not_found_rows.append(row)
        
        rows = filtered_rows
        after_filter = len(rows)
        
        if before_filter != after_filter:
            print(f"\n{'='*70}")
            print("CONFIGURATION FILTERING STATISTICS")
            print(f"{'='*70}")
            print(f"Total configurations in reproducibility results: {before_filter}")
            print(f"Configurations found in benchmark results: {after_filter} ({after_filter/before_filter*100:.1f}%)")
            print(f"Configurations NOT found in benchmark results: {before_filter - after_filter} ({(before_filter - after_filter)/before_filter*100:.1f}%)")
            
            # Breakdown by detector
            print(f"\n{'-'*70}")
            print("CONFIGURATIONS NOT FOUND - BY DETECTOR")
            print(f"{'-'*70}")
            detector_stats = {}
            for row in not_found_rows:
                detector = row.get('detector', 'Unknown')
                if detector not in detector_stats:
                    detector_stats[detector] = 0
                detector_stats[detector] += 1
            
            for detector in sorted(detector_stats.keys()):
                print(f"{detector:<30} {detector_stats[detector]:>6}")
            
            # Breakdown by dataset
            print(f"\n{'-'*70}")
            print("CONFIGURATIONS NOT FOUND - BY DATASET")
            print(f"{'-'*70}")
            dataset_stats = {}
            for row in not_found_rows:
                dataset = row.get('dataset', 'Unknown')
                if dataset not in dataset_stats:
                    dataset_stats[dataset] = 0
                dataset_stats[dataset] += 1
            
            for dataset in sorted(dataset_stats.keys()):
                print(f"{dataset:<30} {dataset_stats[dataset]:>6}")
            
            print(f"{'='*70}\n")
    
    # Filter out SineClusters and WaveformDrift2 datasets
    before = len(rows)
    rows = [r for r in rows
            if r.get('dataset', '') not in ['SineClusters', 'WaveformDrift2']]
    after = len(rows)
    if before != after:
        print(f"Filtered out {before - after} rows from excluded datasets")
    
    total = len(rows)
    
    # Parse boolean fields
    accuracy_success = []
    run_success = []
    reqlabels_success = []
    runtime_ratios = []
    
    for row in rows:
        # Parse accuracy_success
        acc_succ = row.get('accuracy_success', '').lower()
        accuracy_success.append(acc_succ == 'true' or acc_succ == '1' or acc_succ == True)
        
        # Parse run_success
        run_succ = row.get('run_success', '').lower()
        run_success.append(run_succ == 'true' or run_succ == '1' or run_succ == True)
        
        # Parse reqlabels_success (if present)
        req_succ = row.get('reqlabels_success', '').lower()
        if req_succ:
            reqlabels_success.append(req_succ == 'true' or req_succ == '1' or req_succ == True)
        
        # Parse runtime_ratio
        ratio = parse_float(row.get('runtime_ratio'))
        runtime_ratios.append(ratio)
    
    # Overall statistics
    acc_successful = sum(accuracy_success)
    acc_success_rate = (acc_successful / total * 100) if total > 0 else 0
    
    run_successful = sum(run_success)
    run_success_rate = (run_successful / total * 100) if total > 0 else 0
    
    print("\n" + "="*70)
    print("OVERALL REPRODUCIBILITY STATISTICS")
    print("="*70)
    print(f"Total configurations: {total}")
    print(f"Run success: {run_successful} ({run_success_rate:.1f}%)")
    print(f"Accuracy reproduced: {acc_successful} ({acc_success_rate:.1f}%)")
    print(f"Accuracy failed: {total - acc_successful} ({100 - acc_success_rate:.1f}%)")
    
    if reqlabels_success:
        req_successful = sum(reqlabels_success)
        req_total = len(reqlabels_success)
        req_success_rate = (req_successful / req_total * 100) if req_total > 0 else 0
        print(f"ReqLabels reproduced: {req_successful} ({req_success_rate:.1f}%)")
        print(f"ReqLabels failed: {req_total - req_successful} ({100 - req_success_rate:.1f}%)")
    
    # Runtime statistics
    runtime_stats = compute_runtime_stats(runtime_ratios, mad_k)
    print("\n" + "-"*70)
    print("RUNTIME RATIO STATISTICS (reproduced / original)")
    print("-"*70)
    if mad_k > 0:
        print(f"MAD outlier filter: k={mad_k} (median +/- k*1.4826*MAD)")
        print(f"Outliers removed: {runtime_stats['n_dropped']}")
    print(f"Valid runtime ratios: {runtime_stats['count']}")
    print(f"Mean: {runtime_stats['mean']:.3f}")
    print(f"Std: {runtime_stats['std']:.3f}")
    print(f"Median: {runtime_stats['median']:.3f}")
    print(f"P25: {safe_percentile([r for r in runtime_ratios if r is not None and r == r], 25):.3f}")
    print(f"P75: {safe_percentile([r for r in runtime_ratios if r is not None and r == r], 75):.3f}")
    print(f"P10: {safe_percentile([r for r in runtime_ratios if r is not None and r == r], 10):.3f}")
    print(f"P90: {safe_percentile([r for r in runtime_ratios if r is not None and r == r], 90):.3f}")
    print(f"Min: {runtime_stats['min']:.3f}")
    print(f"Max: {runtime_stats['max']:.3f}")

    # Accuracy ratio statistics (reproduced / original)
    acc_ratios = []
    for r in rows:
        orig = parse_float(r.get('original_accuracy'))
        rep = parse_float(r.get('reproduced_accuracy'))
        if orig is not None and rep is not None and orig != 0:
            acc_ratios.append(rep / orig)

    print("\n" + "-"*70)
    print("ACCURACY RATIO STATISTICS (reproduced / original)")
    print("-"*70)
    if acc_ratios:
        print(f"Valid accuracy ratios: {len(acc_ratios)}")
        print(f"Mean: {statistics.mean(acc_ratios):.3f}")
        print(f"Std:  {statistics.stdev(acc_ratios) if len(acc_ratios) > 1 else 0.0:.3f}")
        print(f"Median: {statistics.median(acc_ratios):.3f}")
        print(f"P25: {safe_percentile(acc_ratios, 25):.3f}")
        print(f"P75: {safe_percentile(acc_ratios, 75):.3f}")
        print(f"P10: {safe_percentile(acc_ratios, 10):.3f}")
        print(f"P90: {safe_percentile(acc_ratios, 90):.3f}")
        print(f"Min:  {min(acc_ratios):.3f}")
        print(f"Max:  {max(acc_ratios):.3f}")
    else:
        print("No valid accuracy ratios found")

    # Group by detector
    print("\n" + "="*70)
    print("BY DETECTOR")
    print("="*70)
    
    detectors = {}
    for row in rows:
        detector = row.get('detector', 'Unknown')
        if detector not in detectors:
            detectors[detector] = []
        detectors[detector].append(row)
    
    print(f"{'Detector':<30} {'Total':>8} {'Acc Success':>12} {'Rate':>8} {'Acc Ratio Mean':>14} {'Acc Ratio Std':>14} {'RT Ratio Mean':>13} {'RT Ratio Std':>13}")
    print("-" * 120)
    
    for detector in sorted(detectors.keys()):
        det_rows = detectors[detector]
        det_total = len(det_rows)
        
        det_acc_success = sum(
            1 for r in det_rows 
            if r.get('accuracy_success', '').lower() in ['true', '1']
        )
        det_rate = (det_acc_success / det_total * 100) if det_total > 0 else 0
        
        det_ratios = [parse_float(r.get('runtime_ratio')) for r in det_rows]
        det_runtime_stats = compute_runtime_stats(det_ratios, mad_k)

        det_acc_ratios = []
        for r in det_rows:
            orig = parse_float(r.get('original_accuracy'))
            rep = parse_float(r.get('reproduced_accuracy'))
            if orig is not None and rep is not None and orig != 0:
                det_acc_ratios.append(rep / orig)
        det_acc_ratio_mean = statistics.mean(det_acc_ratios) if det_acc_ratios else float('nan')
        det_acc_ratio_std = statistics.stdev(det_acc_ratios) if len(det_acc_ratios) > 1 else 0.0
        
        print(f"{detector:<30} {det_total:>8} {det_acc_success:>12} {det_rate:>7.1f}% {det_acc_ratio_mean:>14.3f} {det_acc_ratio_std:>14.3f} {det_runtime_stats['mean']:>13.3f} {det_runtime_stats['std']:>13.3f}")
    
    # Group by dataset
    print("\n" + "="*70)
    print("BY DATASET")
    print("="*70)
    
    datasets = {}
    for row in rows:
        dataset = row.get('dataset', 'Unknown')
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(row)
    
    print(f"{'Dataset':<30} {'Total':>8} {'Acc Success':>12} {'Rate':>8} {'Acc Ratio Mean':>14} {'Acc Ratio Std':>14} {'RT Ratio Mean':>13} {'RT Ratio Std':>13}")
    print("-" * 120)
    
    for dataset in sorted(datasets.keys()):
        ds_rows = datasets[dataset]
        ds_total = len(ds_rows)
        
        ds_acc_success = sum(
            1 for r in ds_rows 
            if r.get('accuracy_success', '').lower() in ['true', '1']
        )
        ds_rate = (ds_acc_success / ds_total * 100) if ds_total > 0 else 0
        
        ds_ratios = [parse_float(r.get('runtime_ratio')) for r in ds_rows]
        ds_runtime_stats = compute_runtime_stats(ds_ratios, mad_k)

        ds_acc_ratios = []
        for r in ds_rows:
            orig = parse_float(r.get('original_accuracy'))
            rep = parse_float(r.get('reproduced_accuracy'))
            if orig is not None and rep is not None and orig != 0:
                ds_acc_ratios.append(rep / orig)
        ds_acc_ratio_mean = statistics.mean(ds_acc_ratios) if ds_acc_ratios else float('nan')
        ds_acc_ratio_std = statistics.stdev(ds_acc_ratios) if len(ds_acc_ratios) > 1 else 0.0
        
        print(f"{dataset:<30} {ds_total:>8} {ds_acc_success:>12} {ds_rate:>7.1f}% {ds_acc_ratio_mean:>14.3f} {ds_acc_ratio_std:>14.3f} {ds_runtime_stats['mean']:>13.3f} {ds_runtime_stats['std']:>13.3f}")
    
    # Failed configurations
    failed = [r for r in rows if r.get('accuracy_success', '').lower() not in ['true', '1']]
    if failed:
        print("\n" + "="*70)
        print(f"FAILED CONFIGURATIONS ({len(failed)})")
        print("="*70)
        print(f"{'Detector':<30} {'Dataset':<30} {'Error':<40}")
        print("-" * 100)

        for row in failed[:20]:  # Show first 20
            detector = row.get('detector', 'Unknown')[:30]
            dataset = row.get('dataset', 'Unknown')[:30]
            error = row.get('error', '')[:40]
            print(f"{detector:<30} {dataset:<30} {error:<40}")

        if len(failed) > 20:
            print(f"... and {len(failed) - 20} more failed configurations")

    # Low reproducibility combinations (Detector/Dataset)
    print("\n" + "="*70)
    print("DETECTOR/DATASET COMBINATIONS WITH LOW REPRODUCIBILITY")
    print("="*70)

    # Group by detector/dataset combination
    combinations = {}
    for row in rows:
        detector = row.get('detector', 'Unknown')
        dataset = row.get('dataset', 'Unknown')
        key = (detector, dataset)
        if key not in combinations:
            combinations[key] = []
        combinations[key].append(row)

    # Calculate success rate for each combination
    combo_stats = []
    for (detector, dataset), combo_rows in combinations.items():
        combo_total = len(combo_rows)
        combo_success = sum(
            1 for r in combo_rows
            if r.get('accuracy_success', '').lower() in ['true', '1']
        )
        combo_rate = (combo_success / combo_total * 100) if combo_total > 0 else 0
        combo_stats.append({
            'detector': detector,
            'dataset': dataset,
            'total': combo_total,
            'successful': combo_success,
            'failed': combo_total - combo_success,
            'success_rate': combo_rate
        })

    # Sort by success rate (ascending) to show worst first
    combo_stats.sort(key=lambda x: x['success_rate'])

    # Show combinations with success rate < 90%
    low_repro = [c for c in combo_stats if c['success_rate'] < repro_threshold]

    if low_repro:
        print(f"Showing combinations with success rate < {repro_threshold}% ({len(low_repro)} combinations)")
        print(f"{'Detector':<30} {'Dataset':<30} {'Total':>6} {'Success':>8} {'Failed':>6} {'Rate':>8}")
        print("-" * 100)
        for stat in low_repro:
            print(f"{stat['detector']:<30} {stat['dataset']:<30} {stat['total']:>6} "
                  f"{stat['successful']:>8} {stat['failed']:>6} {stat['success_rate']:>7.1f}%")
    else:
        print(f"All detector/dataset combinations have success rate >= {repro_threshold}%")

    # Pareto hypervolume ratio per detector/dataset combination
    print("\n" + "="*70)
    print("PARETO HYPERVOLUME RATIO (reproduced / original) BY DETECTOR/DATASET")
    print("="*70)

    hv_ratios = []
    combo_records = []  # for aggregation
    print(f"{'Detector':<22} {'Dataset':<22} {'AccR Mean':>9} {'AccR Std':>9} {'RTR Mean':>9} {'RTR Std':>9} {'Orig HV':>10} {'Repro HV':>10} {'HV Ratio':>9}")
    print("-" * 120)

    for (detector, dataset), combo_rows in sorted(combinations.items()):
        orig_points = []
        repro_points = []
        combo_rt_ratios = []
        combo_acc_ratios = []
        for r in combo_rows:
            orig_acc = parse_float(r.get('original_accuracy'))
            orig_rt = parse_float(r.get('original_runtime'))
            repro_acc = parse_float(r.get('reproduced_accuracy'))
            repro_rt = parse_float(r.get('reproduced_runtime'))
            if orig_acc is not None and orig_rt is not None and orig_rt > 0:
                orig_points.append((orig_acc, orig_rt))
            if repro_acc is not None and repro_rt is not None and repro_rt > 0:
                repro_points.append((repro_acc, repro_rt))
            rt_ratio = parse_float(r.get('runtime_ratio'))
            if rt_ratio is not None and rt_ratio == rt_ratio:
                combo_rt_ratios.append(rt_ratio)
            if orig_acc is not None and repro_acc is not None and orig_acc != 0:
                combo_acc_ratios.append(repro_acc / orig_acc)

        if not orig_points or not repro_points:
            continue

        all_accs = [a for a, _ in orig_points + repro_points]
        all_rts = [r for _, r in orig_points + repro_points]
        ref_acc = min(all_accs)
        ref_rt = max(all_rts)

        orig_hv = compute_hypervolume_2d(orig_points, ref_acc, ref_rt)
        repro_hv = compute_hypervolume_2d(repro_points, ref_acc, ref_rt)

        acc_r_mean = statistics.mean(combo_acc_ratios) if combo_acc_ratios else float('nan')
        acc_r_std = statistics.stdev(combo_acc_ratios) if len(combo_acc_ratios) > 1 else 0.0
        rt_r_mean = statistics.mean(combo_rt_ratios) if combo_rt_ratios else float('nan')
        rt_r_std = statistics.stdev(combo_rt_ratios) if len(combo_rt_ratios) > 1 else 0.0

        if orig_hv > 0:
            ratio = repro_hv / orig_hv
            hv_ratios.append(ratio)
            combo_records.append({
                'detector': detector,
                'dataset': dataset,
                'acc_r_mean': acc_r_mean,
                'acc_r_std': acc_r_std,
                'rt_r_mean': rt_r_mean,
                'rt_r_std': rt_r_std,
                'hv_ratio': ratio,
            })
            print(f"{detector:<22} {dataset:<22} {acc_r_mean:>9.3f} {acc_r_std:>9.3f} {rt_r_mean:>9.3f} {rt_r_std:>9.3f} {orig_hv:>10.2f} {repro_hv:>10.2f} {ratio:>9.3f}")

    if hv_ratios:
        print("-" * 120)
        print(f"Hypervolume ratio summary: n={len(hv_ratios)}, "
              f"mean={statistics.mean(hv_ratios):.3f}, "
              f"std={statistics.stdev(hv_ratios) if len(hv_ratios) > 1 else 0.0:.3f}, "
              f"median={statistics.median(hv_ratios):.3f}, "
              f"P25={safe_percentile(hv_ratios, 25):.3f}, P75={safe_percentile(hv_ratios, 75):.3f}, "
              f"P10={safe_percentile(hv_ratios, 10):.3f}, P90={safe_percentile(hv_ratios, 90):.3f}")

        # Aggregated by dataset
        print("\n" + "="*70)
        print("AGGREGATED BY DATASET")
        print("="*70)
        ds_groups = {}
        for rec in combo_records:
            ds_groups.setdefault(rec['dataset'], []).append(rec)
        print(f"{'Dataset':<25} {'n':>4} {'AccR Med':>8} {'P25':>8} {'P75':>8} {'P10':>8} {'P90':>8} {'RTR Med':>8} {'P25':>8} {'P75':>8} {'P10':>8} {'P90':>8} {'HV Med':>8} {'P25':>8} {'P75':>8} {'P10':>8} {'P90':>8}")
        print("-" * 155)
        for ds in sorted(ds_groups.keys()):
            recs = ds_groups[ds]
            acc_means = [r['acc_r_mean'] for r in recs]
            rt_means = [r['rt_r_mean'] for r in recs]
            hv_vals = [r['hv_ratio'] for r in recs]
            print(f"{ds:<25} {len(recs):>4} "
                  f"{statistics.median(acc_means):>8.3f} {safe_percentile(acc_means, 25):>8.3f} {safe_percentile(acc_means, 75):>8.3f} {safe_percentile(acc_means, 10):>8.3f} {safe_percentile(acc_means, 90):>8.3f} "
                  f"{statistics.median(rt_means):>8.3f} {safe_percentile(rt_means, 25):>8.3f} {safe_percentile(rt_means, 75):>8.3f} {safe_percentile(rt_means, 10):>8.3f} {safe_percentile(rt_means, 90):>8.3f} "
                  f"{statistics.median(hv_vals):>8.3f} {safe_percentile(hv_vals, 25):>8.3f} {safe_percentile(hv_vals, 75):>8.3f} {safe_percentile(hv_vals, 10):>8.3f} {safe_percentile(hv_vals, 90):>8.3f}")

        # Aggregated by detector
        print("\n" + "="*70)
        print("AGGREGATED BY DETECTOR")
        print("="*70)
        det_groups = {}
        for rec in combo_records:
            det_groups.setdefault(rec['detector'], []).append(rec)
        print(f"{'Detector':<25} {'n':>4} {'AccR Med':>8} {'P25':>8} {'P75':>8} {'P10':>8} {'P90':>8} {'RTR Med':>8} {'P25':>8} {'P75':>8} {'P10':>8} {'P90':>8} {'HV Med':>8} {'P25':>8} {'P75':>8} {'P10':>8} {'P90':>8}")
        print("-" * 155)
        for det in sorted(det_groups.keys()):
            recs = det_groups[det]
            acc_means = [r['acc_r_mean'] for r in recs]
            rt_means = [r['rt_r_mean'] for r in recs]
            hv_vals = [r['hv_ratio'] for r in recs]
            print(f"{det:<25} {len(recs):>4} "
                  f"{statistics.median(acc_means):>8.3f} {safe_percentile(acc_means, 25):>8.3f} {safe_percentile(acc_means, 75):>8.3f} {safe_percentile(acc_means, 10):>8.3f} {safe_percentile(acc_means, 90):>8.3f} "
                  f"{statistics.median(rt_means):>8.3f} {safe_percentile(rt_means, 25):>8.3f} {safe_percentile(rt_means, 75):>8.3f} {safe_percentile(rt_means, 10):>8.3f} {safe_percentile(rt_means, 90):>8.3f} "
                  f"{statistics.median(hv_vals):>8.3f} {safe_percentile(hv_vals, 25):>8.3f} {safe_percentile(hv_vals, 75):>8.3f} {safe_percentile(hv_vals, 10):>8.3f} {safe_percentile(hv_vals, 90):>8.3f}")
    else:
        print("No valid hypervolume ratios could be computed.")

    # MAD outlier detection for low reproducibility
    if use_mad_outlier and len(combo_stats) >= 3:
        print("\n" + "="*70)
        print("OUTLIER DETECTION FOR LOW REPRODUCIBILITY (MAD-based)")
        print("="*70)

        # We're looking for LOW success rates, so we want to detect outliers on the low end
        # Convert to failure rates for outlier detection (higher = worse)
        failure_rates = [100.0 - c['success_rate'] for c in combo_stats]

        # Apply MAD filter to identify outliers
        med = statistics.median(failure_rates)
        abs_dev = [abs(v - med) for v in failure_rates]
        mad = statistics.median(abs_dev)

        if mad > 0:
            sigma = 1.4826 * mad
            cutoff = outlier_mad_k * sigma  # Use configurable k for outlier detection

            # Identify outliers (failure rate > median + cutoff)
            outliers = []
            for i, (stat, fail_rate) in enumerate(zip(combo_stats, failure_rates)):
                if fail_rate > med + cutoff:
                    outliers.append(stat)

            if outliers:
                print(f"Found {len(outliers)} outlier combinations (failure rate > median + {outlier_mad_k}*MAD)")
                print(f"Median failure rate: {med:.2f}%")
                print(f"MAD cutoff: {cutoff:.2f}%")
                print(f"{'Detector':<30} {'Dataset':<30} {'Total':>6} {'Success':>8} {'Failed':>6} {'Rate':>8}")
                print("-" * 100)
                for stat in outliers:
                    print(f"{stat['detector']:<30} {stat['dataset']:<30} {stat['total']:>6} "
                          f"{stat['successful']:>8} {stat['failed']:>6} {stat['success_rate']:>7.1f}%")

                # Calculate overall statistics after removing outliers
                outlier_keys = {(o['detector'], o['dataset']) for o in outliers}
                non_outlier_rows = [r for r in rows
                                    if (r.get('detector', 'Unknown'), r.get('dataset', 'Unknown')) not in outlier_keys]

                if non_outlier_rows:
                    non_outlier_total = len(non_outlier_rows)
                    non_outlier_success = sum(
                        1 for r in non_outlier_rows
                        if r.get('accuracy_success', '').lower() in ['true', '1']
                    )
                    non_outlier_rate = (non_outlier_success / non_outlier_total * 100) if non_outlier_total > 0 else 0

                    print("\n" + "-"*70)
                    print("OVERALL STATISTICS AFTER REMOVING OUTLIERS")
                    print("-"*70)
                    print(f"Total configurations (without outliers): {non_outlier_total}")
                    print(f"Accuracy reproduced: {non_outlier_success} ({non_outlier_rate:.1f}%)")
                    print(f"Accuracy failed: {non_outlier_total - non_outlier_success} ({100 - non_outlier_rate:.1f}%)")
                    print(f"Configurations removed as outliers: {len(rows) - non_outlier_total}")
            else:
                print("No outlier combinations detected using MAD (k=3)")
        else:
            print("MAD is zero, cannot detect outliers (all combinations have same failure rate)")


def plot_box_whisker(rows: List[Dict[str, Any]], mad_k: float, output_dir: Path) -> None:
    """Generate box-whisker plots for runtime and accuracy ratios, grouped by dataset."""
    # Prepare data
    datasets = {}
    for row in rows:
        dataset = row.get('dataset', 'Unknown')
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(row)

    dataset_names = sorted(datasets.keys())

    # Collect runtime ratios and accuracy ratios per dataset
    rt_data = []
    acc_data = []
    for ds in dataset_names:
        ds_rows = datasets[ds]
        rt_ratios = [parse_float(r.get('runtime_ratio')) for r in ds_rows]
        rt_ratios = [r for r in rt_ratios if r is not None and r == r]  # filter None and NaN
        rt_ratios, _ = mad_filter(rt_ratios, mad_k)
        rt_data.append(rt_ratios)

        acc_ratios = []
        for r in ds_rows:
            orig = parse_float(r.get('original_accuracy'))
            rep = parse_float(r.get('reproduced_accuracy'))
            if orig is not None and rep is not None and orig != 0:
                acc_ratios.append(rep / orig)
        acc_ratios, _ = mad_filter(acc_ratios, mad_k)
        acc_data.append(acc_ratios)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Runtime ratio box plot
    fig, ax = plt.subplots(figsize=(max(12, len(dataset_names) * 1.2), 6))
    bp = ax.boxplot(rt_data, labels=dataset_names, showfliers=True, flierprops={'markersize': 2, 'alpha': 0.3})
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Perfect reproduction (1.0)')
    ax.set_ylabel('Runtime Ratio (reproduced / original)')
    ax.set_title('Runtime Reproduction Ratio by Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    fig.tight_layout()
    rt_path = output_dir / 'runtime_ratio_boxplot.png'
    fig.savefig(rt_path, dpi=150)
    plt.close(fig)
    print(f"Saved runtime ratio box plot to {rt_path}")

    # Accuracy ratio box plot
    fig, ax = plt.subplots(figsize=(max(12, len(dataset_names) * 1.2), 6))
    bp = ax.boxplot(acc_data, labels=dataset_names, showfliers=True, flierprops={'markersize': 2, 'alpha': 0.3})
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Perfect reproduction (1.0)')
    ax.set_ylabel('Accuracy Ratio (reproduced / original)')
    ax.set_title('Accuracy Reproduction Ratio by Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    fig.tight_layout()
    acc_path = output_dir / 'accuracy_ratio_boxplot.png'
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"Saved accuracy ratio box plot to {acc_path}")

    # Hypervolume ratio box plot, grouped by dataset
    # For each dataset, compute per-detector hypervolume ratios
    detectors_per_ds = {}
    for row in rows:
        ds = row.get('dataset', 'Unknown')
        det = row.get('detector', 'Unknown')
        if ds not in detectors_per_ds:
            detectors_per_ds[ds] = {}
        if det not in detectors_per_ds[ds]:
            detectors_per_ds[ds][det] = []
        detectors_per_ds[ds][det].append(row)

    hv_data = []
    hv_dataset_names = []
    for ds in dataset_names:
        if ds not in detectors_per_ds:
            continue
        ds_hv_ratios = []
        for det, det_rows in detectors_per_ds[ds].items():
            orig_points = []
            repro_points = []
            for r in det_rows:
                orig_acc = parse_float(r.get('original_accuracy'))
                orig_rt = parse_float(r.get('original_runtime'))
                repro_acc = parse_float(r.get('reproduced_accuracy'))
                repro_rt = parse_float(r.get('reproduced_runtime'))
                if orig_acc is not None and orig_rt is not None and orig_rt > 0:
                    orig_points.append((orig_acc, orig_rt))
                if repro_acc is not None and repro_rt is not None and repro_rt > 0:
                    repro_points.append((repro_acc, repro_rt))
            if not orig_points or not repro_points:
                continue
            all_accs = [a for a, _ in orig_points + repro_points]
            all_rts = [r for _, r in orig_points + repro_points]
            ref_acc = min(all_accs)
            ref_rt = max(all_rts)
            orig_hv = compute_hypervolume_2d(orig_points, ref_acc, ref_rt)
            repro_hv = compute_hypervolume_2d(repro_points, ref_acc, ref_rt)
            if orig_hv > 0:
                ds_hv_ratios.append(repro_hv / orig_hv)
        if ds_hv_ratios:
            hv_data.append(ds_hv_ratios)
            hv_dataset_names.append(ds)

    if hv_data:
        fig, ax = plt.subplots(figsize=(max(12, len(hv_dataset_names) * 1.2), 6))
        ax.boxplot(hv_data, labels=hv_dataset_names, showfliers=True, flierprops={'markersize': 3, 'alpha': 0.5})
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Perfect reproduction (1.0)')
        ax.set_ylabel('Hypervolume Ratio (reproduced / original)')
        ax.set_title('Pareto Hypervolume Reproduction Ratio by Dataset')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        fig.tight_layout()
        hv_path = output_dir / 'hypervolume_ratio_boxplot.png'
        fig.savefig(hv_path, dpi=150)
        plt.close(fig)
        print(f"Saved hypervolume ratio box plot to {hv_path}")

    # Hypervolume ratio distribution plot (ECDF + histogram)
    all_hv = [r for ds_ratios in hv_data for r in ds_ratios] if hv_data else []
    if all_hv:
        import numpy as np
        all_hv_sorted = np.sort(all_hv)
        cdf = np.arange(1, len(all_hv_sorted) + 1) / len(all_hv_sorted)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ECDF
        ax1.step(all_hv_sorted, cdf, where='post', linewidth=1.5, color='steelblue')
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Perfect (1.0)')
        for p, label in [(10, 'P10'), (25, 'P25'), (50, 'Median'), (75, 'P75'), (90, 'P90')]:
            val = safe_percentile(all_hv, p)
            ax1.axhline(y=p / 100, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
            ax1.plot(val, p / 100, 'o', color='darkorange', markersize=4)
            ax1.annotate(f'{label}={val:.3f}', (val, p / 100),
                         textcoords="offset points", xytext=(8, -3), fontsize=7, color='darkorange')
        ax1.set_xlabel('Hypervolume Ratio (reproduced / original)')
        ax1.set_ylabel('Cumulative proportion')
        ax1.set_title('ECDF of Hypervolume Ratios')
        ax1.legend(loc='lower right')
        ax1.set_ylim(0, 1.05)

        # Histogram
        ax2.hist(all_hv, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect (1.0)')
        ax2.axvline(x=statistics.median(all_hv), color='darkorange', linestyle='-', linewidth=1.5, label=f'Median={statistics.median(all_hv):.3f}')
        ax2.set_xlabel('Hypervolume Ratio (reproduced / original)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Hypervolume Ratios')
        ax2.legend()

        fig.tight_layout()
        dist_path = output_dir / 'hypervolume_ratio_distribution.png'
        fig.savefig(dist_path, dpi=150)
        plt.close(fig)
        print(f"Saved hypervolume ratio distribution plot to {dist_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze reproducibility results from individual CSV files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='reproducibility_results',
        help='Directory containing the *_reproducibility.csv files'
    )
    parser.add_argument(
        '--benchmark-results-dir',
        type=str,
        default='results/all_benchmark_results',
        help='Directory containing the all_benchmark_results data. '
             'If provided, only configurations that exist in this directory will be analyzed. '
             'Default: results/all_benchmark_results'
    )
    parser.add_argument(
        '--mad-k',
        type=float,
        default=3.0,
        help='MAD outlier threshold for runtime ratios (median +/- k*1.4826*MAD). '
             'Use 0 to disable. Default: 3.0'
    )
    parser.add_argument(
        '--repro-threshold',
        type=float,
        default=90.0,
        help='Success rate threshold for low reproducibility warning. '
             'Combinations below this rate are shown. Default: 90.0'
    )
    parser.add_argument(
        '--detect-outliers',
        action='store_true',
        help='Enable MAD-based outlier detection for low reproducibility combinations'
    )
    parser.add_argument(
        '--outlier-mad-k',
        type=float,
        default=3.0,
        help='MAD outlier threshold for detecting low reproducibility outlier combinations '
             '(median + k*1.4826*MAD). Only used with --detect-outliers. Default: 3.0'
    )
    parser.add_argument(
        '--ignore-timeouts',
        action='store_true',
        help='Exclude runs that timed out (error field contains "Timeout")'
    )
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='If provided, generate box-whisker plots for runtime and accuracy ratios '
             'in the given output directory. e.g. --plot plots/'
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        sys.exit(1)

    rows = load_csv_files(input_dir)

    if not rows:
        print("No data loaded. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(rows)} configuration rows")

    # Load available configurations from benchmark results if directory exists
    available_configs = None
    benchmark_results_dir = Path(args.benchmark_results_dir)
    
    if benchmark_results_dir.exists():
        # Load all configurations with their full parameter sets
        available_configs = load_available_configurations(benchmark_results_dir)
        
        # Also print columns from reproducibility results for comparison
        if rows:
            sample_row = rows[0]
            reproducibility_columns = set(sample_row.keys())
            # Exclude result columns and metadata columns to get actual parameter columns
            result_columns = {'detector', 'dataset', 'mode', 'accuracy_success', 'run_success', 
                            'reqlabels_success', 'runtime_ratio', 'error', 'index',
                            'original_accuracy', 'original_runtime', 'original_reqlabels', 'original_mtr',
                            'reproduced_accuracy', 'reproduced_runtime', 'reproduced_reqlabels', 'reproduced_mtr',
                            '_original_accuracy', '_original_runtime', '_original_reqlabels', '_original_mtr'}
            reproducibility_params = reproducibility_columns - result_columns
            print(f"Parameter columns found in reproducibility results: {sorted(reproducibility_params)}")
            
            # Get all parameter columns from benchmark results
            all_benchmark_params = set()
            for configs in available_configs.values():
                if configs:
                    all_benchmark_params.update(configs[0].keys())
            
            if all_benchmark_params:
                print(f"Parameter columns in benchmark results but not in reproducibility: {sorted(all_benchmark_params - reproducibility_params)}")
                print(f"Parameter columns in reproducibility but not in benchmark results: {sorted(reproducibility_params - all_benchmark_params)}")
    else:
        print(f"Warning: Benchmark results directory {benchmark_results_dir} does not exist")
        print("Analyzing all configurations without filtering")

    analyze_rows(rows, mad_k=args.mad_k, repro_threshold=args.repro_threshold,
                 use_mad_outlier=args.detect_outliers, outlier_mad_k=args.outlier_mad_k,
                 available_configs=available_configs, ignore_timeouts=args.ignore_timeouts)

    if args.plot:
        plot_box_whisker(rows, mad_k=args.mad_k, output_dir=Path(args.plot))


if __name__ == '__main__':
    main()
