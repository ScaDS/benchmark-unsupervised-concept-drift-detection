#!/usr/bin/env python3
"""
Multi-Seed Reproducibility Analysis Script

This script runs benchmark experiments with 10 different random seeds and computes
correlations between the results across different seeds.

Usage:
    python multi_seed_analysis.py [--max-workers N] [--output-dir DIR] [--seeds N]
"""

import os
import sys
import csv
import ast
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import copy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_mode_from_filename(filename):
    """
    Determine the benchmark mode from the CSV filename.
    
    Returns:
        tuple: (mode, flags) where mode is 'standard' or 'mtr' and flags is a list of booleans
    """
    if 'MTR' in filename:
        return 'mtr', [True, False]
    elif 'ACC' in filename and 'RT' in filename and 'REQL' in filename:
        return 'standard', [True, True, True]
    elif 'ACC' in filename and 'RT' in filename:
        return 'standard', [True, True, False]
    elif 'ACC' in filename and 'REQL' in filename:
        return 'standard', [True, False, True]
    elif 'RT' in filename and 'REQL' in filename:
        return 'standard', [False, True, True]
    elif 'ACC' in filename:
        return 'standard', [True, False, False]
    elif 'RT' in filename:
        return 'standard', [False, True, False]
    elif 'REQL' in filename:
        return 'standard', [False, False, True]
    else:
        return 'standard', [True, True, False]


def extract_configurations(csv_path):
    """
    Extract configurations and original metrics from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        list: List of dictionaries containing configuration parameters and original metrics
    """
    configs = []
    
    try:
        df = pd.read_csv(csv_path)
        
        if 'Status' in df.columns:
            df = df[df['Status'] == 'COMPLETED']
        
        metric_columns = {'Status', 'ACCURACY', 'RUNTIME', 'REQLABELS', 'MTR', 'OO-Info'}
        param_columns = [col for col in df.columns if col not in metric_columns]
        
        for _, row in df.iterrows():
            config = {}
            for param in param_columns:
                config[param] = row[param]
            
            config['_original_accuracy'] = row.get('ACCURACY', None)
            config['_original_runtime'] = row.get('RUNTIME', None)
            config['_original_reqlabels'] = row.get('REQLABELS', None)
            config['_original_mtr'] = row.get('MTR', None)
            
            configs.append(config)
            
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        
    return configs


def _format_param_value(value) -> str:
    """Format a parameter value from a benchmark CSV for use as a command-line argument.

    Handles:
    - str('...') / str("...") wrappers from benchmark configs
    - Backslash-escaped characters from CSV storage
    - Python literals (dicts, lists, numbers, booleans) — passed as-is
    - Enum-like expressions (e.g. EDFSMode.RANDOM) — passed as-is
    - Plain strings — wrapped in quotes for eval in main.py
    """
    value_str = str(value)

    # Handle str('...') / str("...") wrappers
    if value_str.startswith("str('") and value_str.endswith("')"):
        return value_str[5:-2]
    if value_str.startswith('str("') and value_str.endswith('")'):
        return value_str[5:-2]

    # Clean backslash escapes that may come from CSV storage
    if '\\' in value_str:
        value_str = value_str.replace("\\'", "'").replace('\\"', '"')
        value_str = value_str.replace('\\(', '(').replace('\\)', ')')
        value_str = value_str.replace('\\{', '{').replace('\\}', '}')

    # Try ast.literal_eval — if it succeeds, the value is a valid Python literal
    # (dict, list, number, bool, None, nested string) and can be passed as-is
    try:
        ast.literal_eval(value_str)
        return value_str
    except (ValueError, SyntaxError):
        pass

    # Check if it's a number
    try:
        float(value_str)
        return value_str
    except ValueError:
        pass

    # Check if it's a boolean
    if value_str.lower() in ['true', 'false']:
        return value_str

    # Check if it looks like a Python expression (e.g. EDFSMode.RANDOM)
    # These are resolved by eval() in main.py and should NOT be quoted
    try:
        ast.parse(value_str, mode='eval')
        # Only pass as-is if it's not a plain string literal
        node = ast.parse(value_str, mode='eval').body
        if not isinstance(node, ast.Constant):
            return value_str
    except SyntaxError:
        pass

    # It's a plain string — wrap in quotes for eval in main.py
    return f"'{value_str}'"


def build_command(detector, dataset, mode, flags, config, seed, n_training_samples=2000, classifier="HoeffdingTreeClassifier"):
    """
    Build the command line arguments for running the benchmark with a seed.
    
    Args:
        detector: Detector name
        dataset: Dataset name
        mode: 'standard' or 'mtr'
        flags: List of boolean flags
        config: Dictionary of detector parameters
        seed: Random seed value
        n_training_samples: Number of training samples
        classifier: Classifier name
        
    Returns:
        list: Command line arguments
    """
    cmd = ['python', 'main.py']
    
    cmd.extend([str(flag) for flag in flags])
    cmd.append(dataset)
    cmd.append(str(n_training_samples))
    cmd.append(classifier)
    cmd.append(detector)
    
    # Add seed parameter
    cmd.append('seed')
    cmd.append(str(seed))
    
    # Add detector parameters
    for key, value in config.items():
        if not key.startswith('_'):
            cmd.append(str(key))
            cmd.append(_format_param_value(value))
    
    return cmd


def run_single_experiment(cmd, timeout=7200):
    """
    Run a single benchmark experiment and extract metrics from output.
    
    Args:
        cmd: Command to run
        timeout: Timeout in seconds
        
    Returns:
        tuple: (success, metrics_dict, error_message)
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            return False, None, result.stderr[:500]
        
        metrics = {}
        output = result.stdout + result.stderr
        
        for line in output.split('\n'):
            if 'ACCURACY:' in line:
                try:
                    metrics['accuracy'] = float(line.split('ACCURACY:')[1].strip())
                except:
                    pass
            elif 'RUNTIME:' in line:
                try:
                    metrics['runtime'] = float(line.split('RUNTIME:')[1].strip())
                except:
                    pass
            elif 'REQLABELS:' in line:
                try:
                    metrics['reqlabels'] = float(line.split('REQLABELS:')[1].strip())
                except:
                    pass
            elif 'MTR:' in line:
                try:
                    metrics['mtr'] = float(line.split('MTR:')[1].strip())
                except:
                    pass
            elif 'OO-Info: accuracy:' in line:
                try:
                    metrics['accuracy'] = float(line.split('OO-Info: accuracy:')[1].strip())
                except:
                    pass
            elif 'OO-Info: runtime:' in line:
                try:
                    metrics['runtime'] = float(line.split('OO-Info: runtime:')[1].strip())
                except:
                    pass
            elif 'OO-Info: portion_req_label:' in line:
                try:
                    metrics['reqlabels'] = float(line.split('OO-Info: portion_req_label:')[1].strip())
                except:
                    pass
            elif 'OO-Info: mtr:' in line:
                try:
                    metrics['mtr'] = float(line.split('OO-Info: mtr:')[1].strip())
                except:
                    pass
        
        return True, metrics, None
            
    except subprocess.TimeoutExpired:
        return False, None, "Timeout"
    except Exception as e:
        return False, None, str(e)[:500]


def compute_pareto_rank(accuracies, runtimes, maximize_accuracy=True, minimize_runtime=True):
    """
    Compute Pareto rank for configurations based on accuracy and runtime.
    
    Args:
        accuracies: List of accuracy values
        runtimes: List of runtime values
        maximize_accuracy: Whether to maximize accuracy (default: True)
        minimize_runtime: Whether to minimize runtime (default: True)
        
    Returns:
        list: Pareto ranks for each configuration
    """
    n = len(accuracies)
    ranks = [0] * n
    
    for i in range(n):
        rank = 1
        for j in range(n):
            if i == j:
                continue
            
            # Check if j dominates i
            acc_better = (accuracies[j] > accuracies[i]) if maximize_accuracy else (accuracies[j] < accuracies[i])
            time_better = (runtimes[j] < runtimes[i]) if minimize_runtime else (runtimes[j] > runtimes[i])
            
            if acc_better and time_better:
                rank += 1
            elif acc_better and runtimes[j] == runtimes[i]:
                rank += 1
            elif time_better and accuracies[j] == accuracies[i]:
                rank += 1
        
        ranks[i] = rank
    
    return ranks


def pareto_front_2d(points):
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


def compute_hypervolume_2d(points, ref_acc, ref_rt):
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


def compute_correlation_matrix(values_dict, method='spearman'):
    """
    Compute correlation matrix for multiple runs.
    
    Args:
        values_dict: Dictionary mapping seed to list of values
        method: 'spearman' or 'pearson'
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    seeds = sorted(values_dict.keys())
    n = len(seeds)
    corr_matrix = pd.DataFrame(index=seeds, columns=seeds)
    
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds):
            if i == j:
                corr_matrix.loc[seed1, seed2] = 1.0
            else:
                vals1 = values_dict[seed1]
                vals2 = values_dict[seed2]
                
                # Only compute if we have matching data
                if len(vals1) == len(vals2) and len(vals1) > 0:
                    arr1 = np.array(vals1, dtype=float)
                    arr2 = np.array(vals2, dtype=float)
                    # Pearson is undefined for constant inputs; treat
                    # identical constant arrays as perfectly correlated.
                    if method == 'pearson':
                        if np.std(arr1) == 0 and np.std(arr2) == 0:
                            corr = 1.0 if np.array_equal(arr1, arr2) else 0.0
                        elif np.std(arr1) == 0 or np.std(arr2) == 0:
                            corr = np.nan
                        else:
                            try:
                                corr, _ = pearsonr(vals1, vals2)
                            except Exception:
                                corr = np.nan
                        corr_matrix.loc[seed1, seed2] = corr
                    else:
                        try:
                            corr, _ = spearmanr(vals1, vals2)
                            corr_matrix.loc[seed1, seed2] = corr
                        except Exception:
                            corr_matrix.loc[seed1, seed2] = np.nan
                else:
                    corr_matrix.loc[seed1, seed2] = np.nan
    
    return corr_matrix


def main():
    parser = argparse.ArgumentParser(description='Multi-seed reproducibility analysis')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='multi_seed_results', help='Output directory for results')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout per experiment in seconds')
    parser.add_argument('--results-dir', type=str, default='results/all_benchmark_results', help='Directory containing benchmark results')
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds to use')
    parser.add_argument('--seed-start', type=int, default=42, help='Starting seed value')
    parser.add_argument('--detector', type=str, default=None, help='Only process specific detector')
    parser.add_argument('--dataset', type=str, default=None, help='Only process specific dataset')
    args = parser.parse_args()
    
    # Generate seeds
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"Using seeds: {seeds}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all CSV files
    script_dir = Path(__file__).parent.parent  # Go up to project root
    results_dir = script_dir / args.results_dir
    if not results_dir.is_absolute():
        results_dir = results_dir.resolve()
    
    print(f"Looking for CSV files in: {results_dir}")
    print(f"Results dir exists: {results_dir.exists()}")
    print(f"Results dir is absolute: {results_dir.is_absolute()}")
    print(f"Script file location: {Path(__file__).resolve()}")
    print(f"Script parent: {Path(__file__).parent.resolve()}")
    print(f"Script parent.parent: {Path(__file__).parent.parent.resolve()}")
    
    csv_files = list(results_dir.glob('*/*/*.csv'))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    all_results = {}
    
    for csv_file in csv_files:
        # Extract detector and dataset from path for filtering
        parts = csv_file.parts
        detector = parts[-3]
        dataset = parts[-2]
        
        # Filter by detector/dataset if specified
        if args.detector and detector != args.detector:
            continue
        if args.dataset and dataset != args.dataset:
            continue

        if dataset in ('WaveformDrift2', 'SineClusters'):
            continue
        
        filename = csv_file.name
        print(f"\nProcessing {csv_file}")
        
        mode, flags = parse_mode_from_filename(filename)
        
        configs = extract_configurations(csv_file)
        print(f"  Found {len(configs)} configurations for {detector} on {dataset}")
        
        if not configs:
            continue
        
        # Limit configurations for testing (remove this for full run)
        # configs = configs[:10]
        
        # Store results for each seed
        seed_results = {seed: [] for seed in seeds}
        
        # Run each configuration with all seeds in parallel
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            
            for config_idx, config in enumerate(configs):
                for seed in seeds:
                    cmd = build_command(detector, dataset, mode, flags, config, seed)
                    future = executor.submit(run_single_experiment, cmd, args.timeout)
                    futures[future] = (config_idx, seed, copy.deepcopy(config))
            
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                config_idx, seed, config = futures[future]
                try:
                    success, metrics, error = future.result()
                    
                    if success and metrics:
                        seed_results[seed].append({
                            'config_idx': config_idx,
                            'accuracy': metrics.get('accuracy'),
                            'runtime': metrics.get('runtime'),
                            'reqlabels': metrics.get('reqlabels'),
                            'mtr': metrics.get('mtr')
                        })
                    else:
                        seed_results[seed].append({
                            'config_idx': config_idx,
                            'accuracy': None,
                            'runtime': None,
                            'reqlabels': None,
                            'mtr': None,
                            'error': error
                        })
                    
                    completed += 1
                    print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)", end='\r')
                    
                except Exception as e:
                    print(f"  Error processing config {config_idx} seed {seed}: {e}")
                    seed_results[seed].append({
                        'config_idx': config_idx,
                        'accuracy': None,
                        'runtime': None,
                        'reqlabels': None,
                        'mtr': None,
                        'error': str(e)
                    })
        
        print(f"\n  Completed {len(configs)} configurations")
        
        # Extract accuracy and runtime arrays for each seed
        accuracy_by_seed = {}
        runtime_by_seed = {}
        
        for seed in seeds:
            accuracies = [r['accuracy'] for r in seed_results[seed] if r['accuracy'] is not None]
            runtimes = [r['runtime'] for r in seed_results[seed] if r['runtime'] is not None]
            accuracy_by_seed[seed] = accuracies
            runtime_by_seed[seed] = runtimes
        
        # Compute Spearman correlation matrices
        accuracy_corr = compute_correlation_matrix(accuracy_by_seed, method='spearman')
        runtime_corr = compute_correlation_matrix(runtime_by_seed, method='spearman')
        
        # Compute Pearson correlation matrices
        accuracy_corr_pearson = compute_correlation_matrix(accuracy_by_seed, method='pearson')
        runtime_corr_pearson = compute_correlation_matrix(runtime_by_seed, method='pearson')
        
        # Compute Pareto ranks for each seed
        pareto_ranks_by_seed = {}
        for seed in seeds:
            if len(accuracy_by_seed[seed]) > 0 and len(runtime_by_seed[seed]) > 0:
                pareto_ranks_by_seed[seed] = compute_pareto_rank(
                    accuracy_by_seed[seed], 
                    runtime_by_seed[seed]
                )
        
        # Compute Pareto rank correlation matrices
        pareto_corr = compute_correlation_matrix(pareto_ranks_by_seed, method='spearman')
        pareto_corr_pearson = compute_correlation_matrix(pareto_ranks_by_seed, method='pearson')

        # Compute hypervolume for each seed using a common reference point
        # The reference point is derived from all seeds' data combined
        all_accs_combined = []
        all_rts_combined = []
        for seed in seeds:
            all_accs_combined.extend(accuracy_by_seed[seed])
            all_rts_combined.extend(runtime_by_seed[seed])

        hv_by_seed = {}
        if all_accs_combined and all_rts_combined:
            ref_acc = min(all_accs_combined)
            ref_rt = max(all_rts_combined)
            for seed in seeds:
                accs = accuracy_by_seed[seed]
                rts = runtime_by_seed[seed]
                if len(accs) > 0 and len(rts) > 0:
                    points = list(zip(accs, rts))
                    hv_by_seed[seed] = compute_hypervolume_2d(points, ref_acc, ref_rt)
                else:
                    hv_by_seed[seed] = 0.0

            # Print HV summary for this detector/dataset
            hv_values_list = [hv_by_seed[s] for s in seeds]
            hv_mean = np.nanmean(hv_values_list) if hv_values_list else 0.0
            hv_std = np.nanstd(hv_values_list) if hv_values_list else 0.0
            hv_cv = (hv_std / hv_mean) if hv_mean > 0 else float('nan')
            print(f"  Hypervolume by seed: {dict((s, round(hv_by_seed[s], 4)) for s in seeds)}")
            print(f"  HV stability: mean={hv_mean:.4f}, std={hv_std:.4f}, CV={hv_cv:.4f}")
        else:
            hv_by_seed = {seed: 0.0 for seed in seeds}

        # Print per-seed summary alongside existing correlations
        for seed in seeds:
            accs = accuracy_by_seed[seed]
            rts = runtime_by_seed[seed]
            acc_mean = np.nanmean(accs) if accs else float('nan')
            rt_mean = np.nanmean(rts) if rts else float('nan')
            print(f"  Seed {seed}: acc_mean={acc_mean:.4f}, rt_mean={rt_mean:.4f}, "
                  f"hv={hv_by_seed.get(seed, 0.0):.4f}")

        # Store results
        key = f"{detector}_{dataset}"
        all_results[key] = {
            'detector': detector,
            'dataset': dataset,
            'accuracy_correlation': accuracy_corr,
            'runtime_correlation': runtime_corr,
            'pareto_correlation': pareto_corr,
            'accuracy_correlation_pearson': accuracy_corr_pearson,
            'runtime_correlation_pearson': runtime_corr_pearson,
            'pareto_correlation_pearson': pareto_corr_pearson,
            'hypervolume_by_seed': hv_by_seed,
            'seed_results': seed_results
        }
        
        # Save per-experiment results
        exp_dir = output_dir / f"{detector}_{dataset}"
        exp_dir.mkdir(exist_ok=True)
        
        accuracy_corr.to_csv(exp_dir / "accuracy_correlation.csv")
        runtime_corr.to_csv(exp_dir / "runtime_correlation.csv")
        pareto_corr.to_csv(exp_dir / "pareto_correlation.csv")
        accuracy_corr_pearson.to_csv(exp_dir / "accuracy_correlation_pearson.csv")
        runtime_corr_pearson.to_csv(exp_dir / "runtime_correlation_pearson.csv")
        pareto_corr_pearson.to_csv(exp_dir / "pareto_correlation_pearson.csv")

        # Save per-seed metrics (hypervolume + raw accuracy/runtime means)
        per_seed_rows = []
        for seed in seeds:
            accs = accuracy_by_seed[seed]
            rts = runtime_by_seed[seed]
            per_seed_rows.append({
                'seed': seed,
                'detector': detector,
                'dataset': dataset,
                'hypervolume': hv_by_seed.get(seed, 0.0),
                'accuracy_mean': np.nanmean(accs) if accs else float('nan'),
                'accuracy_std': np.nanstd(accs) if accs else float('nan'),
                'runtime_mean': np.nanmean(rts) if rts else float('nan'),
                'runtime_std': np.nanstd(rts) if rts else float('nan'),
            })
        per_seed_df = pd.DataFrame(per_seed_rows)
        per_seed_df.to_csv(exp_dir / "per_seed_metrics.csv", index=False)

        # Save raw per-config per-seed results for post-hoc analysis (e.g. Pearson)
        raw_rows = []
        for seed in seeds:
            for r in seed_results[seed]:
                raw_rows.append({
                    'seed': seed,
                    'config_idx': r['config_idx'],
                    'accuracy': r.get('accuracy'),
                    'runtime': r.get('runtime'),
                    'reqlabels': r.get('reqlabels'),
                    'mtr': r.get('mtr'),
                })
        if raw_rows:
            pd.DataFrame(raw_rows).to_csv(exp_dir / "raw_per_config.csv", index=False)

        print(f"  Saved correlation matrices and per-seed metrics to {exp_dir}")
    
    # Generate summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    summary_data = []
    for key, results in all_results.items():
        detector = results['detector']
        dataset = results['dataset']
        
        # Get mean correlation values (excluding diagonal)
        acc_corr = results['accuracy_correlation']
        runtime_corr = results['runtime_correlation']
        pareto_corr = results['pareto_correlation']
        acc_corr_pearson = results['accuracy_correlation_pearson']
        runtime_corr_pearson = results['runtime_correlation_pearson']
        pareto_corr_pearson = results['pareto_correlation_pearson']
        
        # Calculate mean off-diagonal correlations
        acc_values = acc_corr.values[np.triu_indices_from(acc_corr.values, k=1)]
        runtime_values = runtime_corr.values[np.triu_indices_from(runtime_corr.values, k=1)]
        pareto_values = pareto_corr.values[np.triu_indices_from(pareto_corr.values, k=1)]
        acc_values_pearson = acc_corr_pearson.values[np.triu_indices_from(acc_corr_pearson.values, k=1)]
        runtime_values_pearson = runtime_corr_pearson.values[np.triu_indices_from(runtime_corr_pearson.values, k=1)]
        pareto_values_pearson = pareto_corr_pearson.values[np.triu_indices_from(pareto_corr_pearson.values, k=1)]
        
        # Hypervolume statistics (stability across seeds)
        hv_by_seed = results.get('hypervolume_by_seed', {})
        hv_values = list(hv_by_seed.values())
        hv_mean = np.nanmean(hv_values) if hv_values else np.nan
        hv_std = np.nanstd(hv_values) if hv_values else np.nan
        hv_cv = (hv_std / hv_mean) if hv_mean and hv_mean > 0 else float('nan')

        summary_data.append({
            'detector': detector,
            'dataset': dataset,
            'mean_accuracy_correlation': np.nanmean(acc_values) if len(acc_values) > 0 else np.nan,
            'mean_runtime_correlation': np.nanmean(runtime_values) if len(runtime_values) > 0 else np.nan,
            'mean_pareto_correlation': np.nanmean(pareto_values) if len(pareto_values) > 0 else np.nan,
            'std_accuracy_correlation': np.nanstd(acc_values) if len(acc_values) > 0 else np.nan,
            'std_runtime_correlation': np.nanstd(runtime_values) if len(runtime_values) > 0 else np.nan,
            'std_pareto_correlation': np.nanstd(pareto_values) if len(pareto_values) > 0 else np.nan,
            'mean_accuracy_correlation_pearson': np.nanmean(acc_values_pearson) if len(acc_values_pearson) > 0 else np.nan,
            'mean_runtime_correlation_pearson': np.nanmean(runtime_values_pearson) if len(runtime_values_pearson) > 0 else np.nan,
            'mean_pareto_correlation_pearson': np.nanmean(pareto_values_pearson) if len(pareto_values_pearson) > 0 else np.nan,
            'std_accuracy_correlation_pearson': np.nanstd(acc_values_pearson) if len(acc_values_pearson) > 0 else np.nan,
            'std_runtime_correlation_pearson': np.nanstd(runtime_values_pearson) if len(runtime_values_pearson) > 0 else np.nan,
            'std_pareto_correlation_pearson': np.nanstd(pareto_values_pearson) if len(pareto_values_pearson) > 0 else np.nan,
            'mean_hypervolume': hv_mean,
            'std_hypervolume': hv_std,
            'cv_hypervolume': hv_cv,
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string())

    # Print correlation + stability comparison
    print("\n" + "="*90)
    print("CROSS-SEED STABILITY COMPARISON")
    print("="*90)
    print(f"{'Detector':<22} {'Dataset':<22} {'AccSp':>8} {'RtSp':>8} {'ParSp':>8} {'AccPe':>8} {'RtPe':>8} {'ParPe':>8} {'HV CV':>8}")
    print("-" * 120)
    for _, row in summary_df.iterrows():
        print(f"{row['detector']:<22} {row['dataset']:<22} "
              f"{row['mean_accuracy_correlation']:>8.4f} {row['mean_runtime_correlation']:>8.4f} "
              f"{row['mean_pareto_correlation']:>8.4f} "
              f"{row['mean_accuracy_correlation_pearson']:>8.4f} {row['mean_runtime_correlation_pearson']:>8.4f} "
              f"{row['mean_pareto_correlation_pearson']:>8.4f} "
              f"{row['cv_hypervolume']:>8.4f}")

    # Overall aggregation
    valid_acc_corr = summary_df['mean_accuracy_correlation'].dropna()
    valid_rt_corr = summary_df['mean_runtime_correlation'].dropna()
    valid_hv_cv = summary_df['cv_hypervolume'].dropna()
    valid_acc_corr_pearson = summary_df['mean_accuracy_correlation_pearson'].dropna()
    valid_rt_corr_pearson = summary_df['mean_runtime_correlation_pearson'].dropna()
    valid_par_corr_pearson = summary_df['mean_pareto_correlation_pearson'].dropna()
    valid_par_corr = summary_df['mean_pareto_correlation'].dropna()
    if len(valid_hv_cv) > 0:
        print("-" * 120)
        print(f"Overall (mean across detector/dataset combos):")
        print(f"  Accuracy Spearman corr: {valid_acc_corr.mean():.4f}  Pearson corr: {valid_acc_corr_pearson.mean():.4f}")
        print(f"  Runtime Spearman corr:  {valid_rt_corr.mean():.4f}  Pearson corr: {valid_rt_corr_pearson.mean():.4f}")
        print(f"  Pareto Spearman corr:   {valid_par_corr.mean():.4f}  Pearson corr: {valid_par_corr_pearson.mean():.4f}")
        print(f"  Hypervolume CV:         {valid_hv_cv.mean():.4f}")
        print(f"  (Lower HV CV = more stable hypervolume across seeds)")

    # Save summary
    summary_df.to_csv(output_dir / "summary_correlations.csv", index=False)
    print(f"\nSummary saved to {output_dir / 'summary_correlations.csv'}")


if __name__ == '__main__':
    main()
