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
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from scipy.stats import spearmanr
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
            df = df[df['Status'] == 'Completed']
        
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
            
            value_str = str(value)
            if value_str.startswith("str('") and value_str.endswith("')"):
                actual_value = value_str[5:-2]
                cmd.append(actual_value)
            elif value_str.startswith('str("') and value_str.endswith('")'):
                actual_value = value_str[5:-2]
                cmd.append(actual_value)
            else:
                cmd.append(value_str)
    
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


def compute_correlation_matrix(values_dict):
    """
    Compute Spearman correlation matrix for multiple runs.
    
    Args:
        values_dict: Dictionary mapping seed to list of values
        
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
                    try:
                        corr, _ = spearmanr(vals1, vals2)
                        corr_matrix.loc[seed1, seed2] = corr
                    except:
                        corr_matrix.loc[seed1, seed2] = np.nan
                else:
                    corr_matrix.loc[seed1, seed2] = np.nan
    
    return corr_matrix


def main():
    parser = argparse.ArgumentParser(description='Multi-seed reproducibility analysis')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='multi_seed_results', help='Output directory for results')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout per experiment in seconds')
    parser.add_argument('--results-dir', type=str, default='all_benchmark_results', help='Directory containing benchmark results')
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds to use')
    parser.add_argument('--seed-start', type=int, default=42, help='Starting seed value')
    args = parser.parse_args()
    
    # Generate seeds
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"Using seeds: {seeds}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all CSV files
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    if not results_dir.is_absolute():
        results_dir = results_dir.resolve()
    csv_files = list(results_dir.glob('*/*/*.csv'))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    all_results = {}
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}")
        
        parts = csv_file.parts
        detector = parts[-3]
        dataset = parts[-2]
        filename = csv_file.name
        
        mode, flags = parse_mode_from_filename(filename)
        
        configs = extract_configurations(csv_file)
        print(f"  Found {len(configs)} configurations for {detector} on {dataset}")
        
        if not configs:
            continue
        
        # Limit configurations for testing (remove this for full run)
        # configs = configs[:10]
        
        # Store results for each seed
        seed_results = {seed: [] for seed in seeds}
        
        # Run each configuration with all seeds
        for config_idx, config in enumerate(configs):
            print(f"  Processing config {config_idx + 1}/{len(configs)}", end='\r')
            
            for seed in seeds:
                cmd = build_command(detector, dataset, mode, flags, config, seed)
                
                success, metrics, error = run_single_experiment(cmd, args.timeout)
                
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
        
        print(f"\n  Completed {len(configs)} configurations")
        
        # Extract accuracy and runtime arrays for each seed
        accuracy_by_seed = {}
        runtime_by_seed = {}
        
        for seed in seeds:
            accuracies = [r['accuracy'] for r in seed_results[seed] if r['accuracy'] is not None]
            runtimes = [r['runtime'] for r in seed_results[seed] if r['runtime'] is not None]
            accuracy_by_seed[seed] = accuracies
            runtime_by_seed[seed] = runtimes
        
        # Compute correlation matrices
        accuracy_corr = compute_correlation_matrix(accuracy_by_seed)
        runtime_corr = compute_correlation_matrix(runtime_by_seed)
        
        # Compute Pareto ranks for each seed
        pareto_ranks_by_seed = {}
        for seed in seeds:
            if len(accuracy_by_seed[seed]) > 0 and len(runtime_by_seed[seed]) > 0:
                pareto_ranks_by_seed[seed] = compute_pareto_rank(
                    accuracy_by_seed[seed], 
                    runtime_by_seed[seed]
                )
        
        # Compute Pareto rank correlation matrix
        pareto_corr = compute_correlation_matrix(pareto_ranks_by_seed)
        
        # Store results
        key = f"{detector}_{dataset}"
        all_results[key] = {
            'detector': detector,
            'dataset': dataset,
            'accuracy_correlation': accuracy_corr,
            'runtime_correlation': runtime_corr,
            'pareto_correlation': pareto_corr,
            'seed_results': seed_results
        }
        
        # Save per-experiment results
        exp_dir = output_dir / f"{detector}_{dataset}"
        exp_dir.mkdir(exist_ok=True)
        
        accuracy_corr.to_csv(exp_dir / "accuracy_correlation.csv")
        runtime_corr.to_csv(exp_dir / "runtime_correlation.csv")
        pareto_corr.to_csv(exp_dir / "pareto_correlation.csv")
        
        print(f"  Saved correlation matrices to {exp_dir}")
    
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
        
        # Calculate mean off-diagonal correlations
        acc_values = acc_corr.values[np.triu_indices_from(acc_corr.values, k=1)]
        runtime_values = runtime_corr.values[np.triu_indices_from(runtime_corr.values, k=1)]
        pareto_values = pareto_corr.values[np.triu_indices_from(pareto_corr.values, k=1)]
        
        summary_data.append({
            'detector': detector,
            'dataset': dataset,
            'mean_accuracy_correlation': np.nanmean(acc_values) if len(acc_values) > 0 else np.nan,
            'mean_runtime_correlation': np.nanmean(runtime_values) if len(runtime_values) > 0 else np.nan,
            'mean_pareto_correlation': np.nanmean(pareto_values) if len(pareto_values) > 0 else np.nan,
            'std_accuracy_correlation': np.nanstd(acc_values) if len(acc_values) > 0 else np.nan,
            'std_runtime_correlation': np.nanstd(runtime_values) if len(runtime_values) > 0 else np.nan,
            'std_pareto_correlation': np.nanstd(pareto_values) if len(pareto_values) > 0 else np.nan,
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string())
    
    # Save summary
    summary_df.to_csv(output_dir / "summary_correlations.csv", index=False)
    print(f"\nSummary saved to {output_dir / 'summary_correlations.csv'}")


if __name__ == '__main__':
    main()
