#!/usr/bin/env python3
"""
Script to show summary correlations for all available multi-seed data.

This script scans for correlation matrices from multi-seed analysis and
generates a summary of correlations across all detector/dataset combinations.

Usage:
    python show_summary_correlations.py [--results-dir DIR] [--output FILE]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def find_correlation_data(base_dir):
    """
    Find all correlation matrices in the given directory structure.
    
    Args:
        base_dir: Base directory to search for correlation data
        
    Returns:
        dict: Dictionary mapping detector_dataset to correlation data
    """
    base_path = Path(base_dir)
    correlation_data = {}
    
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist")
        return correlation_data
    
    # Look for directories containing correlation matrices
    # Expected structure: base_dir/{detector}_{dataset}/
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        
        # Check if this directory contains correlation matrices
        acc_corr_file = item / "accuracy_correlation.csv"
        runtime_corr_file = item / "runtime_correlation.csv"
        pareto_corr_file = item / "pareto_correlation.csv"
        
        if acc_corr_file.exists() or runtime_corr_file.exists() or pareto_corr_file.exists():
            # Extract detector and dataset from directory name
            dir_name = item.name
            # Expected format: {detector}_{dataset}
            if '_' in dir_name:
                parts = dir_name.split('_', 1)
                detector = parts[0]
                dataset = parts[1]
            else:
                detector = dir_name
                dataset = "unknown"
            
            correlation_data[dir_name] = {
                'detector': detector,
                'dataset': dataset,
                'accuracy_correlation': None,
                'runtime_correlation': None,
                'pareto_correlation': None,
                'path': item
            }
            
            # Load correlation matrices if they exist
            if acc_corr_file.exists():
                try:
                    correlation_data[dir_name]['accuracy_correlation'] = pd.read_csv(acc_corr_file, index_col=0)
                except Exception as e:
                    print(f"Error loading {acc_corr_file}: {e}")
            
            if runtime_corr_file.exists():
                try:
                    correlation_data[dir_name]['runtime_correlation'] = pd.read_csv(runtime_corr_file, index_col=0)
                except Exception as e:
                    print(f"Error loading {runtime_corr_file}: {e}")
            
            if pareto_corr_file.exists():
                try:
                    correlation_data[dir_name]['pareto_correlation'] = pd.read_csv(pareto_corr_file, index_col=0)
                except Exception as e:
                    print(f"Error loading {pareto_corr_file}: {e}")
    
    return correlation_data


def compute_summary_stats(correlation_data):
    """
    Compute summary statistics from correlation data.
    
    Args:
        correlation_data: Dictionary of correlation data
        
    Returns:
        pd.DataFrame: Summary statistics
    """
    summary_list = []
    
    for key, data in correlation_data.items():
        detector = data['detector']
        dataset = data['dataset']
        
        # Process accuracy correlation
        acc_corr = data['accuracy_correlation']
        if acc_corr is not None and acc_corr.size > 0:
            acc_values = acc_corr.values[np.triu_indices_from(acc_corr.values, k=1)]
            acc_mean = np.nanmean(acc_values) if len(acc_values) > 0 else np.nan
            acc_std = np.nanstd(acc_values) if len(acc_values) > 0 else np.nan
            acc_min = np.nanmin(acc_values) if len(acc_values) > 0 else np.nan
            acc_max = np.nanmax(acc_values) if len(acc_values) > 0 else np.nan
            n_seeds_acc = acc_corr.shape[0]
        else:
            acc_mean = acc_std = acc_min = acc_max = np.nan
            n_seeds_acc = 0
        
        # Process runtime correlation
        runtime_corr = data['runtime_correlation']
        if runtime_corr is not None and runtime_corr.size > 0:
            runtime_values = runtime_corr.values[np.triu_indices_from(runtime_corr.values, k=1)]
            runtime_mean = np.nanmean(runtime_values) if len(runtime_values) > 0 else np.nan
            runtime_std = np.nanstd(runtime_values) if len(runtime_values) > 0 else np.nan
            runtime_min = np.nanmin(runtime_values) if len(runtime_values) > 0 else np.nan
            runtime_max = np.nanmax(runtime_values) if len(runtime_values) > 0 else np.nan
            n_seeds_runtime = runtime_corr.shape[0]
        else:
            runtime_mean = runtime_std = runtime_min = runtime_max = np.nan
            n_seeds_runtime = 0
        
        # Process Pareto correlation
        pareto_corr = data['pareto_correlation']
        if pareto_corr is not None and pareto_corr.size > 0:
            pareto_values = pareto_corr.values[np.triu_indices_from(pareto_corr.values, k=1)]
            pareto_mean = np.nanmean(pareto_values) if len(pareto_values) > 0 else np.nan
            pareto_std = np.nanstd(pareto_values) if len(pareto_values) > 0 else np.nan
            pareto_min = np.nanmin(pareto_values) if len(pareto_values) > 0 else np.nan
            pareto_max = np.nanmax(pareto_values) if len(pareto_values) > 0 else np.nan
            n_seeds_pareto = pareto_corr.shape[0]
        else:
            pareto_mean = pareto_std = pareto_min = pareto_max = np.nan
            n_seeds_pareto = 0
        
        summary_list.append({
            'detector': detector,
            'dataset': dataset,
            'n_seeds': max(n_seeds_acc, n_seeds_runtime, n_seeds_pareto),
            'mean_accuracy_correlation': acc_mean,
            'std_accuracy_correlation': acc_std,
            'min_accuracy_correlation': acc_min,
            'max_accuracy_correlation': acc_max,
            'mean_runtime_correlation': runtime_mean,
            'std_runtime_correlation': runtime_std,
            'min_runtime_correlation': runtime_min,
            'max_runtime_correlation': runtime_max,
            'mean_pareto_correlation': pareto_mean,
            'std_pareto_correlation': pareto_std,
            'min_pareto_correlation': pareto_min,
            'max_pareto_correlation': pareto_max,
        })
    
    return pd.DataFrame(summary_list)


def main():
    parser = argparse.ArgumentParser(description='Show summary correlations for multi-seed data')
    parser.add_argument('--results-dir', type=str, default='multi_seed_results',
                        help='Directory containing multi-seed correlation results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for summary (default: print to console)')
    parser.add_argument('--format', type=str, default='table',
                        choices=['table', 'csv'],
                        help='Output format')
    args = parser.parse_args()
    
    # Resolve path
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    
    print(f"Searching for correlation data in: {results_dir}")
    
    # Find correlation data
    correlation_data = find_correlation_data(results_dir)
    
    if not correlation_data:
        print(f"No correlation data found in {results_dir}")
        print("\nLooking for alternative locations...")
        
        # Try common alternative locations
        alternative_dirs = [
            script_dir / 'reproducibility' / 'multi_seed_results',
            script_dir / 'multi_seed_analysis_results',
            Path('/home/eliasw/benchmark-unsupervised-concept-drift-detection/multi_seed_results'),
        ]
        
        for alt_dir in alternative_dirs:
            print(f"  Checking: {alt_dir}")
            correlation_data = find_correlation_data(alt_dir)
            if correlation_data:
                print(f"  Found data in: {alt_dir}")
                results_dir = alt_dir
                break
        
        if not correlation_data:
            print("No correlation data found in any location.")
            print("\nIf you have correlation matrices in a different location, specify it with --results-dir")
            sys.exit(1)
    
    print(f"Found correlation data for {len(correlation_data)} detector/dataset combinations")
    
    # Compute summary statistics
    summary_df = compute_summary_stats(correlation_data)
    
    # Sort by detector and dataset
    summary_df = summary_df.sort_values(['detector', 'dataset'])
    
    # Output results
    if args.format == 'table':
        print("\n" + "="*100)
        print("SUMMARY CORRELATIONS")
        print("="*100)
        print(summary_df.to_string(index=False))
        
        # Print overall statistics
        print("\n" + "="*100)
        print("OVERALL STATISTICS")
        print("="*100)
        
        valid_acc = summary_df['mean_accuracy_correlation'].dropna()
        valid_runtime = summary_df['mean_runtime_correlation'].dropna()
        valid_pareto = summary_df['mean_pareto_correlation'].dropna()
        
        if len(valid_acc) > 0:
            print(f"Accuracy Correlation:")
            print(f"  Mean across all experiments: {valid_acc.mean():.4f} ± {valid_acc.std():.4f}")
            print(f"  Range: [{valid_acc.min():.4f}, {valid_acc.max():.4f}]")
            print(f"  Number of experiments: {len(valid_acc)}")
        
        if len(valid_runtime) > 0:
            print(f"\nRuntime Correlation:")
            print(f"  Mean across all experiments: {valid_runtime.mean():.4f} ± {valid_runtime.std():.4f}")
            print(f"  Range: [{valid_runtime.min():.4f}, {valid_runtime.max():.4f}]")
            print(f"  Number of experiments: {len(valid_runtime)}")
        
        if len(valid_pareto) > 0:
            print(f"\nPareto Correlation:")
            print(f"  Mean across all experiments: {valid_pareto.mean():.4f} ± {valid_pareto.std():.4f}")
            print(f"  Range: [{valid_pareto.min():.4f}, {valid_pareto.max():.4f}]")
            print(f"  Number of experiments: {len(valid_pareto)}")
    
    elif args.format == 'csv':
        print(summary_df.to_csv(index=False))
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        if args.format == 'csv':
            summary_df.to_csv(output_path, index=False)
        else:
            summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to {output_path}")


if __name__ == '__main__':
    main()
