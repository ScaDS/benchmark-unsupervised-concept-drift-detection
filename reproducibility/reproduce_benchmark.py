#!/usr/bin/env python3
"""
Reproducibility Script for Benchmark Results

This script takes all results from all_benchmark_results and attempts to reproduce
them by running the benchmark with the same configurations. It runs experiments in
parallel and generates reports on reproducibility success/failure.

Usage:
    python reproduce_benchmark.py [--max-workers N] [--output-dir DIR]
"""

import os
import sys
import csv
import glob
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import traceback
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
        return 'mtr', [True, False]  # Runtime=True, MTR=True
    elif 'ACC' in filename and 'RT' in filename and 'REQL' in filename:
        return 'standard', [True, True, True]  # Accuracy=True, Runtime=True, ReqLabels=True
    elif 'ACC' in filename and 'RT' in filename:
        return 'standard', [True, True, False]  # Accuracy=True, Runtime=True, ReqLabels=False
    elif 'ACC' in filename and 'REQL' in filename:
        return 'standard', [True, False, True]  # Accuracy=True, Runtime=False, ReqLabels=True
    elif 'RT' in filename and 'REQL' in filename:
        return 'standard', [False, True, True]  # Accuracy=False, Runtime=True, ReqLabels=True
    elif 'ACC' in filename:
        return 'standard', [True, False, False]  # Accuracy=True, Runtime=False, ReqLabels=False
    elif 'RT' in filename:
        return 'standard', [False, True, False]  # Accuracy=False, Runtime=True, ReqLabels=False
    elif 'REQL' in filename:
        return 'standard', [False, False, True]  # Accuracy=False, Runtime=False, ReqLabels=True
    else:
        # Default to standard mode with accuracy and runtime
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
        
        # Filter out rows that are not completed
        if 'Status' in df.columns:
            df = df[df['Status'] == 'Completed']
        
        # Get parameter columns (exclude metric columns)
        metric_columns = {'Status', 'ACCURACY', 'RUNTIME', 'REQLABELS', 'MTR', 'OO-Info'}
        param_columns = [col for col in df.columns if col not in metric_columns]
        
        for _, row in df.iterrows():
            config = {}
            for param in param_columns:
                config[param] = row[param]
            
            # Store original metrics
            config['_original_accuracy'] = row.get('ACCURACY', None)
            config['_original_runtime'] = row.get('RUNTIME', None)
            config['_original_reqlabels'] = row.get('REQLABELS', None)
            config['_original_mtr'] = row.get('MTR', None)
            
            configs.append(config)
            
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        
    return configs


def build_command(detector, dataset, mode, flags, config, n_training_samples=2000, classifier="HoeffdingTreeClassifier"):
    """
    Build the command line arguments for running the benchmark.
    
    Args:
        detector: Detector name
        dataset: Dataset name
        mode: 'standard' or 'mtr'
        flags: List of boolean flags
        config: Dictionary of detector parameters
        n_training_samples: Number of training samples
        classifier: Classifier name
        
    Returns:
        list: Command line arguments
    """
    cmd = ['python', 'main.py']
    
    # Add flags (use Python True/False, not lowercase)
    cmd.extend([str(flag) for flag in flags])
    
    # Add dataset
    cmd.append(dataset)
    
    # Add training samples
    cmd.append(str(n_training_samples))
    
    # Add classifier
    cmd.append(classifier)
    
    # Add detector and parameters
    cmd.append(detector)
    
    # Add detector parameters as key-value pairs (skip internal metrics)
    for key, value in config.items():
        if not key.startswith('_'):
            cmd.append(str(key))
            
            # Handle string parameters that are stored as str('...') in CSV
            value_str = str(value)
            if value_str.startswith("str('") and value_str.endswith("')"):
                # Extract the actual string value from str('...')
                actual_value = value_str[5:-2]  # Remove str(' and ')
                cmd.append(actual_value)
            elif value_str.startswith('str("') and value_str.endswith('")'):
                # Extract the actual string value from str("...")
                actual_value = value_str[5:-2]  # Remove str(" and ")
                cmd.append(actual_value)
            else:
                cmd.append(value_str)
    
    return cmd


def run_single_experiment(cmd, timeout=300):
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
        
        # Parse output to extract metrics
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


def main():
    parser = argparse.ArgumentParser(description='Reproduce benchmark results')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='reproducibility_results', help='Output directory for results')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout per experiment in seconds')
    parser.add_argument('--results-dir', type=str, default='all_benchmark_results', help='Directory containing benchmark results')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results, skip already processed configs')
    parser.add_argument('--detector', type=str, default=None, help='Only process specific detector')
    parser.add_argument('--dataset', type=str, default=None, help='Only process specific dataset')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load existing results if resuming
    existing_results = {}
    if args.resume:
        combined_file = output_dir / "all_reproducibility_results.csv"
        if combined_file.exists():
            try:
                existing_df = pd.read_csv(combined_file)
                # Create a key for each existing result: detector_dataset_index
                for _, row in existing_df.iterrows():
                    key = f"{row['detector']}_{row['dataset']}_{row['index']}"
                    existing_results[key] = row
                print(f"Loaded {len(existing_results)} existing results for resume")
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")
    
    # Find all CSV files (resolve path relative to script location)
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    if not results_dir.is_absolute():
        results_dir = results_dir.resolve()
    csv_files = list(results_dir.glob('*/*/*.csv'))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    all_results = []
    
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
        
        filename = csv_file.name
        print(f"\nProcessing {csv_file}")
        
        # Determine mode and flags from filename
        mode, flags = parse_mode_from_filename(filename)
        
        # Extract configurations
        configs = extract_configurations(csv_file)
        print(f"  Found {len(configs)} configurations for {detector} on {dataset}")
        
        if not configs:
            continue
        
        # Prepare results for this experiment
        experiment_results = []
        
        # Track statistics for this experiment
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Filter out already processed configurations if resuming
        configs_to_process = []
        for i, config in enumerate(configs):
            key = f"{detector}_{dataset}_{i}"
            if args.resume and key in existing_results:
                # Add existing result to experiment results
                existing_row = existing_results[key]
                experiment_results.append(existing_row)
                all_results.append(existing_row)
                skipped_count += 1
            else:
                configs_to_process.append((i, config))
        
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} already processed configurations")
        
        if not configs_to_process:
            print(f"  All configurations already processed, skipping")
            continue
        
        # Run configurations in parallel using separate processes
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            
            for i, config in configs_to_process:
                cmd = build_command(detector, dataset, mode, flags, config)
                future = executor.submit(run_single_experiment, cmd, args.timeout)
                futures[future] = (i, copy.deepcopy(config), cmd)
            
            for future in as_completed(futures):
                idx, config, cmd = futures[future]
                try:
                    success, metrics, error = future.result()
                    
                    # Extract original metrics
                    original_accuracy = config.get('_original_accuracy', None)
                    original_runtime = config.get('_original_runtime', None)
                    original_reqlabels = config.get('_original_reqlabels', None)
                    original_mtr = config.get('_original_mtr', None)
                    
                    # Determine success based on accuracy
                    reproduced_accuracy = metrics.get('accuracy', None) if metrics else None
                    reproduced_runtime = metrics.get('runtime', None) if metrics else None
                    reproduced_reqlabels = metrics.get('reqlabels', None) if metrics else None
                    reproduced_mtr = metrics.get('mtr', None) if metrics else None
                    
                    # Success criteria: based on accuracy only (within 1% tolerance)
                    accuracy_success = False
                    if original_accuracy is not None and reproduced_accuracy is not None:
                        accuracy_success = abs(original_accuracy - reproduced_accuracy) < 0.01
                    
                    # For req labels experiments, also check reqlabels
                    reqlabels_success = False
                    if original_reqlabels is not None and reproduced_reqlabels is not None:
                        reqlabels_success = abs(original_reqlabels - reproduced_reqlabels) < 0.01
                    
                    # Calculate runtime ratio
                    runtime_ratio = None
                    if original_runtime is not None and reproduced_runtime is not None and original_runtime > 0:
                        runtime_ratio = reproduced_runtime / original_runtime
                    
                    result_row = {
                        'index': idx,
                        'detector': detector,
                        'dataset': dataset,
                        'mode': mode,
                        'run_success': success,
                        'accuracy_success': accuracy_success,
                        'reqlabels_success': reqlabels_success,
                        'original_accuracy': original_accuracy,
                        'reproduced_accuracy': reproduced_accuracy,
                        'original_runtime': original_runtime,
                        'reproduced_runtime': reproduced_runtime,
                        'runtime_ratio': runtime_ratio,
                        'original_reqlabels': original_reqlabels,
                        'reproduced_reqlabels': reproduced_reqlabels,
                        'original_mtr': original_mtr,
                        'reproduced_mtr': reproduced_mtr,
                        'error': error if error else '',
                        **config
                    }
                    
                    experiment_results.append(result_row)
                    all_results.append(result_row)
                    
                    # Update statistics
                    if accuracy_success:
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    total_processed = successful_count + failed_count
                    success_rate = (successful_count / total_processed * 100) if total_processed > 0 else 0
                    
                    status = "✓" if accuracy_success else "✗"
                    print(f"  [{total_processed}/{len(configs)}] {status} Success: {successful_count} ({success_rate:.1f}%) Failed: {failed_count}{' ' * 20}", end='\r')
                    
                except Exception as e:
                    print(f"  Error processing config {idx}: {e}")
        
        print(f"\n  Completed {len(experiment_results)} configurations")
        
        # Save results for this experiment
        if experiment_results:
            exp_df = pd.DataFrame(experiment_results)
            exp_filename = f"{detector}_{dataset}_reproducibility.csv"
            exp_df.to_csv(output_dir / exp_filename, index=False)
            print(f"  Saved results to {exp_filename}")
    
    # Generate summary report
    if all_results:
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        df = pd.DataFrame(all_results)
        
        # Overall statistics
        total = len(df)
        accuracy_successful = len(df[df['accuracy_success'] == True])
        accuracy_failed = total - accuracy_successful
        accuracy_success_rate = (accuracy_successful / total * 100) if total > 0 else 0
        
        # For req labels experiments
        reqlabels_configs = df[df['original_reqlabels'].notna()]
        if len(reqlabels_configs) > 0:
            reqlabels_successful = len(reqlabels_configs[reqlabels_configs['reqlabels_success'] == True])
            reqlabels_failed = len(reqlabels_configs) - reqlabels_successful
            reqlabels_success_rate = (reqlabels_successful / len(reqlabels_configs) * 100)
        else:
            reqlabels_successful = 0
            reqlabels_failed = 0
            reqlabels_success_rate = 0
        
        print(f"\nTotal configurations: {total}")
        print(f"Accuracy reproduced: {accuracy_successful} ({accuracy_success_rate:.1f}%)")
        print(f"Accuracy failed: {accuracy_failed} ({100-accuracy_success_rate:.1f}%)")
        if len(reqlabels_configs) > 0:
            print(f"ReqLabels reproduced: {reqlabels_successful} ({reqlabels_success_rate:.1f}%)")
            print(f"ReqLabels failed: {reqlabels_failed} ({100-reqlabels_success_rate:.1f}%)")
        
        # Statistics by detector
        print("\n" + "-"*60)
        print("BY DETECTOR (Accuracy)")
        print("-"*60)
        by_detector = df.groupby('detector').agg({
            'accuracy_success': ['count', 'sum']
        })
        by_detector.columns = ['total', 'successful']
        by_detector['failed'] = by_detector['total'] - by_detector['successful']
        by_detector['success_rate'] = (by_detector['successful'] / by_detector['total'] * 100).round(1)
        print(by_detector.to_string())
        
        # Statistics by dataset
        print("\n" + "-"*60)
        print("BY DATASET (Accuracy)")
        print("-"*60)
        by_dataset = df.groupby('dataset').agg({
            'accuracy_success': ['count', 'sum']
        })
        by_dataset.columns = ['total', 'successful']
        by_dataset['failed'] = by_dataset['total'] - by_dataset['successful']
        by_dataset['success_rate'] = (by_dataset['successful'] / by_dataset['total'] * 100).round(1)
        print(by_dataset.to_string())
        
        # Save combined results
        combined_filename = "all_reproducibility_results.csv"
        df.to_csv(output_dir / combined_filename, index=False)
        print(f"\nCombined results saved to {combined_filename}")
        
        # Save summary report
        report_filename = "reproducibility_report.txt"
        with open(output_dir / report_filename, 'w') as f:
            f.write("REPRODUCIBILITY REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total configurations: {total}\n")
            f.write(f"Accuracy reproduced: {accuracy_successful} ({accuracy_success_rate:.1f}%)\n")
            f.write(f"Accuracy failed: {accuracy_failed} ({100-accuracy_success_rate:.1f}%)\n")
            if len(reqlabels_configs) > 0:
                f.write(f"ReqLabels reproduced: {reqlabels_successful} ({reqlabels_success_rate:.1f}%)\n")
                f.write(f"ReqLabels failed: {reqlabels_failed} ({100-reqlabels_success_rate:.1f}%)\n\n")
            else:
                f.write("\n")
            
            f.write("BY DETECTOR (Accuracy)\n")
            f.write("-"*60 + "\n")
            f.write(by_detector.to_string() + "\n\n")
            
            f.write("BY DATASET (Accuracy)\n")
            f.write("-"*60 + "\n")
            f.write(by_dataset.to_string() + "\n")
        
        print(f"Summary report saved to {report_filename}")
        
        # List failed configurations (by accuracy)
        if accuracy_failed > 0:
            failed_df = df[df['accuracy_success'] == False]
            failed_filename = "failed_configurations.csv"
            failed_df.to_csv(output_dir / failed_filename, index=False)
            print(f"Failed configurations saved to {failed_filename}")


if __name__ == '__main__':
    main()
