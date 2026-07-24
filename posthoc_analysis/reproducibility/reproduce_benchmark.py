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
import ast
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
        
        # Filter out rows that are not completed (case-insensitive)
        if 'Status' in df.columns:
            df = df[df['Status'].str.upper() == 'COMPLETED']
        
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
            cmd.append(_format_param_value(value))
    
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
    parser.add_argument('--results-dir', type=str, default='../results/all_benchmark_results', help='Directory containing benchmark results')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results, skip already processed configs')
    parser.add_argument('--retry-failed', action='store_true', help='Only re-run failed configurations from existing results')
    parser.add_argument('--retry-timeout', type=int, default=None, help='Override timeout for retry runs (default: use --timeout)')
    parser.add_argument('--detector', type=str, default=None, help='Only process specific detector')
    parser.add_argument('--dataset', type=str, default=None, help='Only process specific dataset')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Don't load all results at startup - load per detector/dataset for efficiency
    # This avoids O(n*m) complexity when matching configurations
    if args.resume or args.retry_failed:
        print("Will load existing results per detector/dataset for efficiency")
        if args.retry_failed:
            print("Will retry only failed configurations")
            
            # Pre-scan to count total failed configurations and new configurations
            reproducibility_files = list(output_dir.glob("*_reproducibility.csv"))
            total_successful = 0
            total_failed = 0
            
            # Build a set of existing reproducibility configurations
            # Key: detector_dataset_metric
            existing_repro_configs = set()
            if reproducibility_files:
                for csv_file in reproducibility_files:
                    try:
                        df = pd.read_csv(csv_file)
                        for _, row in df.iterrows():
                            acc_success = str(row.get('accuracy_success', False)).lower() in ['true', '1', 'yes']
                            if acc_success:
                                total_successful += 1
                            else:
                                total_failed += 1
                            
                            # Extract metric suffix from filename
                            metric_suffix = csv_file.name.replace("_reproducibility.csv", "")
                            key = f"{row['detector']}_{row['dataset']}_{metric_suffix}"
                            existing_repro_configs.add(key)
                    except Exception as e:
                        pass  # Skip files that can't be read
                
                print(f"Total existing results: {total_successful + total_failed}")
                print(f"  Successful: {total_successful}")
                print(f"  Failed: {total_failed} (will retry)")
            
            # Count new configurations from benchmark results
            script_dir = Path(__file__).parent
            results_dir = script_dir / args.results_dir
            if not results_dir.is_absolute():
                results_dir = results_dir.resolve()
            benchmark_files = list(results_dir.glob('*/*/*.csv'))
            
            total_benchmark_configs = 0
            total_new_configs = 0
            
            for csv_file in benchmark_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'Status' in df.columns:
                        df = df[df['Status'].str.upper() == 'COMPLETED']
                    
                    # Extract detector, dataset, metric from filename
                    filename = csv_file.name
                    parts = filename.replace('.csv', '').split('_')
                    if len(parts) >= 3:
                        detector = parts[0]
                        dataset = parts[1]
                        metric_suffix = '_'.join(parts[2:])
                        
                        total_benchmark_configs += len(df)
                        key = f"{detector}_{dataset}_{metric_suffix}"
                        
                        if key not in existing_repro_configs:
                            total_new_configs += len(df)
                except Exception as e:
                    pass
            
            print(f"Total benchmark configurations: {total_benchmark_configs}")
            print(f"  New configurations to test: {total_new_configs}")
            print(f"  Total to process: {total_failed + total_new_configs}")
    
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
        
        # Extract metric suffix from filename for unique reproducibility file naming
        # e.g., "BNDM_Electricity_ACC_RT.csv" -> "ACC_RT"
        metric_suffix = filename.replace(f"{detector}_{dataset}_", "").replace(".csv", "")

        run_dataset = dataset

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
        
        # Filter out already processed configurations if resuming or retrying failed
        configs_to_process = []
        existing_count = 0
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Load existing results for this specific detector/dataset only
        existing_results_by_config = {}
        if args.resume or args.retry_failed:
            reproducibility_file = output_dir / f"{detector}_{dataset}_{metric_suffix}_reproducibility.csv"
            if reproducibility_file.exists():
                try:
                    df = pd.read_csv(reproducibility_file)
                    for _, row in df.iterrows():
                        # Extract configuration parameters (exclude result columns)
                        result_columns = {'detector', 'dataset', 'mode', 'run_success', 
                                        'accuracy_success', 'reqlabels_success', 'error', 'index',
                                        'original_accuracy', 'reproduced_accuracy',
                                        'original_runtime', 'reproduced_runtime',
                                        'runtime_ratio', 'original_reqlabels', 'reproduced_reqlabels',
                                        'original_mtr', 'reproduced_mtr'}
                        config_params = {k: str(v) for k, v in row.items() 
                                       if not k.startswith('_') and k not in result_columns}
                        existing_results_by_config[frozenset(config_params.items())] = row
                    print(f"  Loaded {len(existing_results_by_config)} existing results for {detector}/{dataset}/{metric_suffix}")
                except Exception as e:
                    print(f"  Warning: Could not load existing results for {detector}/{dataset}/{metric_suffix}: {e}")
        
        for i, config in enumerate(configs):
            # Extract configuration parameters (exclude internal metrics)
            config_params = {k: str(v) for k, v in config.items() if not k.startswith('_')}
            
            # Try to find a match by checking if all common parameters match
            matched_row = None
            for param_signature, existing_row in existing_results_by_config.items():
                # Get intersection of parameters
                common_params = set(config_params.keys()) & set(dict(param_signature).keys())
                
                # Check if all common parameters match
                all_match = True
                for param in common_params:
                    if config_params[param] != dict(param_signature)[param]:
                        all_match = False
                        break
                
                if all_match:
                    matched_row = existing_row
                    break
            
            if matched_row is not None:
                existing_row = matched_row
                existing_count += 1
                
                if args.retry_failed:
                    # In retry mode, only skip if the existing result was successful
                    # Handle both boolean and string values
                    acc_success = str(existing_row.get('accuracy_success', False)).lower() in ['true', '1', 'yes']
                    if acc_success:
                        # Keep successful result
                        experiment_results.append(existing_row)
                        all_results.append(existing_row)
                        skipped_count += 1
                        successful_count += 1
                    else:
                        # Re-run failed configurations
                        configs_to_process.append((i, config))
                        failed_count += 1
                elif args.resume:
                    # In resume mode, skip all already processed configurations
                    experiment_results.append(existing_row)
                    all_results.append(existing_row)
                    skipped_count += 1
                else:
                    # Not resuming, process all
                    configs_to_process.append((i, config))
            else:
                # No existing result, process it
                configs_to_process.append((i, config))
        
        if args.retry_failed and existing_count > 0:
            print(f"  Existing results: {existing_count} (Successful: {successful_count}, Failed: {failed_count})")
            print(f"  New configurations to test: {len(configs) - existing_count}")
        elif args.retry_failed and len(existing_results_by_config) > 0:
            print(f"  WARNING: Loaded {len(existing_results_by_config)} existing results but matched {existing_count}")
            print(f"  This suggests parameter matching is not working correctly")
            print(f"  Benchmark config params sample: {list(configs[0].keys())[:5]}")
            if existing_results_by_config:
                print(f"  Existing config params sample: {list(dict(list(existing_results_by_config.keys())[0]).keys())[:5]}")
        
        if skipped_count > 0:
            if args.retry_failed:
                print(f"  Skipped {skipped_count} successful configurations (will retry failed)")
            else:
                print(f"  Skipped {skipped_count} already processed configurations")
        
        if not configs_to_process:
            print(f"  No configurations to process, skipping")
            continue
        
        # Run configurations in parallel using separate processes
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            
            # Use retry-timeout if specified, otherwise use default timeout
            timeout = args.retry_timeout if args.retry_failed and args.retry_timeout else args.timeout
            
            for i, config in configs_to_process:
                cmd = build_command(detector, run_dataset, mode, flags, config)
                future = executor.submit(run_single_experiment, cmd, timeout)
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
            exp_filename = f"{detector}_{dataset}_{metric_suffix}_reproducibility.csv"
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
