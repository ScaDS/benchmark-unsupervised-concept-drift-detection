#!/usr/bin/env python3
"""
SLURM Job Submission Script for Runtime + Accuracy Stability Study

This script generates and submits SLURM jobs for each detector/dataset combination
to run the runtime_accuracy_stability_study.py script.

Usage:
    python submit_runtime_accuracy_stability_jobs.py [--sbatch-file FILE] [--dry-run] [--max-jobs N]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def find_detector_dataset_combinations(results_dir):
    """
    Find all detector/dataset combinations in the results directory.

    Args:
        results_dir: Path to all_benchmark_results directory

    Returns:
        list: List of (detector, dataset) tuples
    """
    results_path = Path(results_dir)
    combinations = []

    if not results_path.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return combinations

    excluded_datasets = {"SineClusters", "WaveformDrift2"}

    for detector_dir in results_path.iterdir():
        if not detector_dir.is_dir():
            continue

        detector = detector_dir.name

        for dataset_dir in detector_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset = dataset_dir.name

            if dataset in excluded_datasets:
                continue

            csv_files = list(dataset_dir.glob('*.csv'))
            if csv_files:
                combinations.append((detector, dataset))

    return sorted(combinations)


def generate_job_name(detector, dataset):
    """
    Generate a SLURM job name for a detector/dataset combination.

    Args:
        detector: Detector name
        dataset: Dataset name

    Returns:
        str: Job name
    """
    clean_detector = detector.replace('_', '-')[:20]
    clean_dataset = dataset.replace('_', '-')[:20]
    return f"rt-acc-stability-{clean_detector}-{clean_dataset}"


def submit_sbatch(sbatch_file, detector, dataset, n_configs, n_runs, dry_run=False):
    """
    Submit a SLURM job using the sbatch template file.

    Args:
        sbatch_file: Path to the sbatch template file
        detector: Detector name
        dataset: Dataset name
        n_configs: Number of random configurations to sample
        n_runs: Number of repetitions per configuration
        dry_run: If True, print command without executing

    Returns:
        tuple: (success, job_id or error_message)
    """
    job_name = generate_job_name(detector, dataset)

    cmd = ['sbatch', '--job-name', job_name, str(sbatch_file),
           detector, dataset, str(n_configs), str(n_runs)]

    if dry_run:
        print(f"Would submit: {' '.join(cmd)}")
        return True, job_name

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        if 'Submitted batch job' in output:
            job_id = output.split()[-1]
            return True, job_id
        else:
            return False, f"Unexpected output: {output}"

    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description='Submit SLURM jobs for runtime + accuracy stability study')
    parser.add_argument('--sbatch-file', type=str, default='run_runtime_accuracy_stability_study.sbatch',
                        help='Path to the sbatch template file')
    parser.add_argument('--results-dir', type=str, default='../../results/all_benchmark_results',
                        help='Directory containing benchmark results')
    parser.add_argument('--n-configs', type=int, default=10,
                        help='Number of random configurations per detector/dataset (default: 10)')
    parser.add_argument('--n-runs', type=int, default=20,
                        help='Repetitions per configuration (default: 20)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without submitting')
    parser.add_argument('--max-jobs', type=int, default=None,
                        help='Maximum number of jobs to submit (for testing)')
    parser.add_argument('--detector', type=str, default=None,
                        help='Only submit jobs for specific detector')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Only submit jobs for specific dataset')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    sbatch_file = script_dir / args.sbatch_file
    results_dir = script_dir / args.results_dir

    if not sbatch_file.exists():
        print(f"Error: Sbatch file {sbatch_file} does not exist")
        sys.exit(1)

    combinations = find_detector_dataset_combinations(results_dir)

    if not combinations:
        print(f"No detector/dataset combinations found in {results_dir}")
        sys.exit(1)

    if args.detector:
        combinations = [(d, ds) for d, ds in combinations if d == args.detector]
    if args.dataset:
        combinations = [(d, ds) for d, ds in combinations if ds == args.dataset]

    if args.max_jobs:
        combinations = combinations[:args.max_jobs]

    print(f"Found {len(combinations)} detector/dataset combinations")
    print(f"Using sbatch file: {sbatch_file}")
    print(f"N_CONFIGS: {args.n_configs}")
    print(f"N_RUNS: {args.n_runs}")

    if args.dry_run:
        print("DRY RUN MODE - No jobs will be submitted")

    submitted_jobs = []
    failed_jobs = []

    for detector, dataset in combinations:
        print(f"\nSubmitting job for {detector} on {dataset}...")

        success, result = submit_sbatch(
            sbatch_file, detector, dataset,
            args.n_configs, args.n_runs, args.dry_run
        )

        if success:
            job_id = result if args.dry_run else result
            submitted_jobs.append({
                'detector': detector,
                'dataset': dataset,
                'job_id': job_id
            })
            print(f"  ✓ Job submitted: {job_id}")
        else:
            failed_jobs.append({
                'detector': detector,
                'dataset': dataset,
                'error': result
            })
            print(f"  ✗ Failed: {result}")

    print("\n" + "=" * 60)
    print("SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"Total combinations: {len(combinations)}")
    print(f"Successfully submitted: {len(submitted_jobs)}")
    print(f"Failed: {len(failed_jobs)}")

    if failed_jobs:
        print("\nFailed jobs:")
        for job in failed_jobs:
            print(f"  {job['detector']}/{job['dataset']}: {job['error']}")

    if not args.dry_run and submitted_jobs:
        print(f"\nJob IDs:")
        for job in submitted_jobs:
            print(f"  {job['detector']}/{job['dataset']}: {job['job_id']}")

        job_list_file = script_dir / "submitted_stability_jobs.txt"
        with open(job_list_file, 'w') as f:
            for job in submitted_jobs:
                f.write(f"{job['job_id']} {job['detector']} {job['dataset']}\n")
        print(f"\nJob list saved to {job_list_file}")


if __name__ == '__main__':
    main()
