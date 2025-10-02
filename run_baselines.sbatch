#!/bin/bash
################################################################################
# SLURM Batch Script for Baseline Drift Detector Evaluation
#
# This script submits baseline evaluation jobs to the SLURM scheduler.
# It runs dummy drift detectors (DummyDDBL1 and DummyDDBL2) across multiple
# data streams to establish baseline performance metrics.
#
# Usage:
#   sbatch run_baselines.sbatch [NUM_ITERATIONS]
#
# Arguments:
#   NUM_ITERATIONS - Optional: Number of iterations per stream (default: 5)
#
# Examples:
#   sbatch run_baselines.sbatch      # Run with default 5 iterations
#   sbatch run_baselines.sbatch 10   # Run with 10 iterations
#
# Output:
#   Results are saved to Baselines_<STREAM>.out files in the working directory
#
################################################################################

#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --job-name=DDBenchmark_baselines
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --nodes=1
#SBATCH --cpu-freq=2000000

################################################################################
# Parse Arguments
################################################################################

# Number of iterations (default: 5)
NUM_ITERATIONS="${1:-5}"

echo "=== SLURM Baseline Evaluation Job ==="
echo "Iterations: $NUM_ITERATIONS"
echo "======================================"
echo ""

################################################################################
# Execute Baseline Evaluation
################################################################################

bash run_baselines.sh "$NUM_ITERATIONS"

echo ""
echo "=== Baseline evaluation job completed ==="
