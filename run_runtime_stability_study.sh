#!/bin/bash

################################################################################
# Runtime Stability Study Launcher
#
# Submits one SLURM job per drift detector. Each job iterates over all
# eligible datasets (all subfolders of ``all_benchmark_results/<DD>/`` except
# the two that are excluded by the study spec: SineClusters, WaveformDrift2)
# and re-executes 10 randomly sampled configurations 20 times to measure
# runtime stability (mean + std).
#
# Usage:
#   ./run_runtime_stability_study.sh [N_CONFIGS] [N_RUNS]
#
# Arguments:
#   N_CONFIGS   - Optional: number of random configurations per detector/
#                 dataset combination (default: 10)
#   N_RUNS      - Optional: repetitions per configuration (default: 20)
#
# Output:
#   - One CSV per (detector, dataset) pair under
#     ``runtime_stability_results/<DETECTOR>_<DATASET>.csv``.
#   - One SLURM stdout file per detector
#     (``runtime_stability_*_<JOBID>.out``).
#
# After all SLURM jobs finish, the per-pair CSVs can be concatenated into a
# single overview file, e.g.:
#   awk 'FNR==1 && NR!=1 { next } 1' \
#       runtime_stability_results/*.csv > runtime_stability_summary.csv
#
################################################################################

set -euo pipefail

################################################################################
# Parse arguments
################################################################################

N_CONFIGS="${1:-10}"
N_RUNS="${2:-20}"

echo "=== Runtime Stability Study Launcher ==="
echo "N_CONFIGS: ${N_CONFIGS}"
echo "N_RUNS:    ${N_RUNS}"
echo "========================================"
echo ""

################################################################################
# Load benchmark configuration
################################################################################

source benchmark_config.sh

################################################################################
# Detect detectors automatically from all_benchmark_results
################################################################################

if [ ! -d "all_benchmark_results" ]; then
    echo "Error: 'all_benchmark_results' directory not found."
    echo "Run this script from the project root."
    exit 1
fi

detectors=()
while IFS= read -r dd; do
    detectors+=("$dd")
done < <(find all_benchmark_results -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

echo "Detectors (${#detectors[@]}): ${detectors[*]}"
echo ""

################################################################################
# Submit one SLURM job per detector
################################################################################

if command -v sbatch &> /dev/null; then
    SLURM_AVAILABLE=true
    echo "SLURM detected -> submitting one job per detector via sbatch."
else
    SLURM_AVAILABLE=false
    echo "SLURM not detected -> running detectors sequentially in foreground."
fi
echo ""

mkdir -p runtime_stability_results

for dd in "${detectors[@]}"; do
    echo "=== Detector: ${dd} ==="
    if [ "$SLURM_AVAILABLE" = true ]; then
        sbatch \
            --job-name="DDStability_${dd}" \
            run_runtime_stability_study.sbatch \
            "${dd}" "${N_CONFIGS}" "${N_RUNS}"
    else
        bash run_runtime_stability_study.sbatch \
            "${dd}" "${N_CONFIGS}" "${N_RUNS}"
    fi
    # Small pause between submissions to avoid overwhelming the scheduler.
    sleep 1
done

echo ""
echo "=== All runtime stability jobs submitted ==="
echo "Per-pair results will appear under runtime_stability_results/."
echo "After all jobs finish, concatenate them with:"
echo "  awk 'FNR==1 && NR!=1 { next } 1' \\"
echo "      runtime_stability_results/*.csv > runtime_stability_summary.csv"
