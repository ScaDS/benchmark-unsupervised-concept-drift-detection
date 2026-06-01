#!/bin/bash

################################################################################
# SLURM Batch Script for the Runtime Stability Study
#
# This script is submitted once per drift detector. Inside the job it iterates
# over all eligible datasets and runs ``runtime_stability_study.py`` to
# re-execute 10 randomly sampled configurations 20 times each, recording the
# mean and standard deviation of their runtime.
#
# Usage:
#   sbatch run_runtime_stability_study.sbatch <DETECTOR> [N_CONFIGS] [N_RUNS]
#
# Arguments:
#   DETECTOR    - Drift detector class name (e.g. DDAL, BNDM, ...)
#   N_CONFIGS   - Optional: number of random configurations (default: 10)
#   N_RUNS      - Optional: repetitions per configuration  (default: 20)
#
# Output:
#   One CSV per (detector, dataset) pair under
#   ``runtime_stability_results/<DETECTOR>_<DATASET>.csv``.
#
################################################################################

#SBATCH --job-name=DDBenchmark_runtime_stability
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --cpu-freq=2000000
#SBATCH --output=runtime_stability_%x_%j.out

set -euo pipefail

################################################################################
# Validate arguments
################################################################################

if [ "$#" -lt 1 ]; then
    echo "Error: Detector argument is required."
    echo "Usage: sbatch $0 <DETECTOR> [N_CONFIGS] [N_RUNS]"
    exit 1
fi

DETECTOR="$1"
N_CONFIGS="${2:-10}"
N_RUNS="${3:-20}"

echo "=== Runtime Stability Study (SLURM) ==="
echo "Detector:   ${DETECTOR}"
echo "N_CONFIGS:  ${N_CONFIGS}"
echo "N_RUNS:     ${N_RUNS}"
echo "Job ID:     ${SLURM_JOB_ID:-<local>}"
echo "Host:       $(hostname)"
echo "========================================"
echo ""

################################################################################
# Environment setup
################################################################################

# Avoid OmniOpt-style early termination signals (mirrors other run_*.sbatch).
export DONT_SEND_SIGNAL_BEFORE_END=1

# Strip stale SLURM_* variables from the environment so child Python processes
# do not get confused by their parent job (mirrors other run_*.sbatch).
for i in $(env | grep -i "^slurm" | sed -e 's#=.*##'); do
    unset "$i"
done

source benchmark_config.sh

PYTHON_BIN="python"
if command -v squeue &> /dev/null; then
    # On the cluster, load Python and activate the project venv.
    module load GCCcore/10.3.0 Python
    source "${VENV_PATH}/bin/activate"
    PYTHON_BIN="$(command -v python)"
fi

################################################################################
# Datasets
################################################################################
# All datasets present under all_benchmark_results/<DETECTOR>/, minus the two
# datasets that the study spec excludes (SineClusters, WaveformDrift2).

EXCLUDE=("SineClusters" "WaveformDrift2")

DETECTOR_DIR="all_benchmark_results/${DETECTOR}"
if [ ! -d "${DETECTOR_DIR}" ]; then
    echo "Error: Detector directory not found: ${DETECTOR_DIR}"
    exit 1
fi

datasets=()
while IFS= read -r ds; do
    skip=false
    for ex in "${EXCLUDE[@]}"; do
        if [ "$ds" = "$ex" ]; then
            skip=true
            break
        fi
    done
    if [ "$skip" = false ]; then
        datasets+=("$ds")
    fi
done < <(find "${DETECTOR_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

echo "Datasets (${#datasets[@]}): ${datasets[*]}"
echo ""

################################################################################
# Run the study for each dataset
################################################################################

for dataset in "${datasets[@]}"; do
    echo "=== ${DETECTOR} / ${dataset} ==="
    "$PYTHON_BIN" runtime_stability_study.py \
        --detector "${DETECTOR}" \
        --dataset "${dataset}" \
        --n-configs "${N_CONFIGS}" \
        --n-runs "${N_RUNS}" \
        --python-exe "$PYTHON_BIN" \
        || echo "WARNING: study failed for ${DETECTOR}/${dataset}"
    echo ""
done

echo "=== Runtime stability study finished for ${DETECTOR} ==="
