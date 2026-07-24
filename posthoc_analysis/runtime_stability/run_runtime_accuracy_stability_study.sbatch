#!/bin/bash

################################################################################
# SLURM Batch Script for the Runtime + Accuracy Stability Study
#
# This script is submitted once per detector/dataset combination and runs
# ``runtime_accuracy_stability_study.py`` to re-execute randomly sampled
# configurations multiple times, recording the mean and standard deviation
# of both runtime and accuracy.
#
# Usage:
#   sbatch run_runtime_accuracy_stability_study.sbatch <DETECTOR> <DATASET> [N_CONFIGS] [N_RUNS]
#
# Arguments:
#   DETECTOR    - Drift detector class name (e.g. DDAL, BNDM, ...)
#   DATASET     - Dataset name (e.g. Electricity, NOAAWeather, ...)
#   N_CONFIGS   - Optional: number of random configurations (default: 10)
#   N_RUNS      - Optional: repetitions per configuration  (default: 20)
#
# Output:
#   One CSV under ``runtime_accuracy_stability_results/<DETECTOR>_<DATASET>.csv``.
#
################################################################################

#SBATCH --job-name=DDBenchmark_runtime_accuracy_stability
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --cpu-freq=2000000
#SBATCH --output=runtime_accuracy_stability_%x_%j.out

set -euo pipefail

################################################################################
# Validate arguments
################################################################################

if [ "$#" -lt 2 ]; then
    echo "Error: Detector and dataset arguments are required."
    echo "Usage: sbatch $0 <DETECTOR> <DATASET> [N_CONFIGS] [N_RUNS]"
    exit 1
fi

DETECTOR="$1"
DATASET="$2"
N_CONFIGS="${3:-10}"
N_RUNS="${4:-20}"

echo "=== Runtime + Accuracy Stability Study (SLURM) ==="
echo "Detector:   ${DETECTOR}"
echo "Dataset:    ${DATASET}"
echo "N_CONFIGS:  ${N_CONFIGS}"
echo "N_RUNS:     ${N_RUNS}"
echo "Job ID:     ${SLURM_JOB_ID:-<local>}"
echo "Host:       $(hostname)"
echo "=================================================="
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
# Run the study
################################################################################

echo "=== ${DETECTOR} / ${DATASET} ==="
"$PYTHON_BIN" posthoc_analysis/runtime_stability/runtime_accuracy_stability_study.py \
    --detector "${DETECTOR}" \
    --dataset "${DATASET}" \
    --n-configs "${N_CONFIGS}" \
    --n-runs "${N_RUNS}" \
    --python-exe "$PYTHON_BIN" \
    || echo "WARNING: study failed for ${DETECTOR}/${DATASET}"
echo ""

echo "=== Runtime + accuracy stability study finished for ${DETECTOR} / ${DATASET} ==="
