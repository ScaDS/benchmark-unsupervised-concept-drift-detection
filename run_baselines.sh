#!/bin/bash

################################################################################
# Baseline Drift Detector Evaluation Script
#
# This script computes baseline performance metrics for dummy drift detectors
# across multiple data streams. Baselines include:
#   - DummyDDBL1: Never detects drift (no retraining)
#   - DummyDDBL2: Periodic retraining with various configurations
#
# The script runs multiple iterations per stream to ensure statistical
# reliability of the baseline measurements.
#
# Usage:
#   ./run_baselines.sh [NUM_ITERATIONS]
#
# Arguments:
#   NUM_ITERATIONS - Optional: Number of iterations per stream (default: 5)
#
# Output:
#   Results are appended to Baselines_<STREAM>.out files
#
# Configuration:
#   - Edit the 'streams' array to select which datasets to evaluate
#   - Adjust iteration range as needed for statistical significance
#
################################################################################

################################################################################
# Configuration
################################################################################

# Number of iterations per stream (default: 5, range: 6-10 for backward compatibility)
NUM_ITERATIONS="${1:-5}"
START_ITERATION=1
END_ITERATION=$((START_ITERATION + NUM_ITERATIONS - 1))

# Data streams to evaluate
# Uncomment the streams you want to test
streams=(
    #"Electricity"
    #"NOAAWeather"
    #"OutdoorObjects"
    #"PokerHand"
    #"RialtoBridgeTimelapse"
    #"ForestCovertype"
    #"GasSensor"
    #"Ozone"
    "SensorStream"
)

################################################################################
# Baseline Evaluation
################################################################################

echo "=== Baseline Drift Detector Evaluation ==="
echo "Streams:    ${#streams[@]} configured"
echo "Iterations: $START_ITERATION to $END_ITERATION ($NUM_ITERATIONS total)"
echo "==========================================="
echo ""

# Check if SLURM is available
if command -v squeue &> /dev/null || [ -n "$SLURM_JOB_ID" ]; then
    SLURM_AVAILABLE=true
    echo "SLURM detected - using module load and venv"
else
    SLURM_AVAILABLE=false
    echo "Local machine detected - using python directly"
fi
echo ""

# Iterate through each stream
for stream in "${streams[@]}"; do
    echo "=== Evaluating baselines on: $stream ==="
    
    output_file="Baselines_${stream}.out"
    
    # Run multiple iterations for statistical reliability
    for ((i=START_ITERATION; i<=END_ITERATION; i++)); do
        echo "  Iteration $i/$END_ITERATION..."
        
        # Log iteration number
        echo "Iteration $i" >> "$output_file"
        
        # Run baseline computation and append results
        if [ "$SLURM_AVAILABLE" = true ]; then
            # SLURM environment - use module load and venv
            module load GCCcore/10.3.0 Python && \
            source /data/horse/ws/s4122485-compPerfDD/benchmark/venv/bin/activate && \
            python compute_baselines.py "$stream" >> "$output_file"
        else
            # Local machine - use python directly
            python compute_baselines.py "$stream" >> "$output_file"
        fi
        
        # Check exit status
        if [ $? -ne 0 ]; then
            echo "  WARNING: Iteration $i failed for $stream" | tee -a "$output_file"
        fi
    done
    
    echo "  Results saved to: $output_file"
    echo ""
done

echo "=== Baseline evaluation completed ==="
echo "Output files: Baselines_*.out"
