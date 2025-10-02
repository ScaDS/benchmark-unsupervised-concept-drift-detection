#!/bin/bash

################################################################################
# SLURM Batch Script for Drift Detector Hyperparameter Optimization
#
# This script submits hyperparameter optimization jobs to the SLURM cluster.
# It serves as a launcher for the optimization scripts, configuring the
# cluster resources and environment before executing the experiments.
#
# Usage:
#   sbatch run_experiment.sbatch
#
# The script calls optimization scripts with the following parameters:
#   Argument 1: ACCURACY     - Optimize for accuracy (1=true, 0=false)
#   Argument 2: RUNTIME      - Optimize for runtime (1=true, 0=false)
#   Argument 3: REQLABELS    - Optimize for requested labels (1=true, 0=false)
#   Argument 4: CONTINUE     - Continue from previous run (1=true, 0=false)
#   Argument 5: MTR          - Optimize for Mean Time Ratio (1=true, 0=false) [optional]
#   Argument 6: FEATURETEST  - Run feature testing mode (1=true, 0=false) [optional]
#
################################################################################

################################################################################
# SLURM Resource Configuration
################################################################################

#SBATCH --job-name=DDBenchmark_main    # Job name for identification
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=1              # CPU cores per task
#SBATCH --mem=48GB                     # Memory per node
#SBATCH --time=02:00:00                # Maximum runtime (HH:MM:SS)
#SBATCH --cpu-freq=2000000             # CPU frequency (2 GHz)

################################################################################
# Environment Setup
################################################################################

# Prevent OmniOpt from sending signals before job completion
export DONT_SEND_SIGNAL_BEFORE_END=1

# Clean up SLURM environment variables to avoid conflicts with nested jobs
for i in $(env | grep -i "^slurm" | sed -e 's#=.*##'); do 
    unset $i
done

################################################################################
# Run Optimization Experiments
################################################################################

# Active experiment: Optimize for ACCURACY and RUNTIME
bash run_stream_detector_optimization.sh 1 1 0 0

# Alternative experiment configurations (uncomment as needed):

# Optimize for ACCURACY, RUNTIME, and REQLABELS
#bash run_stream_detector_optimization.sh 1 1 1 0

# Continue previous ACCURACY + RUNTIME optimization
#bash run_stream_detector_optimization.sh 1 1 0 1

# Run MTR (Mean Time Ratio) experiment on synthetic streams
# Arguments: ACCURACY RUNTIME REQLABELS CONTINUE MTR FEATURETEST
# Note: MTR mode uses synthetic streams (SineClustersPre, WaveformPre) and main_mtr.py
#bash run_stream_detector_optimization.sh 0 1 0 0 1 0

# Continue previous MTR experiment
#bash run_stream_detector_optimization.sh 0 1 0 1 1 0

# Run Feature Testing experiment to evaluate each feature individually
# Arguments: ACCURACY RUNTIME REQLABELS CONTINUE MTR FEATURETEST
# Note: Feature testing mode iterates through all features, fixing feature_id for each run
#bash run_stream_detector_optimization.sh 1 1 0 0 0 1

# Continue previous Feature Testing experiment
#bash run_stream_detector_optimization.sh 1 1 0 1 0 1
