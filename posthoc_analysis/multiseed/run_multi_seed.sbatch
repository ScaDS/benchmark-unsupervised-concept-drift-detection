#!/bin/bash
#SBATCH --job-name=multiseed
#SBATCH --output=multiseed_%j.out
#SBATCH --error=multiseed_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=standard

# Get detector and dataset from command line arguments
DETECTOR=$1
DATASET=$2

echo "Running multi-seed analysis for ${DETECTOR} on ${DATASET}"

# Activate conda environment if needed
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

# Run the multi_seed_analysis.py script
python posthoc_analysis/multiseed/multi_seed_analysis.py \
    --detector ${DETECTOR} \
    --dataset ${DATASET} \
    --max-workers 8 \
    --timeout 7200 \
    --seeds 10 \
    --seed-start 42 \
    --output-dir multi_seed_results
