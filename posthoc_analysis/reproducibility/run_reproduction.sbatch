#!/bin/bash

#SBATCH --job-name=DDBenchmark_reproduce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --cpu-freq=2000000
#SBATCH --output=benchmark_reproduce_%x_%j.out

# Get detector and dataset from command-line arguments
DETECTOR=$1
DATASET=$2

cd /data/horse/ws/s4122485-ddbenchmark/benchmark-unsupervised-concept-drift-detection
module load release/24.04 GCCcore/10.3.0 Python
source venv/bin/activate
cd reproducibility

python reproduce_benchmark.py \
    --results-dir /data/horse/ws/s4122485-ddbenchmark/benchmark-unsupervised-concept-drift-detection/all_benchmark_results \
    --max-workers 32 \
    --detector "$DETECTOR" \
    --dataset "$DATASET" \
    #--retry-failed \
    #--retry-timeout 14400
