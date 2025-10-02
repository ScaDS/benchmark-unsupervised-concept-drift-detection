#!/bin/bash
################################################################################
# SLURM Batch Script for Classifier Training
#
# This script submits classifier training jobs to the SLURM scheduler.
# It uses the unified run_train_classifiers.sh script which supports both
# HoeffdingTreeClassifier (with hyperparameter optimization) and GaussianNB
# (direct training).
#
# Usage:
#   sbatch run_train_classifiers.sbatch [CLASSIFIER]
#
# Arguments:
#   CLASSIFIER - Optional: Specific classifier to train
#                Options: "HoeffdingTreeClassifier", "GaussianNB", "all"
#                Default: "all" (trains both classifiers)
#
# Examples:
#   sbatch run_train_classifiers.sbatch                          # Train all
#   sbatch run_train_classifiers.sbatch HoeffdingTreeClassifier  # Train HT only
#   sbatch run_train_classifiers.sbatch GaussianNB               # Train NB only
#
################################################################################

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --job-name=DDBenchmark_trainClf
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --nodes=1
##SBATCH --cpu-freq=2000000

################################################################################
# Parse Arguments
################################################################################

# Default: train all classifiers
CLASSIFIER="${1:-all}"

echo "=== SLURM Classifier Training Job ==="
echo "Classifier:   $CLASSIFIER"
echo "======================================"
echo ""

################################################################################
# Execute Training
################################################################################

# Temporarily modify the run_train_classifiers.sh to use specified classifier
if [ "$CLASSIFIER" = "HoeffdingTreeClassifier" ]; then
    echo "Training HoeffdingTreeClassifier only..."
    # Create temporary script with only HT
    sed 's/classifiers=(/classifiers=(\n    #"GaussianNB"/' run_train_classifiers.sh > run_train_classifiers_temp.sh
    sed -i 's/"GaussianNB"/#"GaussianNB"/' run_train_classifiers_temp.sh
    bash run_train_classifiers_temp.sh
    rm run_train_classifiers_temp.sh
elif [ "$CLASSIFIER" = "GaussianNB" ]; then
    echo "Training GaussianNB only..."
    # Create temporary script with only NB
    sed 's/"HoeffdingTreeClassifier"/#"HoeffdingTreeClassifier"/' run_train_classifiers.sh > run_train_classifiers_temp.sh
    bash run_train_classifiers_temp.sh
    rm run_train_classifiers_temp.sh
else
    echo "Training all classifiers..."
    bash run_train_classifiers.sh
fi

echo ""
echo "=== Training job completed ==="
