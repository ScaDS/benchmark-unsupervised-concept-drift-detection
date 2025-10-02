#!/bin/bash

################################################################################
# Classifier Training Script
#
# This script trains classifiers on streaming data using hyperparameter 
# optimization via OmniOpt. It supports both classifiers with hyperparameters
# (e.g., HoeffdingTreeClassifier) and parameter-free classifiers (e.g., GaussianNB).
#
# Usage:
#   ./run_train_classifiers.sh
#
# Features:
#   - Automated hyperparameter optimization for HoeffdingTreeClassifier
#   - Direct training for GaussianNB (no hyperparameters)
#   - Configurable training and validation sample sizes
#   - Parallel job execution via OmniOpt/SLURM
#   - Support for multiple data streams
#
# Configuration:
#   - Edit the 'streams' array to select which datasets to train on
#   - Edit the 'classifiers' array to select which classifiers to use
#   - Adjust n_training_samples and n_val_samples as needed
#
################################################################################

################################################################################
# Configuration
################################################################################

# Training and validation sample sizes
n_training_samples=1600
n_val_samples=400

# Classifiers to train
# Options: "HoeffdingTreeClassifier", "GaussianNB"
classifiers=(
    "HoeffdingTreeClassifier"
    "GaussianNB"
)

# Data streams to train on
# Uncomment the streams you want to use
streams=(
    #"Electricity"
    #"NOAAWeather"
    #"OutdoorObjects"
    #"PokerHand"
    #"RialtoBridgeTimelapse"
    #"ForestCovertype"
    #"GasSensor"
    #"Ozone"
    #"SensorStream"
    "SineClustersPre"
    "WaveformPre"
)

################################################################################
# Classifier Hyperparameter Definitions
################################################################################

# HoeffdingTreeClassifier hyperparameter search space
# Format: "param_name:type:dtype:values"
# Types: range (min max), choice (comma-separated), fixed (single value)
HoeffdingTreeClassifier=(
    "grace_period:range:int:20 400"
    "split_criterion:choice:str:str\(\'gini\'\),str\(\'info_gain\'\),str\(\'hellinger\'\)"
    "delta:range:float:0.00000001 0.01"
    "tau:range:float:0.001 0.3"
    "leaf_prediction:choice:str:str\(\'mc\'\),str\(\'nb\'\),str\(\'nba\'\)"
    "nb_threshold:range:int:0 20"
    "binary_split:choice:bool:True,False"
    "min_branch_fraction:range:float:0.001 0.1"
    "max_share_to_split:range:float:0.8 0.99"
    "merit_preprune:choice:bool:True,False"
)

# GaussianNB has no hyperparameters to optimize

################################################################################
# Training Execution
################################################################################

echo "=== Classifier Training Configuration ==="
echo "Training samples:   $n_training_samples"
echo "Validation samples: $n_val_samples"
echo "Classifiers:        ${classifiers[*]}"
echo "Streams:            ${#streams[@]} configured"
echo "=========================================="
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

# Iterate through each stream and classifier combination
for stream in "${streams[@]}"; do
    for clf in "${classifiers[@]}"; do
        echo "=== Training: $clf on $stream ==="
        
        # Check if classifier has hyperparameters to optimize
        if declare -p "$clf" 2>/dev/null | grep -q 'declare \-a'; then
            # Classifier with hyperparameters: Use OmniOpt for optimization
            echo "Mode: Hyperparameter optimization via OmniOpt"
            
            # Build parameter lists for Python and OmniOpt
            declare -n list=$clf
            python_args_list=()
            omniopt_args_list=()
            
            for arg in "${list[@]}"; do
                IFS=':' read -r name type dtype val rest <<< "$arg"
                
                # Add to Python arguments (will be substituted by OmniOpt)
                python_args_list+=("$name")
                python_args_list+=("%($name)")
                
                # Add to OmniOpt arguments based on parameter type
                if [[ "$type" == "choice" ]]; then
                    omniopt_args_list+=("--parameter $name $type $val")
                elif [[ "$type" == "fixed" ]]; then
                    omniopt_args_list+=("--parameter $name $type $rest")
                else
                    omniopt_args_list+=("--parameter $name $type $val $dtype")
                fi
            done
            
            # Build run command based on environment
            if [ "$SLURM_AVAILABLE" = true ]; then
                # SLURM environment - use module load and venv
                run_program=$(echo -n "module load GCCcore/10.3.0 Python && source /data/horse/ws/s4122485-compPerfDD/benchmark/venv/bin/activate && python train_classifiers.py $stream $n_training_samples $n_val_samples $clf ${python_args_list[*]}" | base64 -w 0)
            else
                # Local machine - use python directly
                run_program=$(echo -n "python train_classifiers.py $stream $n_training_samples $n_val_samples $clf ${python_args_list[*]}" | base64 -w 0)
            fi
            
            # Launch OmniOpt hyperparameter optimization
            ./OmniOpt/omniopt \
                --partition=romeo \
                --experiment_name="trainClf_${stream}_${clf}" \
                --mem_gb=32 \
                --time=120 \
                --worker_timeout=120 \
                --max_eval=200 \
                --num_parallel_jobs=20 \
                --gpus=0 \
                --num_random_steps=20 \
                --follow \
                --send_anonymized_usage_stats \
                --run_program=$run_program \
                --cpus_per_task=1 \
                --model=BOTORCH_MODULAR \
                --run_mode=local \
                "${omniopt_args_list[@]}"
        else
            # Classifier without hyperparameters: Direct training
            echo "Mode: Direct training (no hyperparameters)"
            
            if [ "$SLURM_AVAILABLE" = true ]; then
                # SLURM environment - use module load and venv
                module load GCCcore/10.3.0 Python && \
                source /data/horse/ws/s4122485-compPerfDD/benchmark/venv/bin/activate && \
                python train_classifiers.py "$stream" "$n_training_samples" "$n_val_samples" "$clf"
            else
                # Local machine - use python directly
                python train_classifiers.py "$stream" "$n_training_samples" "$n_val_samples" "$clf"
            fi
        fi
        
        echo ""
    done
done

echo "=== Classifier training completed ==="
