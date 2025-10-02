#!/bin/bash

################################################################################
# Hyperparameter Optimization for Stream-Specific Drift Detector Combinations
#
# This script runs hyperparameter optimization experiments for specific
# combinations of data streams and drift detectors using OmniOpt. Unlike the
# general experiment script, this allows fine-grained control over which
# detectors are tested on which streams, and supports constraint handling.
#
# Usage:
#   ./run_stream_detector_optimization.sh <Accuracy> <Runtime> <ReqLabels> <Continue> [MTR] [FeatureTest]
#
# Arguments:
#   Accuracy     - Optimize for accuracy (true/false)
#   Runtime      - Optimize for runtime (true/false)
#   ReqLabels    - Optimize for requested labels (true/false)
#   Continue     - Continue from previous run (true/false)
#   MTR          - Optimize for Mean Time Ratio (true/false) [optional]
#   FeatureTest  - Run feature testing mode (true/false) [optional]
#
# Examples:
#   ./run_stream_detector_optimization.sh true true false false
#   ./run_stream_detector_optimization.sh true true false false true
#   ./run_stream_detector_optimization.sh true true false false false true
#
# Features:
#   - Stream-specific detector assignments via associative array
#   - Support for parameter constraints (via constraint_<DD> arrays)
#   - Multi-objective optimization (accuracy, runtime, requested labels, MTR)
#   - MTR mode for synthetic streams with drift recovery evaluation
#   - Feature testing mode to evaluate each feature individually
#   - Continue from previous runs or start fresh
#
################################################################################

################################################################################
# Validate Input Arguments
################################################################################

if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
    echo "Error: Illegal number of parameters."
    echo "Usage: $0 <Accuracy> <Runtime> <ReqLabels> <Continue> [MTR] [FeatureTest]"
    echo "Example: $0 true true false false"
    echo "Example (with MTR): $0 true true false false true"
    echo "Example (with FeatureTest): $0 true true false false false true"
    echo ""
    echo "Arguments:"
    echo "  Accuracy     - Optimize for accuracy (true/false)"
    echo "  Runtime      - Optimize for runtime (true/false)"
    echo "  ReqLabels    - Optimize for requested labels (true/false)"
    echo "  Continue     - Continue from previous run (true/false)"
    echo "  MTR          - Optimize for Mean Time Ratio (true/false) [optional]"
    echo "  FeatureTest  - Run feature testing mode (true/false) [optional]"
    exit 1
fi

# Set MTR flag (default to false if not provided)
MTR_MODE="${5:-false}"

# Set FeatureTest flag (default to false if not provided)
FEATURE_TEST_MODE="${6:-false}"

echo "=== Optimization Configuration ==="
echo "Accuracy:     $1"
echo "Runtime:      $2"
echo "ReqLabels:    $3"
echo "Continue:     $4"
echo "MTR:          $MTR_MODE"
echo "FeatureTest:  $FEATURE_TEST_MODE"
echo "=================================="
echo ""

################################################################################
# Configure Optimization Targets
################################################################################

exp_dir=""
oo_targets=""

# Build experiment directory name and OmniOpt targets based on enabled metrics
if [ "$1" = true ] || [ "$1" = True ] || [ "$1" = 1 ]; then
    exp_dir="ACCURACY"
    oo_targets="ACCURACY=max"
fi

if [ "$2" = true ] || [ "$2" = True ] || [ "$2" = 1 ]; then
    exp_dir="${exp_dir}-RUNTIME"
    oo_targets="${oo_targets} RUNTIME=min"
fi

if [ "$3" = true ] || [ "$3" = True ] || [ "$3" = 1 ]; then
    exp_dir="${exp_dir}-REQLABELS"
    oo_targets="${oo_targets} REQLABELS=min"
fi

# Add MTR (Mean Time Ratio) metric if enabled
if [ "$MTR_MODE" = true ] || [ "$MTR_MODE" = True ] || [ "$MTR_MODE" = 1 ]; then
    exp_dir="${exp_dir}-MTR"
    oo_targets="${oo_targets} MTR=max"
fi

# Remove leading dash if present
exp_dir="${exp_dir#-}"

################################################################################
# Load Benchmark Configuration
################################################################################

source benchmark_config.sh

################################################################################
# Configure Experiment Components
################################################################################

# Classifiers to use in experiments
# HoeffdingTreeClassifier: Hoeffding Tree (incremental decision tree)
# GaussianNB: Gaussian Naive Bayes
classifiers=(
    "HoeffdingTreeClassifier"
    #"GaussianNB"
)

################################################################################
# Stream-Detector Mapping
################################################################################
# Define which drift detectors to test on which data streams.
# This allows for targeted optimization of specific stream-detector combinations.
# Format: streamdds["StreamName"]="Detector1 Detector2 ..."

declare -A streamdds

# Determine which stream configurations to use based on mode
if [ "$MTR_MODE" = true ] || [ "$MTR_MODE" = True ] || [ "$MTR_MODE" = 1 ]; then
    # MTR mode: Use synthetic streams for drift recovery evaluation
    streamdds["SineClustersPre"]="NNDVI UCDD SlidShaps"
    streamdds["WaveformPre"]="NNDVI UCDD SlidShaps"
elif [ "$FEATURE_TEST_MODE" = true ] || [ "$FEATURE_TEST_MODE" = True ] || [ "$FEATURE_TEST_MODE" = 1 ]; then
    # Feature testing mode: Use streams for feature evaluation, only valid for single variate DDs
    streamdds["NOAAWeather"]="BNDM"
    streamdds["OutdoorObjects"]="BNDM"
    streamdds["PokerHand"]="BNDM"
    streamdds["RialtoBridgeTimelapse"]="BNDM"
else
    # Standard mode: Use real-world streams
    # Active configurations
    streamdds["Electricity"]="BNDM"
fi

# Example configurations (uncomment as needed):
# Multi-detector combinations
#streamdds["PokerHand"]="IBDD SlidShaps"

################################################################################
# Run Hyperparameter Optimization
################################################################################

echo "Starting hyperparameter optimization..."
echo "Stream-Detector Combinations: ${#streamdds[@]}"
echo "Classifiers: ${#classifiers[@]}"
echo ""

# Iterate through stream-detector combinations
for stream in "${!streamdds[@]}"; do
    for clf in "${classifiers[@]}"; do
        # Parse space-separated list of detectors for this stream
        IFS=' ' read -r -a dds <<< "${streamdds[$stream]}"
        
        # Determine feature range for iteration
        if [ "$FEATURE_TEST_MODE" = true ] || [ "$FEATURE_TEST_MODE" = True ] || [ "$FEATURE_TEST_MODE" = 1 ]; then
            # Feature testing mode: iterate through all features
            declare -n nStreamFeatures=$stream
            feature_start=0
            feature_end=$nStreamFeatures
        else
            # Standard mode: no feature iteration (dummy loop runs once)
            feature_start=0
            feature_end=0
        fi
        
        # Feature iteration loop (runs once in standard mode, multiple times in feature test mode)
        for ((feature_idx=feature_start; feature_idx<=feature_end; feature_idx++)); do
            for dd in "${dds[@]}"; do
                if [ "$FEATURE_TEST_MODE" = true ] || [ "$FEATURE_TEST_MODE" = True ] || [ "$FEATURE_TEST_MODE" = 1 ]; then
                    echo "=== Processing: $dd on $stream (feature $feature_idx) with $clf ==="
                else
                    echo "=== Processing: $dd on $stream with $clf ==="
                fi
                
                # Get detector-specific hyperparameters from benchmark_config.sh
                declare -n list=$dd
                python_args_list=()
                omniopt_args_list=()
                
                # Build parameter lists for Python and OmniOpt
                for arg in "${list[@]}"; do
                    IFS=':' read -r name type dtype val rest <<< "$arg"
                    
                    # Add to Python arguments (will be substituted by OmniOpt)
                    python_args_list+=("$name")
                    python_args_list+=("%(${name})")
                    
                    # Add to OmniOpt arguments based on parameter type
                    if [[ "$type" == "choice" ]]; then
                        omniopt_args_list+=("--parameter $name $type $val")
                    elif [[ "$type" == "fixed" ]]; then
                        omniopt_args_list+=("--parameter $name $type $rest")
                    else
                        # Handle feature_id specially
                        if [[ "$name" == "feature_id" ]]; then
                            if [ "$FEATURE_TEST_MODE" = true ] || [ "$FEATURE_TEST_MODE" = True ] || [ "$FEATURE_TEST_MODE" = 1 ]; then
                                # Feature testing mode: fix feature_id to current iteration
                                omniopt_args_list+=("--parameter $name fixed ${feature_idx}")
                            else
                                # Standard mode: optimize over all features
                                declare -n nStreamFeatures=$stream
                                omniopt_args_list+=("--parameter $name $type 0 $nStreamFeatures $dtype")
                            fi
                        else
                            omniopt_args_list+=("--parameter $name $type $val $dtype")
                        fi
                    fi
                done
            
            # Handle parameter constraints if defined for this detector
            constraint_list="constraint_$dd"
            constraint_args=""
            
            # Check if constraint array exists for this detector
            if declare -p "$constraint_list" 2>/dev/null | grep -q 'declare \-a'; then
                constraint_args=("--experiment_constraints")
                
                # Access the constraint array using indirect reference
                eval "constraints=(\"\${${constraint_list}[@]}\")"
                
                # Encode each constraint as base64
                for constraint in "${constraints[@]}"; do
                    constraint_args+=" "
                    constraint_args+=$(echo "$constraint" | base64 -w0)
                done
                
                echo "Applied ${#constraints[@]} constraint(s) for $dd"
            fi

                # Build experiment name based on mode
                if [ "$FEATURE_TEST_MODE" = true ] || [ "$FEATURE_TEST_MODE" = True ] || [ "$FEATURE_TEST_MODE" = 1 ]; then
                    # Feature testing mode: include feature index in name
                    experiment_name="${dd}_${stream}_ft_${feature_idx}_${clf}_${exp_dir}"
                else
                    # Standard mode: regular naming
                    experiment_name="${dd}_${stream}_${clf}_${exp_dir}"
                fi
                
                # Check if continuing from previous run or starting new
                if [ "$4" = true ] || [ "$4" = True ] || [ "$4" = 1 ]; then
                    # Continue mode: find the most recent run directory
                    highest_dir=$(ls -d /data/horse/ws/s4122485-compPerfDD/benchmark/dfki/benchmarkdd/runs/${experiment_name}/*/ 2>/dev/null | sort -V | tail -n 1)
                    
                    if [ -z "$highest_dir" ]; then
                        echo "Warning: No previous run found for ${experiment_name}"
                        echo "Skipping..."
                        continue
                    fi
                    
                    echo "Continuing from: $highest_dir"
                    
                    # Adjust memory based on feature testing mode
                    if [ "$FEATURE_TEST_MODE" = true ] || [ "$FEATURE_TEST_MODE" = True ] || [ "$FEATURE_TEST_MODE" = 1 ]; then
                        mem_gb=128
                    else
                        mem_gb=256
                    fi
                    
                    ./OmniOpt/omniopt \
                        --continue "$highest_dir" \
                        --max_eval=50000 \
                        --worker_timeout=240 \
                        --cpus_per_task=1 \
                        --mem_gb=$mem_gb \
                        --num_cpus_main_job=8 \
                        --max_num_of_parallel_sruns=1 \
                        --revert_to_random_when_seemingly_exhausted \
                        --num_parallel_jobs=30 \
                        --decimalrounding=12 \
                        --model=SOBOL \
                        $constraint_args
                else
                    # New run mode: start fresh optimization
                    # Use unified main.py for both standard and MTR modes
                    
                    # Check if SLURM is available (check for SLURM environment or squeue command)
                    if command -v squeue &> /dev/null || [ -n "$SLURM_JOB_ID" ]; then
                        # SLURM is available - use module load and venv activation
                        if [ "$MTR_MODE" = true ] || [ "$MTR_MODE" = True ] || [ "$MTR_MODE" = 1 ]; then
                            # MTR mode: Pass Runtime and MTR flags (2 arguments)
                            run_program=$(echo -n "module load GCCcore/10.3.0 Python && source /data/horse/ws/s4122485-compPerfDD/benchmark/venv/bin/activate && python main.py $2 $MTR_MODE $stream $n_training_samples $clf $dd ${python_args_list[*]}" | base64 -w 0)
                        else
                            # Standard mode: Pass Accuracy, Runtime, ReqLabels flags (3 arguments)
                            run_program=$(echo -n "module load GCCcore/10.3.0 Python && source /data/horse/ws/s4122485-compPerfDD/benchmark/venv/bin/activate && python main.py $1 $2 $3 $stream $n_training_samples $clf $dd ${python_args_list[*]}" | base64 -w 0)
                        fi
                    else
                        # Local machine - just use python directly (assumes python is in PATH)
                        if [ "$MTR_MODE" = true ] || [ "$MTR_MODE" = True ] || [ "$MTR_MODE" = 1 ]; then
                            # MTR mode: Pass Runtime and MTR flags (2 arguments)
                            run_program=$(echo -n "python main.py $2 $MTR_MODE $stream $n_training_samples $clf $dd ${python_args_list[*]}" | base64 -w 0)
                        else
                            # Standard mode: Pass Accuracy, Runtime, ReqLabels flags (3 arguments)
                            run_program=$(echo -n "python main.py $1 $2 $3 $stream $n_training_samples $clf $dd ${python_args_list[*]}" | base64 -w 0)
                        fi
                    fi
                    
                    # Adjust memory based on feature testing mode
                    if [ "$FEATURE_TEST_MODE" = true ] || [ "$FEATURE_TEST_MODE" = True ] || [ "$FEATURE_TEST_MODE" = 1 ]; then
                        mem_gb=32
                    else
                        mem_gb=64
                    fi
                    
                    # Launch OmniOpt with configured parameters
                    omniopt \
                        --partition=romeo \
                        --experiment_name="$experiment_name" \
                        --result_names $oo_targets \
                        --mem_gb=$mem_gb \
                        --time=10080 \
                        --worker_timeout=240 \
                        --max_eval=50000 \
                        --num_parallel_jobs=30 \
                        --max_num_of_parallel_sruns=1 \
                        --num_cpus_main_job=8 \
                        --revert_to_random_when_seemingly_exhausted \
                        --generate_all_jobs_at_once \
                        --gpus=0 \
                        --num_random_steps=20 \
                        --send_anonymized_usage_stats \
                        $constraint_args \
                        --run_program="$run_program" \
                        --cpus_per_task=1 \
                        --model=SOBOL \
                        --run_mode=local \
                        "${omniopt_args_list[@]}"
                fi
                
                # Brief pause between job submissions (run in background)
                sleep 2 &
                echo ""
            done
        done
    done
done

# Wait for all background sleep processes to finish
wait

echo "=== Hyperparameter optimization jobs submitted ==="
