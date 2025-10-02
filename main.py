#!/usr/bin/env python3
"""
Drift Detector Benchmark - Main Execution Script

This script runs drift detection experiments on streaming data with configurable
optimization objectives. It supports multiple benchmark modes:
  - Standard mode: Optimize for accuracy, runtime, and/or requested labels
  - MTR mode: Optimize for Mean Time Ratio (for synthetic streams)

Usage:
    Standard mode (3 boolean flags):
        python main.py <Accuracy> <Runtime> <ReqLabels> <Dataset> <TrainSamples> <Classifier> <Detector> [params...]
    
    MTR mode (2 boolean flags):
        python main.py <Runtime> <MTR> <Dataset> <TrainSamples> <Classifier> <Detector> [params...]

Arguments:
    Accuracy/Runtime/ReqLabels/MTR - Boolean flags (True/False/1/0) for optimization objectives
    Dataset                        - Dataset class name or expression (e.g., 'Electricity' or 'SineClustersPre()')
    TrainSamples                   - Number of training samples (int)
    Classifier                     - Classifier class name (e.g., 'HoeffdingTreeClassifier')
    Detector                       - Drift detector class name with optional parameters
    params                         - Optional detector parameters in key-value pairs

Examples:
    python main.py True True False Electricity 1600 HoeffdingTreeClassifier ADWIN delta 0.002
    python main.py True False SineClustersPre 1600 HoeffdingTreeClassifier MCDDD window_size 100

Output:
    Prints general information, detected drift points, predictions, and OmniOpt-compatible metrics
"""

import os
import sys
import ast
import warnings

# Suppress warnings for performance
warnings.filterwarnings("ignore")

# Imports (classes are now accessible via globals())
from optimization.classifiers import *
from metrics.metrics import get_metrics
from datasets import *
from detectors import *
import detectors.base


def parse_expression(expr_str):
    """
    Parses a string like 'ClassName(arg=value, ...)' or 'ClassName()'
    and returns (class name, instantiated object).
    
    Args:
        expr_str: String expression to parse
        
    Returns:
        Tuple of (class_name, instantiated_object)
        
    Raises:
        ValueError: If expression format is invalid
    """
    try:
        node = ast.parse(expr_str, mode='eval').body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            class_name = node.func.id
            obj = eval(expr_str, globals())
            return class_name, obj
        elif isinstance(node, ast.Name):
            class_name = node.id
            obj = eval(expr_str + "()", globals())
            return class_name, obj
        else:
            raise ValueError(f"Invalid expression format: {expr_str}")
    except Exception as e:
        raise ValueError(f"Error parsing expression '{expr_str}': {e}")


def detect_mode(argv):
    """
    Detects whether we're in standard mode (3 flags) or MTR mode (2 flags).
    
    Args:
        argv: Command line arguments
        
    Returns:
        Tuple of (mode, arg_offset) where mode is 'standard' or 'mtr',
        and arg_offset is the index where dataset argument starts
    """
    # Try to determine mode by checking if argv[3] looks like a dataset name
    # In standard mode: argv[1-3] are flags, argv[4] is dataset
    # In MTR mode: argv[1-2] are flags, argv[3] is dataset
    
    # Simple heuristic: if argv[3] starts with uppercase, it's likely a dataset (MTR mode)
    # Otherwise, it's a boolean flag (standard mode)
    if len(argv) > 3:
        third_arg = str(argv[3])
        # Check if it looks like a boolean
        if third_arg.lower() in ('true', 'false', '1', '0'):
            return 'standard', 4  # Standard mode, dataset at index 4
        else:
            return 'mtr', 3  # MTR mode, dataset at index 3
    
    # Default to standard mode if we can't determine
    return 'standard', 4


if __name__ == '__main__':
    # Detect execution mode
    mode, dataset_idx = detect_mode(sys.argv)
    
    # Parse arguments based on mode
    if mode == 'mtr':
        # MTR mode: 2 boolean flags (runtime, mtr)
        benchmark_accuracy = False
        benchmark_runtime = bool(eval(sys.argv[1]))
        benchmark_reqlabels = False
        benchmark_mtr = bool(eval(sys.argv[2]))
        dataset_expr = sys.argv[3]
        n_training_samples = int(sys.argv[4])
        classifier_expr = sys.argv[5]
        detector_start_idx = 6
    else:
        # Standard mode: 3 boolean flags (accuracy, runtime, reqlabels)
        benchmark_accuracy = bool(eval(sys.argv[1]))
        benchmark_runtime = bool(eval(sys.argv[2]))
        benchmark_reqlabels = bool(eval(sys.argv[3]))
        benchmark_mtr = False
        dataset_expr = sys.argv[4]
        n_training_samples = int(sys.argv[5])
        classifier_expr = sys.argv[6]
        detector_start_idx = 7
    
    # Reconstruct detector expression from the rest of the arguments
    detector_expr = sys.argv[detector_start_idx]
    if detector_expr is None:
        print("Error: Not enough arguments - detector expression required")
        sys.exit(1)
        
    if "(" in detector_expr:
        # Valid Python syntax: join everything from detector onwards
        detector_expr = ''.join(sys.argv[detector_start_idx:])
    else:
        # Key-value pairs: DD arg1 val1 arg2 val2... -> build Python string
        args = ""
        key = True
        for arg in sys.argv[detector_start_idx + 1:]:
            if key:
                args += arg + "="
                key = False
            else:
                args += arg + ","
                key = True
        detector_expr += "(" + args + ")"

    # Parse and instantiate components
    dataset_string, stream = parse_expression(dataset_expr)
    classifier_string, classifier = parse_expression(classifier_expr)
    drift_detector_string, drift_detector = parse_expression(detector_expr)
    
    # Special handling for CDLEEDS detector (requires higher recursion limit)
    if drift_detector_string == "CDLEEDS":
        sys.setrecursionlimit(10000)
    
    # Construct classifier model path
    classifier_path = (os.getcwd() + "/model/" + classifier_string + "/" +
                       classifier_string + "_" + dataset_string + ".pkl")

    # Run drift detection on stream
    (drifts, labels, predictions, n_req_labels, runtime, peak_memory,
     mean_memory) = drift_detector.run_stream(stream, n_training_samples,
                                              classifier_path)

    # Calculate metrics (always computed for comprehensive output)
    metrics = get_metrics(stream, drifts, labels, predictions, n_req_labels,
                          n_training_samples)

    # ========================================================================
    # Output Results
    # ========================================================================
    
    # General information
    print(f"General Info: Drift Detector: {drift_detector_string}"
          f"\n\tParameters: {drift_detector.__dict__}"
          f"\n\tDataset: {dataset_string}"
          f"\n\tn_training_samples: {n_training_samples}"
          f"\n\tClassifier: {classifier_string}"
          f"\n\tMode: {mode.upper()}")

    print(f"General Info: detected drift points: {drifts}")

    # True/False predictions for visualization
    true_false_predictions = [int(l == p) for l, p in zip(labels, predictions)]
    print(f"General Info: True/False Predictions: {true_false_predictions}")

    # Optimization objectives (uppercase for OmniOpt parsing)
    if benchmark_accuracy:
        print(f"ACCURACY: {metrics.accuracy:.2f}")
    if benchmark_runtime:
        print(f"RUNTIME: {runtime:.0f}")
    if benchmark_reqlabels:
        print(f"REQLABELS: {metrics.portion_req_labels:.2f}")
    if benchmark_mtr:
        print(f"MTR: {metrics.mtr:.2f}")

    # OmniOpt-compatible output (all metrics for logging)
    print(f"OO-Info: runtime: {runtime:.0f}")
    print(f"OO-Info: peak_memory: {peak_memory}")
    print(f"OO-Info: mean_memory: {mean_memory}")
    print(f"OO-Info: accuracy: {metrics.accuracy:.2f}")
    print(f"OO-Info: portion_req_label: {metrics.portion_req_labels:.2f}")
    print(f"OO-Info: lpd: {metrics.lpd:.2f}")
    
    # MTR-specific output (only meaningful for synthetic streams with known drift points)
    if benchmark_mtr:
        print(f"OO-Info: mtr: {metrics.mtr:.2f}")
