"""
Evaluation Configuration Module

This module contains shared configuration for all evaluation notebooks,
including dataset lists, detector names, color schemes, and common utilities.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# ============================================================================
# Dataset Configurations
# ============================================================================

# Real-world datasets for standard evaluation
REAL_DATASETS = [
    "Electricity",
    "NOAAWeather",
    "OutdoorObjects",
    "PokerHand",
    "RialtoBridgeTimelapse",
    "ForestCovertype",
    "GasSensor",
    "Ozone",
    "SensorStream"
]

# Synthetic datasets for MTR evaluation
SYNTHETIC_DATASETS = [
    "WaveformPre",
    "SineClustersPre"
]

# All datasets
ALL_DATASETS = REAL_DATASETS + SYNTHETIC_DATASETS

# ============================================================================
# Drift Detector Configurations
# ============================================================================

# All drift detectors
ALL_DETECTORS = [
    "CSDDM", "BNDM", "D3", "IBDD",
    "OCDD", "SPLL", "UDetect", "EDFS",
    "NNDVI", "UCDD", "STUDD", "DDAL",
    "DAWIDD", "IKS", "HDDDM", "PCACD",
    "CDBD", "SlidShaps",
    "WindowKDE", "CDLEEDS"
]

# Single-variate drift detectors
SINGLE_VARIATE_DETECTORS = [
    "CDBD",
    "IKS",
    "WindowKDE"
]

# Multi-variate drift detectors
MULTI_VARIATE_DETECTORS = [
    dd for dd in ALL_DETECTORS if dd not in SINGLE_VARIATE_DETECTORS
]

# ============================================================================
# Color Scheme for Visualizations
# ============================================================================

DETECTOR_COLORS = {
    "CSDDM": "#1f77b4",      # Blue
    "BNDM": "#ff7f0e",       # Orange
    "D3": "#2ca02c",         # Green
    "IBDD": "#d62728",       # Red
    "OCDD": "#9467bd",       # Purple
    "SPLL": "#8c564b",       # Brown
    "UDetect": "#e377c2",    # Pink
    "EDFS": "#7f7f7f",       # Gray
    "NNDVI": "#bcbd22",      # Olive
    "UCDD": "#17becf",       # Cyan
    "STUDD": "#ffbb78",      # Light Orange
    "DDAL": "#ff9896",       # Light Red
    "DAWIDD": "#c5b0d5",     # Light Purple
    "IKS": "#c49c94",        # Light Brown
    "HDDDM": "#f7b6d2",      # Light Pink
    "PCACD": "#dbdb8d",      # Light Olive
    "CDBD": "#9edae5",       # Light Cyan
    "MCDDD": "#f5b0b0",      # Very Light Red
    "SlidShaps": "#aec7e8",  # Light Blue
    "WindowKDE": "#c7c7c7",  # Light Gray
    "CDLEEDS": "#ffbb78"     # Peach
}

# ============================================================================
# Path Configurations
# ============================================================================

# Base path for experiment results
# e.g. /home/runs/
BASE_PATH = "<your basepath>"

# Classifier used in experiments
DEFAULT_CLASSIFIER = "HoeffdingTreeClassifier"

# ============================================================================
# Utility Functions
# ============================================================================

def get_experiment_path(detector: str, dataset: str, metrics: List[str],
                       classifier: str = DEFAULT_CLASSIFIER,
                       base_path: str = BASE_PATH) -> str:
    """
    Construct the experiment path for a given detector-dataset combination.
    
    Args:
        detector: Drift detector name
        dataset: Dataset name
        metrics: List of metric names (e.g., ['ACCURACY', 'RUNTIME'])
        classifier: Classifier name (default: HoeffdingTreeClassifier)
        base_path: Base directory for experiments
        
    Returns:
        Full path to experiment directory
    """
    metrics_str = "-".join(metrics)
    exp_name = f"{detector}_{dataset}_{classifier}_{metrics_str}"
    return os.path.join(base_path, exp_name)


def load_results(detector: str, dataset: str, metrics: List[str],
                classifier: str = DEFAULT_CLASSIFIER,
                base_path: str = BASE_PATH) -> Tuple[pd.DataFrame, str]:
    """
    Load experiment results for a detector-dataset combination.
    
    Args:
        detector: Drift detector name
        dataset: Dataset name
        metrics: List of metric names
        classifier: Classifier name
        base_path: Base directory for experiments
        
    Returns:
        Tuple of (results DataFrame, path to results file)
        Returns (None, None) if results don't exist
    """
    exp_path = get_experiment_path(detector, dataset, metrics, classifier, base_path)
    
    if not os.path.exists(exp_path):
        print(f"Experiment path not found: {exp_path}")
        return None, None
    
    # Find the most recent run
    run_dirs = [int(d) for d in os.listdir(exp_path) if d.isdigit()]
    if not run_dirs:
        print(f"No run directories found in: {exp_path}")
        return None, None
    
    latest_run = max(run_dirs)
    results_path = os.path.join(exp_path, str(latest_run), "results.csv")
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return None, None
    
    try:
        df = pd.read_csv(results_path)
        # Filter out abandoned trials
        df = df[df["trial_status"] != "ABANDONED"]
        return df, results_path
    except Exception as e:
        print(f"Error loading results from {results_path}: {e}")
        return None, None


def get_best_result(df: pd.DataFrame, metric: str, 
                   maximize: bool = True) -> pd.Series:
    """
    Get the best result from a results DataFrame.
    
    Args:
        df: Results DataFrame
        metric: Metric column name to optimize
        maximize: If True, find maximum; if False, find minimum
        
    Returns:
        Series containing the best result row
    """
    if df is None or len(df) == 0:
        return None
    
    # Only consider completed trials
    df_completed = df[df["trial_status"] == "COMPLETED"]
    
    if len(df_completed) == 0:
        return None
    
    if maximize:
        best_idx = df_completed[metric].idxmax()
    else:
        best_idx = df_completed[metric].idxmin()
    
    return df_completed.loc[best_idx]


def count_experiments(detectors: List[str], datasets: List[str],
                     metrics: List[str], classifier: str = DEFAULT_CLASSIFIER,
                     base_path: str = BASE_PATH) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Count evaluated and completed experiments for all detector-dataset combinations.
    
    Args:
        detectors: List of drift detector names
        datasets: List of dataset names
        metrics: List of metric names
        classifier: Classifier name
        base_path: Base directory for experiments
        
    Returns:
        Nested dict: {detector: {dataset: (evaluated, completed)}}
    """
    results = {}
    
    for detector in detectors:
        results[detector] = {}
        for dataset in datasets:
            df, _ = load_results(detector, dataset, metrics, classifier, base_path)
            
            if df is not None:
                evaluated = len(df)
                completed = len(df[df["trial_status"] == "COMPLETED"])
                results[detector][dataset] = (evaluated, completed)
            else:
                results[detector][dataset] = (0, 0)
    
    return results


def normalize_metrics(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Normalize metrics to [0, 1] range for comparison.
    
    Args:
        df: DataFrame with metric columns
        metrics: List of metric column names to normalize
        
    Returns:
        DataFrame with normalized metrics
    """
    df_norm = df.copy()
    
    for metric in metrics:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            if max_val > min_val:
                df_norm[f"{metric}_norm"] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[f"{metric}_norm"] = 0.5  # All values are the same
    
    return df_norm


# ============================================================================
# Plotting Utilities
# ============================================================================

def setup_plot_style():
    """Configure matplotlib/seaborn style for consistent plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def get_detector_color(detector: str) -> str:
    """
    Get the color for a detector, with fallback to default.
    
    Args:
        detector: Drift detector name
        
    Returns:
        Hex color string
    """
    return DETECTOR_COLORS.get(detector, "#808080")  # Gray as default
