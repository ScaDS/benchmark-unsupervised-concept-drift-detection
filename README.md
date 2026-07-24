# Computational Performance of Unsupervised Concept Drift Detection: A Survey and Multiobjective Benchmark using Bayesian Optimization

A comprehensive benchmark suite for evaluating drift detection algorithms on data streams using multi-objective Bayesian optimization.

## Related Work

This benchmark builds upon the initial work on unsupervised concept drift detection:
- **Original Repository**: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection)

## Overview

This benchmark suite provides a systematic framework for evaluating and comparing drift detection algorithms across multiple dimensions:

- **Multi-objective optimization** using Bayesian optimization (via OmniOpt)
- **20 drift detectors** including both supervised and unsupervised methods
- **Real-world and synthetic datasets** for comprehensive evaluation
- **Multiple performance metrics**: Accuracy, Runtime, Requested Labels, Mean Time Ratio (MTR)
- **Automated hyperparameter tuning** for fair comparison
- **Scalable execution** on both HPC clusters (SLURM) and local machines

> **Online Results:** Benchmark results are explorable and presented via the interactive dashboard at **<https://imageseg.scads.de/omniax/conceptdrift/>**.

### Key Features

- **Currently implemented detectors**: CSDDM, BNDM, D3, IBDD, OCDD, SPLL, UDetect, EDFS, NNDVI, UCDD, STUDD, DDAL, DAWIDD, IKS, HDDDM, PCACD, CDBD, SlidShaps, WindowKDE, CDLEEDS

- **Multi-Objective Optimization**: Simultaneously optimize for accuracy, runtime, label efficiency, MTR or others

- **Feature Testing Mode**: Single-variate detector evaluation

## Repository Structure

```
benchmark-unsupervised-concept-drift-detection/
├── datasets/                    # Dataset definitions and loaders
│   ├── files/                  # CSV/ARFF data files
│   ├── convert_datasets.py     # ARFF → CSV conversion utility
│   └── *.py                    # Dataset classes (electricity.py, noaa_weather.py, etc.)
│
├── detectors/                   # Drift detector implementations
│   ├── base.py                 # Base detector class
│   └── *.py                    # Individual detectors (csddm.py, bndm.py, etc.)
│
├── metrics/                     # Performance metrics
│   ├── drift.py                 # Drift detection metrics
│   ├── computational_metrics.py # Runtime and memory metrics
│   ├── lift_per_drift.py        # Lift per drift calculation
│   ├── metrics.py               # Core metric aggregation
│   └── requested_labels.py      # Requested labels metric
│
├── optimization/                # Hyperparameter optimization framework
│   ├── classifiers.py           # Classifier definitions
│   ├── config_generator.py      # OmniOpt configuration generation
│   ├── logger.py                # Optimization logging
│   ├── model_optimizer.py       # Model optimization logic
│   └── parameter.py             # Parameter definitions
│
├── evaluation/                  # Experiment tracking and final output generation
│   ├── eval_config.py                  # Shared configuration for notebooks
│   ├── evaluation_unified.ipynb        # Experiment status and completion tracking
│   ├── evaluation_visualization.ipynb  # Performance visualization (Pareto fronts, heatmaps)
│   ├── prediction_analysis.ipynb       # Prediction pattern and ensemble analysis
│   ├── plot_pareto_fronts.py           # Final Pareto-front plot generation → pareto_fronts/
│   ├── create_all_mrp_tables.py        # Final MRP runtime table generation (LaTeX)
│   └── pareto_fronts/                  # Generated Pareto-front PNGs
│
├── posthoc_analysis/            # Post-hoc stability and reproducibility studies
│   ├── reproducibility/        # Re-run all benchmark configs and compare
│   │   ├── reproduce_benchmark.py       # Main reproduction runner
│   │   ├── submit_jobs.py              # SLURM job submission
│   │   ├── run_reproduction.sbatch     # SLURM batch template
│   │   └── analyze_reproducibility.py  # Reproducibility analysis
│   ├── multiseed/              # Multi-seed stability (10 seeds, cross-seed correlations)
│   │   ├── multi_seed_analysis.py           # Main multi-seed runner
│   │   ├── submit_multi_seed_jobs.py        # SLURM job submission
│   │   ├── run_multi_seed.sbatch            # SLURM batch template
│   │   ├── analyze_multiseed_reproducibility.py  # Multi-seed stability analysis
│   │   └── show_summary_correlations.py    # Cross-seed correlation summary
│   └── runtime_stability/      # Runtime + accuracy variance over repeated runs
│       ├── runtime_accuracy_stability_study.py      # Main stability runner
│       ├── submit_runtime_accuracy_stability_jobs.py # SLURM job submission
│       ├── run_runtime_accuracy_stability_study.sbatch # SLURM batch template
│       └── summarize_stability.py                    # Stability summary
│
├── model/                       # Pre-trained classifier models
│   └── HoeffdingTreeClassifier/ # Hoeffding Tree models
│
├── results/                     # Experiment results
│   ├── all_benchmark_results/  # OmniOpt optimization results (per detector/dataset)
│   ├── archive/                # Archived results (omniopt_results.tar.gz)
│   └── baseline_results.tar.gz # Baseline results archive
│
├── test/                        # Unit and integration tests
│
├── main.py                      # Main experiment runner
├── train_classifiers.py         # Classifier training
├── compute_baselines.py         # Baseline computation
├── config.py                    # Global configuration
├── runner.py                    # Batch experiment runner
├── sanity_check.py              # Verify all repository functionalities work
├── requirements.txt             # Python dependencies
│
├── run_stream_detector_optimization.sh     # Main benchmark script
├── run_stream_detector_optimization.sbatch # SLURM batch script
├── run_train_classifiers.sh              # Classifier training script
├── run_train_classifiers.sbatch          # SLURM batch script
├── run_baselines.sh                      # Baseline computation script
├── run_baselines.sbatch                  # SLURM batch script
└── benchmark_config.sh                   # Benchmark configuration
```

## Sanity Check

A sanity check script is provided to verify that all functionalities of the repository are working correctly. It briefly exercises each component (imports, dataset loading, main runner in both modes, baseline computation, classifier training, evaluation scripts, post-hoc analysis scripts, unit tests, and configuration) and reports which checks pass or fail.

```bash
python sanity_check.py
```

The script runs each check with a small subset of data and short timeouts. All 16 checks should pass on a properly configured installation.

> **Note:** The sanity check runs entirely in the current allocation (or local machine) and does **not** submit any SLURM jobs. The `.sbatch` scripts referenced in the post-hoc analysis sections are HPC-cluster-specific and are only validated for importability and CLI correctness here. To actually launch cluster jobs, use the respective `submit_*.py` scripts on a SLURM-enabled environment.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)
- SLURM based scheduler for HPC execution

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd benchmark-unsupervised-concept-drift-detection
```

2. **Install dependencies**
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

3. **Prepare datasets**
```bash
# Datasets can be downloaded from [USP DS Repository](https://sites.google.com/view/uspdsrepository) and extracted in `datasets/files`
# Or download additional datasets if needed
```

4. **Train classifiers** (optional, pre-trained models included)
```bash
# On local machine
./run_train_classifiers.sh

# On HPC cluster
sbatch run_train_classifiers.sbatch
```

## Running the Benchmark

### Direct Execution (Without OmniOpt)

You can run a specific detector with a specific configuration directly using `main.py`, without the hyperparameter optimization:

**Standard Mode (ACCURACY-RUNTIME-REQLABELS):**
```bash
python main.py <Accuracy> <Runtime> <ReqLabels> <Dataset> <TrainSamples> <Classifier> <Detector> [params...]
```

**MTR Mode (RUNTIME-MTR):**
```bash
python main.py <Runtime> <MTR> <Dataset> <TrainSamples> <Classifier> <Detector> [params...]
```

**Arguments:**
- `Accuracy/Runtime/ReqLabels/MTR` - Boolean flags (True/False/1/0) for which metrics to optimize
- `Dataset` - Dataset name (e.g., 'Electricity', 'NOAAWeather')
- `TrainSamples` - Number of training samples (typically 2000)
- `Classifier` - Classifier name (e.g., 'HoeffdingTreeClassifier')
- `Detector` - Drift detector name (e.g., 'CSDDM', 'BNDM')
- `params` - Detector-specific parameters as key-value pairs

**Examples:**

```bash
# Run CSDDM on Electricity dataset, optimize for accuracy and runtime
python main.py True True False Electricity 2000 HoeffdingTreeClassifier CSDDM recent_samples_size 1000 n_samples 500 confidence 0.05 feature_proportion 0.5 n_clusters 5
```


### Automated Optimization with OmniOpt

For automated hyperparameter optimization across multiple configurations:

#### Configuration

Edit `run_stream_detector_optimization.sh` to configure your experiments:

```bash
# Select evaluation mode
MTR_MODE=false              # Set to true for MTR evaluation
FEATURE_TEST_MODE=false     # Set to true for feature testing

# Configure detector-dataset combinations
streamdds["Electricity"]="CSDDM BNDM D3"
streamdds["NOAAWeather"]="CSDDM BNDM"
# Add more combinations as needed
```

### Running on Local Machine

The script automatically detects local execution and adjusts accordingly:

```bash
# Run benchmark
./run_stream_detector_optimization.sh
```

**Example: ACCURACY-RUNTIME-REQLABELS**
```bash
# Edit run_stream_detector_optimization.sh
MTR_MODE=false
FEATURE_TEST_MODE=false

# Configure detectors and datasets
streamdds["Electricity"]="CSDDM BNDM D3 IBDD"
streamdds["NOAAWeather"]="CSDDM BNDM"

# Run
./run_stream_detector_optimization.sh
```

**Example: MTR (Mean Time Ratio)**
```bash
# Edit run_stream_detector_optimization.sh
MTR_MODE=true
FEATURE_TEST_MODE=false

# Configure synthetic datasets
streamdds["SineClustersPre"]="CSDDM BNDM D3"
streamdds["WaveformPre"]="CSDDM BNDM"

# Run
./run_stream_detector_optimization.sh
```

**Example: Feature Testing Mode (single variate detectors)**
```bash
# Edit run_stream_detector_optimization.sh
MTR_MODE=false
FEATURE_TEST_MODE=true

# Configure single-variate detectors
streamdds["Electricity"]="CDBD IKS WindowKDE"

# Run
./run_stream_detector_optimization.sh
```

### Running on HPC Cluster (SLURM)

The script automatically detects SLURM and uses appropriate commands:

```bash
# Submit batch job
sbatch run_stream_detector_optimization.sbatch

# Or run interactively in a job allocation
./run_stream_detector_optimization.sh
```

### Advanced Configuration

**Adjust OmniOpt parameters:**
```bash
# In run_stream_detector_optimization.sh, modify OmniOpt parameters:
--max_eval=50000              # Maximum evaluations
--num_parallel_jobs=30        # Parallel jobs
--worker_timeout=240          # Timeout per evaluation (minutes)
--mem_gb=64                   # Memory per job (GB)
--time=10080                  # Total time limit (minutes)
```

Please check the OmniOpt documentation for further details:
https://imageseg.scads.de/omniax/tutorials


## Analyzing Results

### Exploratory Notebooks

The `evaluation/` directory contains Jupyter notebooks for tracking experiment progress and exploring results interactively during and after benchmark runs:

```bash
cd evaluation
jupyter notebook
```

**Available notebooks:**

1. **`evaluation_unified.ipynb`** - Experiment status and completion
   - Check which experiments have completed
   - Identify missing or incomplete runs
   - View completion statistics

2. **`evaluation_visualization.ipynb`** - Performance visualization
   - Pareto front analysis
   - Performance heatmaps
   - Best configuration identification
   - Export results to CSV

3. **`prediction_analysis.ipynb`** - Prediction pattern analysis
   - Time-series visualization of predictions
   - Ensemble analysis
   - Minimal detector set identification

These notebooks share configuration (dataset lists, detector names, color schemes, result-loading utilities) via `eval_config.py`.

### Final Output Generation

After the benchmark is complete, two standalone scripts produce the definitive, reproducible outputs:

- **`plot_pareto_fronts.py`** - Generates Pareto-front plots (accuracy vs. runtime) per dataset → `evaluation/pareto_fronts/`
- **`create_all_mrp_tables.py`** - Reproduces all MRP runtime tables from benchmark results and optionally generates LaTeX output

### Post-Hoc Analysis

The `posthoc_analysis/` directory contains three independent post-hoc experiment types for assessing the stability and reproducibility of benchmark results:

1. **Reproducibility** (`posthoc_analysis/reproducibility/`) - Re-run all configurations from `results/all_benchmark_results/` and compare against the original OmniOpt results.

2. **Multi-Seed Stability** (`posthoc_analysis/multiseed/`) - Run each configuration with 10 different random seeds and compute cross-seed correlations for accuracy, runtime, and hypervolume.

3. **Runtime + Accuracy Stability** (`posthoc_analysis/runtime_stability/`) - Re-execute sampled configurations multiple times to measure runtime and accuracy variance.

Each subdirectory contains its own SLURM job submission scripts and analysis tools.

## Performance Metrics

The benchmark evaluates detectors across multiple dimensions:

### Standard Metrics
- **Accuracy**: Classification accuracy with drift detection
- **Runtime**: Total execution time (seconds)
- **Requested Labels**: Number of labels requested (for semi-supervised methods)
- **Memory**: Peak and mean memory usage (MB)

### MTR Metrics (Synthetic Datasets)
- **Mean Time Ratio (MTR)**: Ratio of time to recover from drift
- **Runtime**: Execution time
- **Accuracy**: Overall classification accuracy

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Original unsupervised drift detection work: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection)
- OmniOpt framework for Bayesian optimization
- River library for online machine learning
- All contributors and researchers in the drift detection community

## Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Contact: [elias.werner@tu-dresden.de]
