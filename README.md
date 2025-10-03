# Computational Performance of Semi- and Unsupervised Concept Drift Detection: A Survey and Multiobjective Benchmark using Bayesian Optimization

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

### Key Features

- **Currently implemented detectors**: CSDDM, BNDM, D3, IBDD, OCDD, SPLL, UDetect, EDFS, NNDVI, UCDD, STUDD, DDAL, DAWIDD, IKS, HDDDM, PCACD, CDBD, SlidShaps, WindowKDE, CDLEEDS

- **Multi-Objective Optimization**: Simultaneously optimize for accuracy, runtime, label efficiency, MTR or others

- **Feature Testing Mode**: Single-variate detector evaluation

## Repository Structure

```
benchmarkdd/
├── datasets/                    # Dataset definitions and loaders
│   ├── files.tar.gz            # CSV data files to unpack
│   └── *.py                    # Dataset classes (electricity.py, etc.)
│
├── detectors/                   # Drift detector implementations
│   ├── base.py                 # Base detector class
│   └── *.py                    # Individual detectors (csddm.py, etc.)
│
├── metrics/                      # Performance metrics
│   ├── drift.py                 # Drift detection metrics
│   ├── computational_metrics.py # Runtime and memory metrics
│   └── lift_per_drift.py        # Lift per drift calculation
│
├── evaluation_notebooks/        # Analysis and visualization notebooks
│   ├── evaluation_unified.ipynb        # Experiment status and data loading
│   ├── evaluation_visualization.ipynb  # Performance visualization
│   ├── prediction_analysis.ipynb       # Prediction pattern analysis
│   ├── evaluation_radar_graphs.ipynb   # Multi-dimensional comparison
│   └── eval_config.py                  # Shared configuration
│
├── model/                       # Pre-trained classifier models
│   └── HoeffdingTreeClassifier/ # Hoeffding Tree models
│
├── results/                     # Experiment results
│   ├── baselines/              # Baseline results
│   └── omniopt_results/        # OmniOpt optimization results
│
├── runs/                        # Active experiment runs (created by OmniOpt)
│   └── [detector]_[dataset]_[classifier]_[metrics]/
│
├── test/                        # Unit and integration tests
│
├── main.py                      # Main experiment runner
├── train_classifiers.py         # Classifier training
├── compute_baselines.py         # Baseline computation
├── config.py                    # Global configuration
├── requirements.txt             # Python dependencies
│
├── run_stream_detector_optimization.sh     # Main benchmark script
├── run_stream_detector_optimization.sbatch # SLURM batch script
├── run_train_classifiers.sh              # Classifier training script
├── run_baselines.sh                      # Baseline computation script
└── benchmark_config.sh                   # Benchmark configuration
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)
- SLURM based scheduler for HPC execution

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd benchmarkdd
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
python main.py True True False Electricity 1600 HoeffdingTreeClassifier CSDDM recent_samples_size 1000 n_samples 500 confidence 0.05 feature_proportion 0.5 n_clusters 5
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

### Using Jupyter Notebooks

Navigate to `evaluation_notebooks/` for comprehensive analysis:

```bash
cd evaluation_notebooks
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
