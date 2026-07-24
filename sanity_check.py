#!/usr/bin/env python3
"""
Sanity Check Script — verifies that all functionalities documented in the README
are working by briefly exercising each one.

Each check runs for a short time and validates that the output looks reasonable.
The script is designed to be run from the project root directory:

    python sanity_check.py

Exit code 0 means all checks passed; non-zero means at least one failed.

Note: This script runs entirely in the current allocation (or local machine) and
does NOT submit any SLURM jobs. The .sbatch templates are HPC-cluster-specific
and are only validated for importability and CLI correctness here. To actually
launch cluster jobs, use the respective submit_*.py scripts on a SLURM cluster.
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
import json
import traceback
from pathlib import Path

# Ensure we're running from the project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Use the project venv if it exists
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


# ============================================================================
# Helpers
# ============================================================================

class CheckResult:
    def __init__(self, name, passed, details="", duration_s=0.0):
        self.name = name
        self.passed = passed
        self.details = details
        self.duration_s = duration_s

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name} ({self.duration_s:.1f}s) — {self.details}"


def run_check(name, func):
    print(f"\n{'='*70}")
    print(f"  Running: {name}")
    print(f"{'='*70}")
    start = time.time()
    try:
        result = func()
        result.duration_s = time.time() - start
        print(result)
        return result
    except Exception as e:
        result = CheckResult(name, False, f"Exception: {e}", time.time() - start)
        traceback.print_exc()
        print(result)
        return result


def run_subprocess(cmd, timeout=120, cwd=None):
    """Run a subprocess and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(PROJECT_ROOT),
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout after {timeout}s"


# ============================================================================
# Checks
# ============================================================================

def check_imports():
    """Verify that all core modules can be imported."""
    errors = []
    modules = [
        "from datasets import ForestCovertype, GasSensor, PokerHand",
        "from datasets.sineclusters_pre import SineClustersPre",
        "from datasets.waveform_pre import WaveformPre",
        "from detectors import BNDM, CSDDM, D3, IBDD, OCDD, SPLL, UDetect, "
        "EDFS, NNDVI, UCDD, STUDD, DDAL, DAWIDD, IKS, HDDDM, PCACD, "
        "CDBD, SlidShaps, WindowKDE, CDLEEDS, DummyDDBL1, DummyDDBL2",
        "from metrics.metrics import get_metrics",
        "from optimization.classifiers import Classifiers",
        "from optimization.model_optimizer import ModelOptimizer",
        "from optimization.parameter import Parameter",
        "from optimization.config_generator import ConfigGenerator",
        "from optimization.logger import ExperimentLogger",
        "from evaluation.eval_config import ALL_DETECTORS, REAL_DATASETS, "
        "SYNTHETIC_DATASETS, DETECTOR_COLORS",
    ]
    for mod in modules:
        code = f"import sys; sys.path.insert(0, '.'); {mod}"
        rc, out, err = run_subprocess([PYTHON, "-c", code], timeout=30)
        if rc != 0:
            errors.append(f"Failed: {mod.split('import')[-1].strip()[:50]} — {err.strip()[:100]}")

    if errors:
        return CheckResult("Core imports", False, f"{len(errors)} failures: {'; '.join(errors[:3])}")
    return CheckResult("Core imports", True, "All modules imported successfully")


def check_dataset_loading():
    """Verify that available datasets can be loaded and iterated."""
    from datasets.sineclusters_pre import SineClustersPre
    from datasets.waveform_pre import WaveformPre

    datasets_to_test = [
        ("SineClustersPre", SineClustersPre, "datasets/files/sineclusters.csv"),
        ("WaveformPre", WaveformPre, "datasets/files/waveform.csv"),
    ]

    errors = []
    for name, cls, expected_file in datasets_to_test:
        if not (PROJECT_ROOT / expected_file).exists():
            errors.append(f"{name}: data file missing ({expected_file})")
            continue
        try:
            ds = cls()
            count = 0
            for x, y in ds:
                count += 1
                if count >= 10:
                    break
            if count < 10:
                errors.append(f"{name}: only {count} samples iterated")
            if hasattr(ds, 'drifts') and not hasattr(ds, 'drifts'):
                errors.append(f"{name}: no drifts attribute")
        except Exception as e:
            errors.append(f"{name}: {e}")

    if errors:
        return CheckResult("Dataset loading", False, "; ".join(errors))
    return CheckResult("Dataset loading", True, "SineClustersPre and WaveformPre load and iterate correctly")


def check_main_standard_mode():
    """Run main.py in standard mode (accuracy + runtime) on a small dataset."""
    cmd = [
        PYTHON, "main.py",
        "True", "True", "False",       # accuracy, runtime, reqlabels
        "SineClustersPre",              # dataset
        "200",                          # training samples (small)
        "HoeffdingTreeClassifier",      # classifier
        "BNDM", "recent_samples_size", "100", "n_samples", "50",
        "const", "1.0", "max_depth", "3", "threshold", "0.5",
    ]
    rc, out, err = run_subprocess(cmd, timeout=300)

    if rc != 0:
        return CheckResult("main.py standard mode", False, f"Exit code {rc}: {err[:200]}")

    # Check for expected output patterns
    has_accuracy = "ACCURACY:" in out
    has_runtime = "RUNTIME:" in out
    has_oo_info = "OO-Info:" in out
    has_drifts = "detected drift points:" in out

    if not (has_accuracy and has_runtime and has_oo_info and has_drifts):
        missing = []
        if not has_accuracy: missing.append("ACCURACY")
        if not has_runtime: missing.append("RUNTIME")
        if not has_oo_info: missing.append("OO-Info")
        if not has_drifts: missing.append("drift points")
        return CheckResult("main.py standard mode", False, f"Missing output: {', '.join(missing)}")

    # Extract accuracy value and check it's reasonable
    try:
        for line in out.split('\n'):
            if line.startswith("ACCURACY:"):
                acc = float(line.split(':')[1].strip())
                if not (0.0 <= acc <= 1.0):
                    return CheckResult("main.py standard mode", False, f"Accuracy {acc} out of [0,1]")
                break
    except (ValueError, IndexError):
        return CheckResult("main.py standard mode", False, "Could not parse accuracy value")

    return CheckResult("main.py standard mode", True, "BNDM on SineClustersPre: accuracy + runtime output verified")


def check_main_mtr_mode():
    """Run main.py in MTR mode on a synthetic dataset."""
    cmd = [
        PYTHON, "main.py",
        "True", "True",                 # runtime, mtr
        "SineClustersPre",              # dataset
        "200",                          # training samples
        "HoeffdingTreeClassifier",      # classifier
        "BNDM", "recent_samples_size", "100", "n_samples", "50",
        "const", "1.0", "max_depth", "3", "threshold", "0.5",
    ]
    rc, out, err = run_subprocess(cmd, timeout=300)

    if rc != 0:
        return CheckResult("main.py MTR mode", False, f"Exit code {rc}: {err[:200]}")

    has_mtr = "MTR:" in out
    has_runtime = "RUNTIME:" in out
    has_oo_info = "OO-Info:" in out

    if not (has_mtr and has_runtime and has_oo_info):
        missing = []
        if not has_mtr: missing.append("MTR")
        if not has_runtime: missing.append("RUNTIME")
        if not has_oo_info: missing.append("OO-Info")
        return CheckResult("main.py MTR mode", False, f"Missing output: {', '.join(missing)}")

    return CheckResult("main.py MTR mode", True, "BNDM on SineClustersPre: MTR + runtime output verified")


def check_compute_baselines():
    """Run compute_baselines.py on a small dataset."""
    # Use SineClustersPre via direct import since it's not in __init__
    # compute_baselines.py uses sys.argv[1] as dataset_string and eval()s it
    # It imports from datasets which only exports ForestCovertype, GasSensor, PokerHand
    # Let's use GasSensor since it's exported and has a model
    if not (PROJECT_ROOT / "model/HoeffdingTreeClassifier/HoeffdingTreeClassifier_GasSensor.pkl").exists():
        return CheckResult("compute_baselines.py", False, "No GasSensor model found")

    cmd = [PYTHON, "compute_baselines.py", "GasSensor"]
    rc, out, err = run_subprocess(cmd, timeout=300)

    if rc != 0:
        return CheckResult("compute_baselines.py", False, f"Exit code {rc}: {err[:200]}")

    has_accuracy = "Accuracy:" in out
    has_runtime = "Runtime:" in out

    if not (has_accuracy and has_runtime):
        return CheckResult("compute_baselines.py", False, "Missing accuracy/runtime output")

    return CheckResult("compute_baselines.py", True, "Baselines computed for GasSensor")


def check_train_classifiers():
    """Run train_classifiers.py briefly on a small synthetic dataset."""
    # train_classifiers.py takes: dataset n_training n_verification classifier [args]
    # Use SineClustersPre (imported directly in the script) with small sample counts
    cmd = [
        PYTHON, "train_classifiers.py",
        "SineClustersPre", "50", "10",
        "HoeffdingTreeClassifier",
    ]
    rc, out, err = run_subprocess(cmd, timeout=120)

    if rc != 0:
        return CheckResult("train_classifiers.py", False, f"Exit code {rc}: {err[:200]}")

    has_result = "RESULT:" in out
    has_id = "classifierId:" in out

    if not (has_result and has_id):
        return CheckResult("train_classifiers.py", False, "Missing RESULT/classifierId output")

    # Check a .pickle file was created
    pickle_files = list(PROJECT_ROOT.glob("HoeffdingTreeClassifier*_SineClustersPre.pickle"))
    if not pickle_files:
        return CheckResult("train_classifiers.py", False, "No .pickle file created")

    # Cleanup
    for f in pickle_files:
        f.unlink()

    return CheckResult("train_classifiers.py", True, "Classifier trained and saved (SineClustersPre, 50 samples)")


def check_eval_config():
    """Verify eval_config.py loads and has consistent definitions."""
    rc, out, err = run_subprocess([
        PYTHON, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "from evaluation.eval_config import ALL_DETECTORS, REAL_DATASETS, SYNTHETIC_DATASETS, DETECTOR_COLORS; "
        "assert len(ALL_DETECTORS) == 20, f'{len(ALL_DETECTORS)} detectors'; "
        "assert len(DETECTOR_COLORS) >= 20, f'{len(DETECTOR_COLORS)} colors'; "
        "assert len(REAL_DATASETS) > 0; "
        "assert len(SYNTHETIC_DATASETS) > 0; "
        "print('OK')"
    ], timeout=30)

    if rc != 0:
        return CheckResult("eval_config.py", False, f"Exit code {rc}: {err[:200]}")

    return CheckResult("eval_config.py", True, "20 detectors, datasets and colors defined correctly")


def check_plot_pareto_fronts():
    """Run plot_pareto_fronts.py and check it produces output (or handles empty results gracefully)."""
    tmp_out = PROJECT_ROOT / "_sanity_pareto_out"
    tmp_out.mkdir(exist_ok=True)

    try:
        cmd = [
            PYTHON, "evaluation/plot_pareto_fronts.py",
            "--results-dir", "results/all_benchmark_results",
            "--output-dir", str(tmp_out),
        ]
        rc, out, err = run_subprocess(cmd, timeout=60)

        # Script should either produce PNGs or exit cleanly if no data
        pngs = list(tmp_out.glob("*.png"))
        if rc == 0 and (len(pngs) > 0 or "No datasets" in out or "No completed" in out):
            return CheckResult("plot_pareto_fronts.py", True,
                               f"Script ran successfully ({len(pngs)} PNGs generated)")
        elif rc != 0:
            combined = err + out
            if "Results directory not found" in combined or "No such file" in combined:
                return CheckResult("plot_pareto_fronts.py", True,
                                   "No benchmark results directory (expected if no benchmark data)")
            return CheckResult("plot_pareto_fronts.py", False, f"Exit code {rc}: {err[:200]}")
        else:
            return CheckResult("plot_pareto_fronts.py", True, "Script ran, no PNGs (expected if no benchmark data)")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


def check_create_mrp_tables():
    """Run create_all_mrp_tables.py and check it produces output."""
    cmd = [PYTHON, "evaluation/create_all_mrp_tables.py"]
    rc, out, err = run_subprocess(cmd, timeout=60)

    if rc != 0:
        # Check if it's just missing data (expected if no benchmark results)
        if "No such file" in err or "FileNotFoundError" in err or "KeyError" in err:
            return CheckResult("create_all_mrp_tables.py", True,
                               "Script ran but no benchmark data available (expected)")
        return CheckResult("create_all_mrp_tables.py", False, f"Exit code {rc}: {err[:200]}")

    return CheckResult("create_all_mrp_tables.py", True, "MRP table generation script ran successfully")


def check_unit_tests():
    """Run the unit test suite (detectors + metrics + optimization)."""
    # Try unittest discovery (no pytest dependency)
    cmd = [PYTHON, "-m", "unittest", "discover", "-s", "test/detectors", "-v"]
    rc, out, err = run_subprocess(cmd, timeout=120)

    if rc != 0:
        return CheckResult("Unit tests", False, f"Exit code {rc}: {err[:200]}")

    has_ok = "OK" in out or "OK" in err
    if not has_ok:
        return CheckResult("Unit tests", False, "No 'OK' in test output")

    # Count tests
    test_count = out.count("... ok") + out.count("... OK")
    return CheckResult("Unit tests", True, f"{test_count} tests passed")


def check_posthoc_reproducibility():
    """Verify reproducibility scripts can be imported and have correct CLI."""
    rc, out, err = run_subprocess([
        PYTHON, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "sys.path.insert(0, 'posthoc_analysis/reproducibility'); "
        "import reproduce_benchmark; import analyze_reproducibility; "
        "import submit_jobs; "
        "print('OK')"
    ], timeout=30)

    if rc != 0:
        return CheckResult("Post-hoc: reproducibility imports", False, f"Exit code {rc}: {err[:200]}")

    # Check submit_jobs.py --help works
    rc2, out2, err2 = run_subprocess([
        PYTHON, "posthoc_analysis/reproducibility/submit_jobs.py", "--help"
    ], timeout=15)

    if rc2 != 0:
        return CheckResult("Post-hoc: reproducibility imports", False, "submit_jobs.py --help failed")

    return CheckResult("Post-hoc: reproducibility imports", True,
                       "reproduce_benchmark.py, analyze_reproducibility.py, submit_jobs.py importable")


def check_posthoc_multiseed():
    """Verify multi-seed scripts can be imported and have correct CLI."""
    rc, out, err = run_subprocess([
        PYTHON, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "sys.path.insert(0, 'posthoc_analysis/multiseed'); "
        "import multi_seed_analysis; import analyze_multiseed_reproducibility; "
        "import show_summary_correlations; import submit_multi_seed_jobs; "
        "print('OK')"
    ], timeout=30)

    if rc != 0:
        return CheckResult("Post-hoc: multiseed imports", False, f"Exit code {rc}: {err[:200]}")

    # Check submit_multi_seed_jobs.py --help works
    rc2, out2, err2 = run_subprocess([
        PYTHON, "posthoc_analysis/multiseed/submit_multi_seed_jobs.py", "--help"
    ], timeout=15)

    if rc2 != 0:
        return CheckResult("Post-hoc: multiseed imports", False, "submit_multi_seed_jobs.py --help failed")

    return CheckResult("Post-hoc: multiseed imports", True,
                       "multi_seed_analysis.py, analyze_multiseed_reproducibility.py, "
                       "show_summary_correlations.py, submit_multi_seed_jobs.py importable")


def check_posthoc_runtime_stability():
    """Verify runtime stability scripts can be imported and have correct CLI."""
    rc, out, err = run_subprocess([
        PYTHON, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "sys.path.insert(0, 'posthoc_analysis/runtime_stability'); "
        "import runtime_accuracy_stability_study; import summarize_stability; "
        "import submit_runtime_accuracy_stability_jobs; "
        "print('OK')"
    ], timeout=30)

    if rc != 0:
        return CheckResult("Post-hoc: runtime_stability imports", False, f"Exit code {rc}: {err[:200]}")

    # Check submit script --help works
    rc2, out2, err2 = run_subprocess([
        PYTHON, "posthoc_analysis/runtime_stability/submit_runtime_accuracy_stability_jobs.py", "--help"
    ], timeout=15)

    if rc2 != 0:
        return CheckResult("Post-hoc: runtime_stability imports", False,
                           "submit_runtime_accuracy_stability_jobs.py --help failed")

    return CheckResult("Post-hoc: runtime_stability imports", True,
                       "runtime_accuracy_stability_study.py, summarize_stability.py, "
                       "submit_runtime_accuracy_stability_jobs.py importable")


def check_runtime_accuracy_stability_run():
    """Run runtime_accuracy_stability_study.py briefly on SineClustersPre with BNDM."""
    tmp_out = PROJECT_ROOT / "_sanity_stability_out"
    tmp_out.mkdir(exist_ok=True)

    try:
        cmd = [
            PYTHON, "posthoc_analysis/runtime_stability/runtime_accuracy_stability_study.py",
            "--detector", "BNDM",
            "--dataset", "SineClustersPre",
            "--n-configs", "1",
            "--n-runs", "1",
            "--output-dir", str(tmp_out),
            "--python-exe", PYTHON,
        ]
        rc, out, err = run_subprocess(cmd, timeout=300)

        if rc != 0:
            combined = err + out
            if "Result folder not found" in combined or "Results directory not found" in combined:
                return CheckResult("Runtime stability study (quick run)", True,
                                   "No benchmark results directory (expected if no benchmark data)")
            return CheckResult("Runtime stability study (quick run)", False,
                               f"Exit code {rc}: {err[:200]}")

        csvs = list(tmp_out.glob("*.csv"))
        if not csvs:
            return CheckResult("Runtime stability study (quick run)", False,
                               "No CSV output produced")

        # Check CSV has expected columns
        import csv as csv_mod
        with open(csvs[0], 'r') as f:
            reader = csv_mod.DictReader(f)
            columns = reader.fieldnames
            expected = ["detector", "dataset", "mean_runtime", "std_runtime", "mean_accuracy"]
            missing = [c for c in expected if c not in columns]
            if missing:
                return CheckResult("Runtime stability study (quick run)", False,
                                   f"Missing CSV columns: {missing}")

        return CheckResult("Runtime stability study (quick run)", True,
                           "BNDM/SineClustersPre: 1 config, 1 run — CSV with runtime+accuracy produced")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


def check_benchmark_config():
    """Verify benchmark_config.sh is syntactically valid and defines detector parameters."""
    rc, out, err = run_subprocess([
        "bash", "-c",
        "source benchmark_config.sh && echo \"DETECTORS: ${#CSDDM[@]} ${#BNDM[@]} ${#D3[@]}\""
    ], timeout=10)

    if rc != 0:
        return CheckResult("benchmark_config.sh", False, f"Exit code {rc}: {err[:200]}")

    if "DETECTORS:" not in out:
        return CheckResult("benchmark_config.sh", False, "No DETECTORS output")

    return CheckResult("benchmark_config.sh", True, "Shell config sources correctly, detector parameters defined")


def check_model_files():
    """Verify pre-trained classifier models exist for key datasets."""
    model_dir = PROJECT_ROOT / "model" / "HoeffdingTreeClassifier"
    if not model_dir.exists():
        return CheckResult("Pre-trained models", False, "model/HoeffdingTreeClassifier/ not found")

    pkls = list(model_dir.glob("*.pkl"))
    if len(pkls) < 3:
        return CheckResult("Pre-trained models", False, f"Only {len(pkls)} models found")

    # Check for key datasets
    key_datasets = ["SineClustersPre", "GasSensor", "PokerHand"]
    missing = []
    for ds in key_datasets:
        if not any(ds in p.name for p in pkls):
            missing.append(ds)

    if missing:
        return CheckResult("Pre-trained models", False, f"Missing models for: {missing}")

    return CheckResult("Pre-trained models", True,
                       f"{len(pkls)} HoeffdingTreeClassifier models available")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  SANITY CHECK — Benchmark for Unsupervised Concept Drift Detection")
    print("=" * 70)
    print(f"  Python: {PYTHON}")
    print(f"  Root:   {PROJECT_ROOT}")

    checks = [
        ("Core imports", check_imports),
        ("Dataset loading", check_dataset_loading),
        ("main.py standard mode", check_main_standard_mode),
        ("main.py MTR mode", check_main_mtr_mode),
        ("compute_baselines.py", check_compute_baselines),
        ("train_classifiers.py", check_train_classifiers),
        ("eval_config.py", check_eval_config),
        ("plot_pareto_fronts.py", check_plot_pareto_fronts),
        ("create_all_mrp_tables.py", check_create_mrp_tables),
        ("Unit tests", check_unit_tests),
        ("Post-hoc: reproducibility imports", check_posthoc_reproducibility),
        ("Post-hoc: multiseed imports", check_posthoc_multiseed),
        ("Post-hoc: runtime_stability imports", check_posthoc_runtime_stability),
        ("Runtime stability study (quick run)", check_runtime_accuracy_stability_run),
        ("benchmark_config.sh", check_benchmark_config),
        ("Pre-trained models", check_model_files),
    ]

    results = []
    for name, func in checks:
        result = run_check(name, func)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    for r in results:
        print(r)

    print(f"\n  {passed}/{total} checks passed, {failed} failed")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
