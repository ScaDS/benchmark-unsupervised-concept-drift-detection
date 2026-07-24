#!/usr/bin/env python3
"""Reproduce all MRP runtime tables (B10^10, B10^6, B10^3, B10^1, B0) from
all_benchmark_results, compare against the published mtr_runtime.tex, and
optionally generate an updated tex file with new DDAL data.

Usage:
  python3 reproduce_all_mrp_tables.py [--update-ddal] [--output FILE]

Without --update-ddal: uses backup DDAL ForestCovertype data for verification.
With --update-ddal: uses new DDAL ForestCovertype data for final output.
"""
import csv
import math
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

DETECTORS = [
    "BNDM", "CDBD", "CDLEEDS", "CSDDM", "D3", "DAWIDD", "DDAL", "EDFS",
    "HDDDM", "IBDD", "IKS", "NNDVI", "OCDD", "PCACD", "SPLL", "STUDD",
    "SlidShaps", "UCDD", "UDetect", "WindowKDE",
]

DS_DIR = {
    "Pokerhand": "PokerHand",
    "Rialto": "RialtoBridgeTimelapse",
    "ForestCoverType": "ForestCovertype",
    "GasSensor": "GasSensor",
    "Sensorstream": "SensorStream",
}
LABELS = list(DS_DIR.keys())

# ── Baseline definitions ──────────────────────────────────────────────
# Alpha = target accuracy (rounded to 2 decimals, as shown in tex)
# Runtime = baseline's own runtime
# Relative = baseline runtime / median runtime (computed, not used directly)

BASELINES = {
    "B10^10": {
        "alpha": {"Pokerhand": 0.92, "Rialto": 0.72, "ForestCoverType": 0.79,
                  "GasSensor": 0.81, "Sensorstream": 0.38},
        "runtime": {"Pokerhand": 159, "Rialto": 60, "ForestCoverType": 460,
                    "GasSensor": 29, "Sensorstream": 549},
        "tex_label": r"$B_{10}^{10}$",
        "tex_name": "B10^10",
    },
    "B10^6": {
        "alpha": {"Pokerhand": 0.90, "Rialto": 0.51, "ForestCoverType": 0.77,
                  "GasSensor": 0.78, "Sensorstream": 0.30},
        "runtime": {"Pokerhand": 116, "Rialto": 33, "ForestCoverType": 278,
                    "GasSensor": 22, "Sensorstream": 488},
        "tex_label": r"$B_{10}^{6}$",
        "tex_name": "B10^6",
    },
    "B10^3": {
        "alpha": {"Pokerhand": 0.89, "Rialto": 0.35, "ForestCoverType": 0.73,
                  "GasSensor": 0.74, "Sensorstream": 0.18},
        "runtime": {"Pokerhand": 65, "Rialto": 16, "ForestCoverType": 150,
                    "GasSensor": 15, "Sensorstream": 556},
        "tex_label": r"$B_{10}^{3}$",
        "tex_name": "B10^3",
    },
    "B10^1": {
        "alpha": {"Pokerhand": 0.84, "Rialto": 0.29, "ForestCoverType": 0.69,
                  "GasSensor": 0.66, "Sensorstream": 0.07},
        "runtime": {"Pokerhand": 39, "Rialto": 9, "ForestCoverType": 71,
                    "GasSensor": 10, "Sensorstream": 717},
        "tex_label": r"$B_{10}^{1}$",
        "tex_name": "B10^1",
    },
    "B0": {
        "alpha": {"Pokerhand": 0.19, "Rialto": 0.32, "ForestCoverType": 0.58,
                  "GasSensor": 0.52, "Sensorstream": 0.06},
        "runtime": {"Pokerhand": 17, "Rialto": 7, "ForestCoverType": 25,
                    "GasSensor": 5, "Sensorstream": 88},
        "tex_label": r"$B0$",
        "tex_name": "B0",
    },
}

# ── Italic (timeout) markings from original tex ───────────────────────
# True = italic (timeout setup) for that detector/dataset/baseline

TIMEOUT = {
    "B10^10": {
        ("CDLEEDS", "ForestCoverType"): True,
        ("EDFS", "ForestCoverType"): True,
        ("NNDVI", "GasSensor"): True,
        ("UCDD", "GasSensor"): True,
    },
    "B10^6": {
        ("CDLEEDS", "ForestCoverType"): True,
        ("EDFS", "ForestCoverType"): True,
        ("NNDVI", "Rialto"): True,
        ("NNDVI", "GasSensor"): True,
        ("UCDD", "GasSensor"): True,
    },
    "B10^3": {
        ("CDLEEDS", "ForestCoverType"): True,
        ("EDFS", "ForestCoverType"): True,
        ("NNDVI", "Rialto"): True,
        ("NNDVI", "GasSensor"): True,
        ("UCDD", "GasSensor"): True,
    },
    "B10^1": {
        ("CDLEEDS", "ForestCoverType"): True,
        ("EDFS", "ForestCoverType"): True,
        ("IKS", "Sensorstream"): True,
        ("NNDVI", "Rialto"): True,
        ("NNDVI", "GasSensor"): True,
        ("SlidShaps", "Rialto"): True,
        ("UCDD", "Rialto"): True,
        ("UCDD", "GasSensor"): True,
    },
    "B0": {
        ("CDLEEDS", "ForestCoverType"): True,
        ("CDLEEDS", "Sensorstream"): True,
        ("EDFS", "ForestCoverType"): True,
        ("IKS", "Sensorstream"): True,
        ("NNDVI", "Rialto"): True,
        ("NNDVI", "GasSensor"): True,
        ("SlidShaps", "Pokerhand"): True,
        ("SlidShaps", "Rialto"): True,
        ("SlidShaps", "GasSensor"): True,
        ("UCDD", "Rialto"): True,
        ("UCDD", "GasSensor"): True,
    },
}

# ── Reference MRP values from published tex ───────────────────────────

REF_MRP = {
    "B10^10": {
        "BNDM":      [0.72, None, 0.72, 0.11, 0.86],
        "CDBD":      [None, None, None, None, None],
        "CDLEEDS":   [None, 1.69, 13.16, 0.32, None],
        "CSDDM":     [1.08, None, 0.91, None, 1.40],
        "D3":        [0.58, None, 0.62, 1.32, 0.85],
        "DAWIDD":    [None, None, None, None, None],
        "DDAL":      [None, None, None, None, None],
        "EDFS":      [0.85, None, 0.82, None, 1.12],
        "HDDDM":     [2.65, None, 1.47, None, None],
        "IBDD":      [1.09, None, 1.18, None, 1.99],
        "IKS":       [None, None, None, None, None],
        "NNDVI":     [None, None, None, 1.00, None],
        "OCDD":      [1.00, 0.31, 0.89, 0.07, 1.30],
        "PCACD":     [None, None, 3.35, None, None],
        "SPLL":      [1.03, None, 1.06, 1.02, 0.88],
        "STUDD":     [None, None, None, None, None],
        "SlidShaps": [None, None, None, None, None],
        "UCDD":      [None, None, None, 9.21, None],
        "UDetect":   [0.70, None, 1.00, None, 0.87],
        "WindowKDE": [None, None, None, None, None],
    },
    "B10^6": {
        "BNDM":      [0.43, 0.80, 0.87, 0.11, 0.74],
        "CDBD":      [None, None, None, None, None],
        "CDLEEDS":   [None, 15.95, 16.88, 0.63, None],
        "CSDDM":     [0.84, 1.05, 0.78, 0.44, 1.34],
        "D3":        [0.41, 0.62, 0.51, 0.55, 0.77],
        "DAWIDD":    [1.35, 0.85, 5.94, None, 3.30],
        "DDAL":      [None, 0.72, None, None, None],
        "EDFS":      [0.51, 0.85, 0.90, None, 1.00],
        "HDDDM":     [2.91, 1.10, 1.89, None, None],
        "IBDD":      [1.02, 1.00, 1.10, 4.00, 1.36],
        "IKS":       [None, None, None, None, None],
        "NNDVI":     [None, 35.55, None, 1.99, None],
        "OCDD":      [0.98, 1.05, 0.81, 0.13, 1.39],
        "PCACD":     [3.92, 2.55, 4.30, 2.05, None],
        "SPLL":      [1.12, 1.00, 1.17, 1.37, 0.84],
        "STUDD":     [1.23, 2.08, None, None, None],
        "SlidShaps": [None, None, None, None, None],
        "UCDD":      [None, None, None, 18.34, None],
        "UDetect":   [0.41, 0.75, 0.68, None, 0.74],
        "WindowKDE": [None, None, None, None, None],
    },
    "B10^3": {
        "BNDM":      [0.43, 0.76, 0.44, 0.17, 0.55],
        "CDBD":      [None, None, None, None, None],
        "CDLEEDS":   [None, 37.53, 26.73, 1.08, None],
        "CSDDM":     [0.96, 1.41, 0.67, 0.66, 1.12],
        "D3":        [0.37, 0.59, 0.43, 0.50, 0.34],
        "DAWIDD":    [1.58, 0.41, 2.55, 24.61, 1.00],
        "DDAL":      [None, 0.88, None, None, None],
        "EDFS":      [0.60, 0.94, 0.82, 0.79, 0.75],
        "HDDDM":     [3.39, 0.65, 2.99, 0.92, None],
        "IBDD":      [1.07, 1.00, 1.00, 1.68, 1.21],
        "IKS":       [None, None, None, None, None],
        "NNDVI":     [None, 83.65, None, 1.88, None],
        "OCDD":      [1.02, 1.71, 0.70, 0.16, 1.24],
        "PCACD":     [4.07, 4.00, 2.18, 3.49, 3.36],
        "SPLL":      [1.31, 1.59, 1.19, 0.37, 0.61],
        "STUDD":     [0.98, 2.76, 1.65, None, 2.62],
        "SlidShaps": [None, None, None, None, None],
        "UCDD":      [None, None, None, 15.37, None],
        "UDetect":   [0.43, 0.76, 0.44, 1.20, 0.55],
        "WindowKDE": [None, None, None, None, None],
    },
    "B10^1": {
        "BNDM":      [0.38, 0.60, 0.39, 0.84, 0.72],
        "CDBD":      [None, None, None, None, 0.98],
        "CDLEEDS":   [None, 4.00, 46.44, 5.72, None],
        "CSDDM":     [1.10, 1.33, 0.73, 0.72, 1.53],
        "D3":        [0.39, 0.53, 0.39, 0.44, 0.35],
        "DAWIDD":    [1.73, 0.47, 1.53, 18.28, 1.00],
        "DDAL":      [None, 0.67, 0.38, None, 1.17],
        "EDFS":      [0.46, 1.00, 1.42, 1.00, 1.08],
        "HDDDM":     [1.34, 0.47, 0.45, 0.76, 0.18],
        "IBDD":      [0.90, 0.73, 1.16, 0.92, 0.50],
        "IKS":       [None, None, None, None, 1.33],
        "NNDVI":     [None, 43.13, None, 9.96, None],
        "OCDD":      [1.20, 1.60, 0.84, 0.44, 1.74],
        "PCACD":     [5.09, 4.13, 3.78, 1.04, 3.61],
        "SPLL":      [1.69, 1.80, 2.07, 0.76, 0.86],
        "STUDD":     [0.84, 0.93, 1.63, 2.16, 1.57],
        "SlidShaps": [None, 134.73, None, None, None],
        "UCDD":      [None, 166.60, None, 81.48, None],
        "UDetect":   [0.42, 0.47, 0.35, 1.08, 0.79],
        "WindowKDE": [None, None, None, None, None],
    },
    "B0": {
        "BNDM":      [0.30, 0.56, 0.30, 0.67, 0.93],
        "CDBD":      [2.60, None, 9.66, None, 1.27],
        "CDLEEDS":   [None, 3.75, 5.76, 23.83, 5.12],
        "CSDDM":     [2.06, 1.25, 1.34, 1.00, 1.98],
        "D3":        [0.39, 0.50, 0.42, 0.83, 0.34],
        "DAWIDD":    [0.27, 0.44, 0.31, 0.67, 0.24],
        "DDAL":      [0.32, 0.62, 0.33, 1.00, 0.36],
        "EDFS":      [1.00, 1.00, 1.24, 0.67, 1.39],
        "HDDDM":     [0.27, 0.44, 0.33, 0.67, 0.23],
        "IBDD":      [0.79, 0.69, 0.76, 0.83, 0.25],
        "IKS":       [4.78, None, 14.73, None, 1.71],
        "NNDVI":     [None, 40.44, None, 3.00, None],
        "OCDD":      [2.45, 1.56, 1.52, 1.17, 2.26],
        "PCACD":     [7.71, 3.88, 4.64, 1.67, 4.59],
        "SPLL":      [3.56, 1.69, 5.44, 0.67, 1.07],
        "STUDD":     [0.53, 0.88, 0.47, 1.33, 0.64],
        "SlidShaps": [18.86, 126.31, None, 2.50, None],
        "UCDD":      [None, 156.19, None, 4.33, None],
        "UDetect":   [0.25, 0.44, 0.28, 0.67, 0.24],
        "WindowKDE": [33.92, None, None, None, None],
    },
}

REF_MEDIAN = {
    "B10^10": {"Pokerhand": 316, "Rialto": 378, "ForestCoverType": 900,
               "GasSensor": 449, "Sensorstream": 798},
    "B10^6": {"Pokerhand": 288, "Rialto": 40, "ForestCoverType": 702,
              "GasSensor": 226, "Sensorstream": 646},
    "B10^3": {"Pokerhand": 247, "Rialto": 17, "ForestCoverType": 443,
              "GasSensor": 132, "Sensorstream": 674},
    "B10^1": {"Pokerhand": 185, "Rialto": 15, "ForestCoverType": 255,
              "GasSensor": 25, "Sensorstream": 470},
    "B0": {"Pokerhand": 77, "Rialto": 16, "ForestCoverType": 97,
           "GasSensor": 6, "Sensorstream": 364},
}


# ── Data loading ──────────────────────────────────────────────────────

def load_standard_csv(path, alpha):
    """Load RUNTIMEs from standard CSV (Status/ACCURACY/RUNTIME columns).
    Accuracy is rounded to 2 decimals before comparison with alpha."""
    vals = []
    if not path.exists():
        return vals
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("Status", row.get("trial_status", "")).strip().upper()
            if status != "COMPLETED":
                continue
            try:
                a = float(row["ACCURACY"])
                rt = float(row["RUNTIME"])
            except (ValueError, KeyError):
                continue
            if round(a, 2) >= alpha:
                vals.append(rt)
    return vals


def min_runtime(det, label, alpha, ddal_source="current"):
    """Compute minimum runtime for a detector/dataset at the target accuracy."""
    ds = DS_DIR[label]
    vals = []

    if det == "DDAL" and ds == "ForestCovertype" and ddal_source == "backup":
        # Use DDAL_old data for reproduction verification
        ddal_old = BASE / "DDAL_old" / ds
        for suffix in ["ACC_RT", "ACC_RT_REQL"]:
            p = ddal_old / f"DDAL_{ds}_{suffix}.csv"
            vals += load_standard_csv(p, alpha)
    else:
        base_dir = BASE / "results" / "all_benchmark_results" / det / ds
        for suffix in ["ACC_RT", "ACC_RT_REQL"]:
            p = base_dir / f"{det}_{ds}_{suffix}.csv"
            vals += load_standard_csv(p, alpha)

    return min(vals) if vals else None


def compute_table(baseline_name, ddal_source="current"):
    """Compute the full MRP table for a given baseline."""
    bl = BASELINES[baseline_name]
    alpha = bl["alpha"]
    timeout = TIMEOUT.get(baseline_name, {})

    r = {det: {} for det in DETECTORS}
    for det in DETECTORS:
        for label in LABELS:
            r[det][label] = min_runtime(det, label, alpha[label], ddal_source)

    med = {}
    for label in LABELS:
        vals = [r[det][label] for det in DETECTORS if r[det][label] is not None]
        med[label] = statistics.median(vals) if vals else None

    mrp = {det: {} for det in DETECTORS}
    for det in DETECTORS:
        for label in LABELS:
            rv = r[det][label]
            mrp[det][label] = rv / med[label] if rv is not None and med[label] else None

    # MMRP, MAD, #Gains
    stats = {}
    for det in DETECTORS:
        all_vals = [mrp[det][l] for l in LABELS if mrp[det][l] is not None]
        non_italic_vals = [mrp[det][l] for l in LABELS
                           if mrp[det][l] is not None
                           and not timeout.get((det, l), False)]
        italic_vals = [mrp[det][l] for l in LABELS
                       if mrp[det][l] is not None
                       and timeout.get((det, l), False)]

        if not all_vals:
            stats[det] = {"mmrp": None, "mmrp_with_timeout": None,
                          "mad": None, "gains": 0, "all_italic": False}
            continue

        all_italic = len(non_italic_vals) == 0

        if all_italic:
            mmrp_main = sum(all_vals) / len(all_vals)
            mmrp_timeout = None  # no parenthetical
            mad = 0.0
        else:
            mmrp_main = sum(non_italic_vals) / len(non_italic_vals)
            mmrp_timeout = sum(all_vals) / len(all_vals)
            if len(non_italic_vals) >= 2:
                mad = statistics.stdev(non_italic_vals)
            else:
                mad = 0.0

        gains = sum(1 for l in LABELS if r[det][l] is not None)

        stats[det] = {
            "mmrp": mmrp_main,
            "mmrp_with_timeout": mmrp_timeout,
            "mad": mad,
            "gains": gains,
            "all_italic": all_italic,
        }

    return r, med, mrp, stats


# ── Verification ──────────────────────────────────────────────────────

def verify_table(baseline_name, r, med, mrp, stats, tol=0.03):
    """Compare computed table against reference."""
    ref_mrp = REF_MRP[baseline_name]
    ref_med = REF_MEDIAN[baseline_name]

    matches = 0
    total = 0
    mismatches = []

    for det in DETECTORS:
        for i, label in enumerate(LABELS):
            ref = ref_mrp[det][i]
            comp = mrp[det][label]
            if ref is None and comp is None:
                continue
            total += 1
            if comp is None or ref is None:
                mismatches.append((det, label, comp, ref))
            elif abs(comp - ref) <= tol:
                matches += 1
            else:
                mismatches.append((det, label, comp, ref))

    # Check medians
    med_matches = True
    for label in LABELS:
        if med[label] is not None and ref_med[label] is not None:
            if abs(med[label] - ref_med[label]) > 0.5:
                med_matches = False
                mismatches.append(("MEDIAN", label, med[label], ref_med[label]))

    return matches, total, mismatches, med_matches


# ── LaTeX generation ──────────────────────────────────────────────────

GLOBAL_MIN = 0.07


def black_shade(mrp, table_min=GLOBAL_MIN, table_max_italic=None, is_italic=False):
    """Map MRP to black!X grayscale. Lower MRP (better) -> darker.
    Uses per-table minimum for normalization of values <= 1.0.
    For italic values > 1.0, uses per-table max_italic for decay."""
    if mrp is None:
        return None
    if mrp <= 1.0:
        return max(0, min(50, int(50 - (mrp - table_min) / (1.0 - table_min) * 35)))
    if is_italic and table_max_italic is not None and table_max_italic > 1.0:
        return max(0, int(14 - (mrp - 1) / (table_max_italic - 1) * 14))
    return 14


def cyan_shade(mmrp, table_min=GLOBAL_MIN):
    """Map MMRP to cyan!X shade. Lower = better -> darker.
    Uses per-table minimum for normalization of values <= 1.0."""
    if mmrp is None:
        return None
    if mmrp <= 1.0:
        return max(0, min(50, int(50 - (mmrp - table_min) / (1.0 - table_min) * 35)))
    return 14


def fmt_mrp(val, is_italic=False):
    if val is None:
        return "--"
    s = f"{val:.2f}"
    if is_italic:
        s = f"\\textit{{{s}}}"
    return s


def fmt_mmrp(stats):
    """Format MMRP cell with optional parenthetical and italic."""
    mmrp = stats["mmrp"]
    mmrp_to = stats["mmrp_with_timeout"]
    all_italic = stats["all_italic"]

    if mmrp is None:
        return "--"

    if all_italic:
        return f"\\textit{{{mmrp:.2f}}}"
    if mmrp_to is not None and abs(mmrp - mmrp_to) > 0.005:
        return f"{mmrp:.2f} ({mmrp_to:.2f})"
    return f"{mmrp:.2f}"


def generate_sideways_table(baseline_name, r, med, mrp, stats, is_first=True):
    """Generate a sideways table* block (for B10^10, B10^6, B10^3, B10^1)."""
    bl = BASELINES[baseline_name]
    timeout = TIMEOUT.get(baseline_name, {})
    bname = bl["tex_name"]

    lines = []
    if is_first:
        lines.append(r"\begin{sidewaystable*}[!htbp]")
        lines.append(r"\renewcommand{\arraystretch}{1.2}")
        lines.append(r"    \centering")
        lines.append(r"        \caption{MRP with respect to runtime for target accuracies of baselines $B_{10}^{10}$ and $B_{10}^{6}$. A $-$ in the table indicates that the target accuracy was not met by the \gls{dd}.")
        lines.append(r"        For ease of readability, absolute runtimes in seconds are skipped for the \glspl{dd}. Italic values are values where single solutions were provided, but the setup was too slow to finish the whole 500 evaluations (timeout setups). The values in parentheses in the MMRP column are with timeout setups.")
        lines.append(r"        MRP and MMRP cell values are color-coded as a heat map (grey-scale for individual datasets, blue scale for MMRP), where darker shades indicate better performance.")
        lines.append(r"        \#Gains counts the number of datasets on which an improvement over the baseline was achieved.}")
        lines.append(r"    \label{tab:runtime_overview_bl1}")
        lines.append(r"\vspace{0.2cm}")

    # Data table
    col_spec = ">{\centering\arraybackslash}p{0.64cm}" * 5 + \
               ">{\centering\arraybackslash}p{1.85cm}" + \
               ">{\centering\arraybackslash}p{1.4cm}" + \
               ">{\centering\arraybackslash}p{0.8cm}"
    lines.append(r"\begin{tabular}[b]{" + col_spec + "}")
    lines.append(r"\rotatebox{20}{Pokerhand}&\rotatebox{20}{Rialto}&\rotatebox{20}{ForestCoverType}&\rotatebox{20}{GasSensor}&\rotatebox{20}{Sensorstream}&\rotatebox{20}{MMRP}&\rotatebox{20}{MAD} & \rotatebox{20}{\#Gains}\\\hline")

    for det in DETECTORS:
        cells = []
        for label in LABELS:
            v = mrp[det][label]
            is_it = timeout.get((det, label), False)
            cell_text = fmt_mrp(v, is_it)
            if v is not None:
                shade = black_shade(v)
                if shade is not None:
                    cells.append(f"\\cellcolor{{black!{shade}}} {cell_text}")
                else:
                    cells.append(cell_text)
            else:
                cells.append("--")
        # MMRP
        mmrp_str = fmt_mmrp(stats[det])
        mmrp_val = stats[det]["mmrp"]
        if mmrp_val is not None and not stats[det]["all_italic"]:
            cshade = cyan_shade(mmrp_val)
            if cshade is not None:
                cells.append(f"\\cellcolor{{cyan!{cshade}}} {mmrp_str}")
            else:
                cells.append(mmrp_str)
        elif stats[det]["all_italic"]:
            cells.append(mmrp_str)
        else:
            cells.append("--")
        # MAD
        mad = stats[det]["mad"]
        cells.append(f"{mad:.2f}" if mad is not None else "--")
        # #Gains
        cells.append(str(stats[det]["gains"]))
        lines.append("    " + " & ".join(cells) + r" \\")

    # Bottom rows
    lines.append(r"    \hline")
    med_cells = [f"{int(round(med[l]))}" if med[l] is not None else "--" for l in LABELS]
    lines.append("    " + " & ".join(med_cells) + r" & \multicolumn{3}{l}{Median Runtime} \\\hline\hline")

    bl_rt = bl["runtime"]
    rel_cells = []
    rt_cells = []
    alpha_cells = []
    for l in LABELS:
        rt = bl_rt[l]
        m = med[l]
        rel_cells.append(f"{rt / m:.2f}" if m else "--")
        rt_cells.append(str(rt))
        alpha_cells.append(f"{bl['alpha'][l]:.2f}")

    lines.append("    " + " & ".join(rel_cells) + f" & \\multicolumn{{3}}{{l}}{{ {bl['tex_label']} Relative}} \\\\")
    lines.append("    " + " & ".join(rt_cells) + f" & \\multicolumn{{3}}{{l}}{{ {bl['tex_label']} Runtime}} \\\\")
    lines.append("    " + " & ".join(alpha_cells) + f" & \\multicolumn{{3}}{{l}}{{ {bl['tex_label']} Accuracy}} \\\\")
    lines.append(r"    \hline")
    lines.append(r"    \end{tabular}")

    return lines


def generate_detector_column():
    """Generate the detector name column (shared between paired tables)."""
    lines = [
        r"\hfill",
        r"\begin{tabular}[b]{c}",
        r"\textbf{Drift Detector}\\\hline",
    ]
    for i, det in enumerate(DETECTORS):
        if det == "WindowKDE":
            lines.append(rf"\textbf{{{det}}}\\\hline")
        else:
            lines.append(rf"\textbf{{{det}}} \\")
    lines.append(r"\textbf{Runtime Med}\\\hline\hline")
    lines.append(r"\multirow{3}{*}{\textbf{Baseline}}\\")
    lines.append(r"\\")
    lines.append(r"\\\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\hfill")
    return lines


def generate_b0_table(r, med, mrp, stats):
    """Generate the B0 table* block."""
    timeout = TIMEOUT.get("B0", {})

    # Compute table min and max_italic for shading normalization
    all_mrp_vals = [mrp[det][l] for det in DETECTORS for l in LABELS if mrp[det][l] is not None]
    table_min = min(all_mrp_vals) if all_mrp_vals else GLOBAL_MIN
    italic_vals = [mrp[det][l] for det in DETECTORS for l in LABELS
                   if mrp[det][l] is not None and timeout.get((det, l), False) and mrp[det][l] > 1.0]
    table_max_italic = max(italic_vals) if italic_vals else None

    lines = [
        r"\begin{table*}[!htbp]",
        r"\renewcommand{\arraystretch}{1.2}",
        r"    \centering",
        r"        \caption*{\textbf{cont. \autoref{tab:runtime_overview_bl1}:} MRP with respect to runtime for target accuracies of baseline $B0$. A $-$ in the table indicates that the target accuracy was not met by the \gls{dd}. For ease of readability, absolute runtimes in seconds are skipped for the \glspl{dd}. Italic values are values where single solutions were provided, but the setup was too slow to finish the whole 500 evaluations (timeout setups). The values in parentheses in the MMRP column are with timeout setups. MRP and MMRP cell values are color-coded as a heat map (grey-scale for individual datasets, blue scale for MMRP), where darker shades indicate better performance. \#Gains counts the number of datasets on which an improvement over the baseline was achieved.}",
        r"    \label{tab:runtime_overview_bl0}",
        r"    \begin{tabular}[b]{>{\centering\arraybackslash}p{2.4cm}|>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{1.8cm}>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{1cm}}",
        r"& \rotatebox{20}{Pokerhand} & \rotatebox{20}{Rialto} & \rotatebox{20}{ForestCoverType} & \rotatebox{20}{GasSensor} & \rotatebox{20}{Sensorstream} & \rotatebox{20}{MMRP} & \rotatebox{20}{MAD} & \rotatebox{20}{\#Gains}\\ \hline",
    ]

    for det in DETECTORS:
        cells = [det]
        for label in LABELS:
            v = mrp[det][label]
            is_it = timeout.get((det, label), False)
            cell_text = fmt_mrp(v, is_it)
            if v is not None:
                shade = black_shade(v, table_min, table_max_italic, is_it)
                if shade is not None:
                    cells.append(f"\\cellcolor{{black!{shade}}} {cell_text}")
                else:
                    cells.append(cell_text)
            else:
                cells.append("--")
        # MMRP
        mmrp_str = fmt_mmrp(stats[det])
        mmrp_val = stats[det]["mmrp"]
        if mmrp_val is not None and not stats[det]["all_italic"]:
            cshade = cyan_shade(mmrp_val, table_min)
            if cshade is not None:
                cells.append(f"\\cellcolor{{cyan!{cshade}}} {mmrp_str}")
            else:
                cells.append(mmrp_str)
        elif stats[det]["all_italic"]:
            cells.append(mmrp_str)
        else:
            cells.append("--")
        mad = stats[det]["mad"]
        cells.append(f"{mad:.2f}" if mad is not None else "--")
        cells.append(str(stats[det]["gains"]))
        lines.append("    " + " & ".join(cells) + r" \\")

    # Bottom rows
    lines.append(r"    \hline")
    med_cells = [f"{int(round(med[l]))}" if med[l] is not None else "--" for l in LABELS]
    lines.append("    & " + " & ".join(med_cells) + r" & \multicolumn{3}{l}{Median Runtime} \\")

    bl = BASELINES["B0"]
    bl_rt = bl["runtime"]
    rel_cells = []
    rt_cells = []
    alpha_cells = []
    for l in LABELS:
        rt = bl_rt[l]
        m = med[l]
        rel_cells.append(f"{rt / m:.2f}" if m else "--")
        rt_cells.append(str(rt))
        alpha_cells.append(f"{bl['alpha'][l]:.2f}")
    lines.append("    & " + " & ".join(rel_cells) + r" & \multicolumn{3}{l}{B0 Relative} \\")
    lines.append("    & " + " & ".join(rt_cells) + r" & \multicolumn{3}{l}{B0 Runtime} \\")
    lines.append("    & " + " & ".join(alpha_cells) + r" & \multicolumn{3}{l}{B0 Accuracy} \\")
    lines.append(r"    \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")
    return lines


def generate_full_tex(ddal_source="current"):
    """Generate the complete mtr_runtime.tex file."""
    all_lines = []

    # First sidewaysstable: B10^10 (left) + detector column + B10^6 (right)
    r1, med1, mrp1, stats1 = compute_table("B10^10", ddal_source)
    r2, med2, mrp2, stats2 = compute_table("B10^6", ddal_source)

    # Start sidewaysstable
    all_lines.append(r"\begin{sidewaystable*}[!htbp]")
    all_lines.append(r"\renewcommand{\arraystretch}{1.2}")
    all_lines.append(r"    \centering")
    all_lines.append(r"        \caption{MRP with respect to runtime for target accuracies of baselines $B_{10}^{10}$ and $B_{10}^{6}$. A $-$ in the table indicates that the target accuracy was not met by the \gls{dd}.")
    all_lines.append(r"        For ease of readability, absolute runtimes in seconds are skipped for the \glspl{dd}. Italic values are values where single solutions were provided, but the setup was too slow to finish the whole 500 evaluations (timeout setups). The values in parentheses in the MMRP column are with timeout setups.")
    all_lines.append(r"        MRP and MMRP cell values are color-coded as a heat map (grey-scale for individual datasets, blue scale for MMRP), where darker shades indicate better performance.")
    all_lines.append(r"        \#Gains counts the number of datasets on which an improvement over the baseline was achieved.}")
    all_lines.append(r"    \label{tab:runtime_overview_bl1}")
    all_lines.append(r"\vspace{0.2cm}")

    # B10^10 table
    all_lines.extend(generate_sideways_table_data("B10^10", mrp1, med1, stats1))
    # Detector column (includes \hfill before and after)
    all_lines.extend(generate_detector_column())
    # B10^6 table
    all_lines.extend(generate_sideways_table_data("B10^6", mrp2, med2, stats2))
    all_lines.append("")
    all_lines.append("")
    all_lines.append(r"\end{sidewaystable*}")
    all_lines.append("")
    all_lines.append("")

    # Second sidewaysstable: B10^3 (left) + detector column + B10^1 (right)
    r3, med3, mrp3, stats3 = compute_table("B10^3", ddal_source)
    r4, med4, mrp4, stats4 = compute_table("B10^1", ddal_source)

    all_lines.append(r"\begin{sidewaystable*}[!htbp]")
    all_lines.append(r"\renewcommand{\arraystretch}{1.2}")
    all_lines.append(r"    \centering")
    all_lines.append(r"        \caption*{\textbf{cont. \autoref{tab:runtime_overview_bl1}:} MRP with respect to runtime for target accuracies of baselines $B_{10}^{3}$ and $B_{10}^{1}$. A $-$ in the table indicates that the target accuracy was not met by the \gls{dd}.")
    all_lines.append(r"        For ease of readability, absolute runtimes in seconds are skipped for the \glspl{dd}.")
    all_lines.append(r"        Italic values are values where single solutions were provided, but the setup was too slow to finish the whole 500 evaluations (timeout setups). The values in parentheses in the MMRP column are with timeout setups.")
    all_lines.append(r"        MRP and MMRP cell values are color-coded as a heat map (grey-scale for individual datasets, blue scale for MMRP), where darker shades indicate better performance.")
    all_lines.append(r"        \#Gains counts the number of datasets on which an improvement over the baseline was achieved.}")
    all_lines.append(r"    ")
    all_lines.append(r"\vspace{0.2cm}")

    all_lines.extend(generate_sideways_table_data("B10^3", mrp3, med3, stats3))
    all_lines.extend(generate_detector_column())
    all_lines.extend(generate_sideways_table_data("B10^1", mrp4, med4, stats4))
    all_lines.append(r"\end{sidewaystable*}")
    all_lines.append("")

    # B0 table
    r0, med0, mrp0, stats0 = compute_table("B0", ddal_source)
    all_lines.extend(generate_b0_table(r0, med0, mrp0, stats0))

    return "\n".join(all_lines)


def generate_sideways_table_data(baseline_name, mrp, med, stats):
    """Generate just the data tabular block for a sideways table."""
    bl = BASELINES[baseline_name]
    timeout = TIMEOUT.get(baseline_name, {})

    # Compute table min and max_italic for shading normalization
    all_mrp_vals = [mrp[det][l] for det in DETECTORS for l in LABELS if mrp[det][l] is not None]
    table_min = min(all_mrp_vals) if all_mrp_vals else GLOBAL_MIN
    italic_vals = [mrp[det][l] for det in DETECTORS for l in LABELS
                   if mrp[det][l] is not None and timeout.get((det, l), False) and mrp[det][l] > 1.0]
    table_max_italic = max(italic_vals) if italic_vals else None

    col_spec = ">{\centering\\arraybackslash}p{0.64cm}" * 5 + \
               ">{\centering\\arraybackslash}p{1.85cm}" + \
               ">{\centering\\arraybackslash}p{1.4cm}" + \
               ">{\centering\\arraybackslash}p{0.8cm}"
    lines = [
        r"\begin{tabular}[b]{" + col_spec + "}",
        r"\rotatebox{20}{Pokerhand}&\rotatebox{20}{Rialto}&\rotatebox{20}{ForestCoverType}&\rotatebox{20}{GasSensor}&\rotatebox{20}{Sensorstream}&\rotatebox{20}{MMRP}&\rotatebox{20}{MAD} & \rotatebox{20}{\#Gains}\\\hline",
    ]

    for det in DETECTORS:
        cells = []
        for label in LABELS:
            v = mrp[det][label]
            is_it = timeout.get((det, label), False)
            cell_text = fmt_mrp(v, is_it)
            if v is not None:
                shade = black_shade(v, table_min, table_max_italic, is_it)
                if shade is not None:
                    cells.append(f"\\cellcolor{{black!{shade}}} {cell_text}")
                else:
                    cells.append(cell_text)
            else:
                cells.append("--")
        mmrp_str = fmt_mmrp(stats[det])
        mmrp_val = stats[det]["mmrp"]
        if mmrp_val is not None and not stats[det]["all_italic"]:
            cshade = cyan_shade(mmrp_val, table_min)
            if cshade is not None:
                cells.append(f"\\cellcolor{{cyan!{cshade}}} {mmrp_str}")
            else:
                cells.append(mmrp_str)
        elif stats[det]["all_italic"]:
            cells.append(mmrp_str)
        else:
            cells.append("--")
        mad = stats[det]["mad"]
        cells.append(f"{mad:.2f}" if mad is not None else "--")
        cells.append(str(stats[det]["gains"]))
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \hline")
    med_cells = [f"{int(round(med[l]))}" if med[l] is not None else "--" for l in LABELS]
    lines.append("    " + " & ".join(med_cells) + r" & \multicolumn{3}{l}{Median Runtime} \\\hline\hline")

    bl_rt = bl["runtime"]
    rel_cells = []
    rt_cells = []
    alpha_cells = []
    for l in LABELS:
        rt = bl_rt[l]
        m = med[l]
        rel_cells.append(f"{rt / m:.2f}" if m else "--")
        rt_cells.append(str(rt))
        alpha_cells.append(f"{bl['alpha'][l]:.2f}")
    lines.append("    " + " & ".join(rel_cells) + f" & \\multicolumn{{3}}{{l}}{{ {bl['tex_label']} Relative}} \\\\")
    lines.append("    " + " & ".join(rt_cells) + f" & \\multicolumn{{3}}{{l}}{{ {bl['tex_label']} Runtime}} \\\\")
    lines.append("    " + " & ".join(alpha_cells) + f" & \\multicolumn{{3}}{{l}}{{ {bl['tex_label']} Accuracy}} \\\\")
    lines.append(r"    \hline")
    lines.append(r"    \end{tabular}")

    return lines


# ── Main ──────────────────────────────────────────────────────────────

def main():
    update_ddal = "--update-ddal" in sys.argv
    output_file = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    ddal_source = "current" if update_ddal else "backup"

    print("=" * 70)
    print(f"DDAL ForestCovertype source: {ddal_source}")
    print("=" * 70)

    all_match = True
    for bl_name in ["B10^10", "B10^6", "B10^3", "B10^1", "B0"]:
        r, med, mrp, stats = compute_table(bl_name, ddal_source)
        matches, total, mismatches, med_matches = verify_table(bl_name, r, med, mrp, stats)

        print(f"\n{bl_name}: MRP matches: {matches}/{total}", end="")
        if med_matches:
            print(" (medians OK)")
        else:
            print(" (MEDIAN MISMATCH!)")
            all_match = False

        if mismatches:
            all_match = False
            print("  Mismatches (comp vs ref):")
            for det, label, comp, ref in mismatches:
                comp_s = f"{comp:.2f}" if comp is not None else "--"
                ref_s = f"{ref:.2f}" if ref is not None else "--"
                if det == "MEDIAN":
                    print(f"    MEDIAN/{label}: comp={comp_s} ref={ref_s}")
                else:
                    print(f"    {det:10} {label:15} comp={comp_s:>8} ref={ref_s:>8}")
        else:
            print("  PERFECT MATCH!")

    if not update_ddal and all_match:
        print("\n" + "=" * 70)
        print("ALL TABLES REPRODUCED SUCCESSFULLY!")
        print("Now generating updated tex with new DDAL data...")
        print("=" * 70)
        # Auto-generate with new DDAL
        tex = generate_full_tex("current")
        if output_file:
            Path(output_file).write_text(tex)
            print(f"Written to {output_file}")
        else:
            out = BASE / "mtr_runtime_updated.tex"
            out.write_text(tex)
            print(f"Written to {out}")

    if update_ddal:
        tex = generate_full_tex("current")
        if output_file:
            Path(output_file).write_text(tex)
            print(f"\nWritten to {output_file}")
        else:
            out = BASE / "mtr_runtime_updated.tex"
            out.write_text(tex)
            print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
