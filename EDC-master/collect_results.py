"""
collect_results.py

Parses the .log files produced by the three run scripts and prints
a clean comparison table across all methods and datasets.

Usage (run from EDC-master/):
    python collect_results.py

Output:
    - Printed table in the terminal
    - results_comparison.csv  saved next to this script
"""

import os
import re
import csv

# ─── CONFIG ──────────────────────────────────────────────────────────────────
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Map log-file prefix → display method name
METHODS = {
    "edc_baseline":     "EDC Baseline (ImageNet)",
    "edc_ssl_frozen":   "SSL-EDC Frozen",
    "edc_ssl_finetune": "SSL-EDC Finetune",
}

DATASETS = ["aptos", "br35h", "isic2018", "oct2017", "lungct"]

METRICS = ["AUC", "F1-score", "Accuracy", "Recall (Sensitivity)", "Specificity"]

# Regex patterns — match pandas right-aligned metric name + space-separated value
# Actual pandas output looks like:
#   "                 AUC 0.9123"
#   "Recall (Sensitivity) 0.8500"
PATTERNS = {
    "AUC":                  re.compile(r"^\s*AUC\s+([\d.]+)", re.MULTILINE),
    "F1-score":             re.compile(r"^\s*F1-score\s+([\d.]+)", re.MULTILINE),
    "Accuracy":             re.compile(r"^\s*Accuracy\s+([\d.]+)", re.MULTILINE),
    "Recall (Sensitivity)": re.compile(r"^\s*Recall \(Sensitivity\)\s+([\d.]+)", re.MULTILINE),
    "Specificity":          re.compile(r"^\s*Specificity\s+([\d.]+)", re.MULTILINE),
}


def parse_log(log_path):
    """Return dict of metric->float from a log file, or None if file missing."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        text = f.read()

    # Only parse inside the LAST "FINAL EVALUATION METRICS" block
    # to avoid picking up intermediate eval prints during training
    marker = "FINAL EVALUATION METRICS"
    last_pos = text.rfind(marker)
    block = text[last_pos:] if last_pos != -1 else text

    results = {}
    for metric, pattern in PATTERNS.items():
        matches = pattern.findall(block)
        results[metric] = float(matches[-1]) if matches else None

    return results


def fmt(val):
    if val is None:
        return "  N/A  "
    return f"{val:.4f}"


def main():
    # ── Collect all results ───────────────────────────────────────────────────
    data = {}
    for method_key, method_name in METHODS.items():
        data[method_name] = {}
        for dataset in DATASETS:
            log_file = os.path.join(LOGS_DIR, f"{method_key}_{dataset}.log")
            data[method_name][dataset] = parse_log(log_file)

    method_names = list(METHODS.values())

    DS_W  = 12
    COL_W = 26

    # ── Per-metric table ──────────────────────────────────────────────────────
    print("\n" + "=" * (DS_W + len(method_names) * (COL_W + 2) + 2))
    print("  ANOMALY DETECTION RESULTS — COMPARISON TABLE")
    print("=" * (DS_W + len(method_names) * (COL_W + 2) + 2))

    for metric in METRICS:
        print(f"\n  ── {metric} ──")
        header = f"  {'Dataset':<{DS_W}}"
        for m in method_names:
            header += f"  {m:<{COL_W}}"
        print(header)
        print("  " + "-" * (DS_W + len(method_names) * (COL_W + 2)))
        for dataset in DATASETS:
            row = f"  {dataset.upper():<{DS_W}}"
            for m in method_names:
                val = data[m][dataset][metric] if data[m][dataset] else None
                row += f"  {fmt(val):<{COL_W}}"
            print(row)

    print("\n" + "=" * (DS_W + len(method_names) * (COL_W + 2) + 2))

    # ── Per-dataset summary ───────────────────────────────────────────────────
    print("\n\n  PER-DATASET DETAILED SUMMARY\n")
    for dataset in DATASETS:
        print(f"  {'─'*70}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"  {'Metric':<26}", end="")
        for m in method_names:
            short = m.replace("EDC ", "").replace(" (ImageNet)", "")
            print(f"  {short:<22}", end="")
        print()
        for metric in METRICS:
            row = f"  {metric:<26}"
            for m in method_names:
                val = data[m][dataset][metric] if data[m][dataset] else None
                row += f"  {fmt(val):<22}"
            print(row)
    print(f"  {'─'*70}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_comparison.csv")
    rows = [["Method", "Dataset"] + METRICS]
    for method_key, method_name in METHODS.items():
        for dataset in DATASETS:
            parsed = data[method_name][dataset]
            row = [method_name, dataset.upper()]
            for metric in METRICS:
                row.append(f"{parsed[metric]:.4f}" if parsed and parsed[metric] is not None else "")
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  CSV saved → {csv_path}\n")

    # ── Warn about any missing logs ───────────────────────────────────────────
    missing = [
        f"  MISSING: logs/{mk}_{ds}.log"
        for mk in METHODS
        for ds in DATASETS
        if not os.path.exists(os.path.join(LOGS_DIR, f"{mk}_{ds}.log"))
    ]
    if missing:
        print("  ⚠  These log files were not found (run not complete yet):")
        for m in missing:
            print(m)
        print()


if __name__ == "__main__":
    main()
