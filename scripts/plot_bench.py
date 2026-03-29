#!/usr/bin/env python3
"""Generate benchmark charts from CSV data.

Usage:
    python3 scripts/plot_bench.py

Reads all CSV files from figures/*.csv, generates charts in figures/.
"""

import os
import glob
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections import defaultdict

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def load_csv_files():
    """Load all benchmark CSV files and merge into a single dataset."""
    rows = []
    for path in sorted(glob.glob(os.path.join(FIGURES_DIR, "*.csv"))):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["batch"] = int(row["batch"])
                row["mean_ms"] = float(row["mean_ms"])
                row["std_ms"] = float(row["std_ms"])
                row["min_ms"] = float(row["min_ms"])
                row["max_ms"] = float(row["max_ms"])
                rows.append(row)
    return rows


def plot_by_component(rows):
    """One chart per component (bio_encode, text_encode, full_forward)."""
    components = sorted(set(r["component"] for r in rows))
    backends = sorted(set(r["backend"] for r in rows))

    # Color scheme
    colors = {
        "CPU-NdArray":     "#4C72B0",
        "CPU-Accelerate":  "#55A868",
        "GPU-Metal":       "#C44E52",
        "GPU-wgpu":        "#DD8452",
        "Python-CPU":      "#8172B3",
        "Python-MPS":      "#937860",
        "Python-CUDA":     "#DA8BC3",
    }

    for comp in components:
        fig, ax = plt.subplots(figsize=(8, 5))
        comp_rows = [r for r in rows if r["component"] == comp]
        batches = sorted(set(r["batch"] for r in comp_rows))
        n_backends = len(backends)
        width = 0.8 / max(n_backends, 1)
        x = np.arange(len(batches))

        for i, be in enumerate(backends):
            be_rows = {r["batch"]: r for r in comp_rows if r["backend"] == be}
            if not be_rows:
                continue
            means = [be_rows[b]["mean_ms"] if b in be_rows else 0 for b in batches]
            stds  = [be_rows[b]["std_ms"]  if b in be_rows else 0 for b in batches]
            offset = (i - (n_backends - 1) / 2) * width
            color = colors.get(be, f"C{i}")
            ax.bar(x + offset, means, width * 0.9, yerr=stds, label=be,
                   color=color, alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)
            # Value labels
            for xi, m in zip(x + offset, means):
                if m > 0:
                    ax.text(xi, m + max(means) * 0.02, f"{m:.1f}",
                            ha="center", va="bottom", fontsize=7, rotation=0)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"SleepLM — {comp.replace('_', ' ').title()}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in batches])
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        path = os.path.join(FIGURES_DIR, f"bench_{comp}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  → {path}")


def plot_summary(rows):
    """Single summary chart: mean latency at batch=1 across all components."""
    components = ["bio_encode", "text_encode", "full_forward"]
    backends = sorted(set(r["backend"] for r in rows))

    colors = {
        "CPU-NdArray":     "#4C72B0",
        "CPU-Accelerate":  "#55A868",
        "GPU-Metal":       "#C44E52",
        "GPU-wgpu":        "#DD8452",
        "Python-CPU":      "#8172B3",
        "Python-MPS":      "#937860",
        "Python-CUDA":     "#DA8BC3",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(components))
    n = len(backends)
    width = 0.8 / max(n, 1)

    for i, be in enumerate(backends):
        means = []
        stds = []
        for comp in components:
            match = [r for r in rows if r["backend"] == be and r["component"] == comp and r["batch"] == 1]
            if match:
                means.append(match[0]["mean_ms"])
                stds.append(match[0]["std_ms"])
            else:
                means.append(0)
                stds.append(0)
        if all(m == 0 for m in means):
            continue
        offset = (i - (n - 1) / 2) * width
        color = colors.get(be, f"C{i}")
        bars = ax.bar(x + offset, means, width * 0.9, yerr=stds, label=be,
                      color=color, alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)
        for xi, m in zip(x + offset, means):
            if m > 0:
                ax.text(xi, m + max(means) * 0.02, f"{m:.1f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Component")
    ax.set_ylabel("Latency (ms) — batch=1")
    ax.set_title("SleepLM Inference — Rust vs Python (batch=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in components])
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(FIGURES_DIR, "bench_summary.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_scaling(rows):
    """Line chart: latency vs batch size for each backend (full_forward)."""
    backends = sorted(set(r["backend"] for r in rows))
    colors = {
        "CPU-NdArray":     "#4C72B0",
        "CPU-Accelerate":  "#55A868",
        "GPU-Metal":       "#C44E52",
        "GPU-wgpu":        "#DD8452",
        "Python-CPU":      "#8172B3",
        "Python-MPS":      "#937860",
        "Python-CUDA":     "#DA8BC3",
    }
    markers = {"CPU-NdArray": "o", "CPU-Accelerate": "s", "GPU-Metal": "D",
               "GPU-wgpu": "^", "Python-CPU": "v", "Python-MPS": "<", "Python-CUDA": ">"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for be in backends:
        be_rows = sorted(
            [r for r in rows if r["backend"] == be and r["component"] == "full_forward"],
            key=lambda r: r["batch"],
        )
        if not be_rows:
            continue
        batches = [r["batch"] for r in be_rows]
        means = [r["mean_ms"] for r in be_rows]
        stds = [r["std_ms"] for r in be_rows]
        color = colors.get(be, "gray")
        marker = markers.get(be, "o")
        ax.errorbar(batches, means, yerr=stds, label=be, color=color,
                    marker=marker, markersize=6, capsize=3, linewidth=1.5)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("SleepLM Full Forward — Batch Scaling")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    path = os.path.join(FIGURES_DIR, "bench_scaling.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rows = load_csv_files()
    if not rows:
        print("No CSV data found in figures/. Run benchmarks first.", file=sys.stderr)
        return

    print(f"Loaded {len(rows)} rows from figures/*.csv")
    print(f"Backends: {sorted(set(r['backend'] for r in rows))}")
    print(f"Components: {sorted(set(r['component'] for r in rows))}")
    print()

    plot_by_component(rows)
    plot_summary(rows)
    plot_scaling(rows)
    print("\nDone!")


if __name__ == "__main__":
    main()
