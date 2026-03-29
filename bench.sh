#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p figures

echo "═══════════════════════════════════════════════════════════"
echo " SleepLM Benchmark Suite"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Rust CPU (NdArray) ────────────────────────────────────────────────────
echo ""
echo "▶ Building Rust (CPU — NdArray + Rayon)…"
cargo build --release --bin bench 2>&1 | tail -1

echo "▶ Running Rust CPU benchmark…"
cargo run --release --bin bench 2>/dev/null | tee /tmp/sleeplm_bench_cpu.txt
# Extract CSV lines (skip non-CSV)
grep -E '^[A-Za-z].*,' /tmp/sleeplm_bench_cpu.txt > figures/rust_cpu.csv || true
echo "  → figures/rust_cpu.csv"

# ── 2. Rust GPU (Metal) — macOS only ────────────────────────────────────────
if [[ "$(uname)" == "Darwin" ]]; then
    echo ""
    echo "▶ Building Rust (GPU — Metal)…"
    cargo build --release --bin bench --no-default-features --features metal 2>&1 | tail -1

    echo "▶ Running Rust GPU (Metal) benchmark…"
    cargo run --release --bin bench --no-default-features --features metal 2>/dev/null \
        | tee /tmp/sleeplm_bench_gpu.txt
    grep -E '^[A-Za-z].*,' /tmp/sleeplm_bench_gpu.txt > figures/rust_gpu.csv || true
    echo "  → figures/rust_gpu.csv"
fi

# ── 3. Python CPU ────────────────────────────────────────────────────────────
echo ""
echo "▶ Running Python CPU benchmark…"
python3 scripts/bench_python.py --device cpu 2>/dev/null | tee /tmp/sleeplm_bench_py_cpu.txt
grep -E '^[A-Za-z].*,' /tmp/sleeplm_bench_py_cpu.txt > figures/python_cpu.csv || true
echo "  → figures/python_cpu.csv"

# ── 4. Python MPS (Apple Silicon) ───────────────────────────────────────────
if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    echo ""
    echo "▶ Running Python MPS benchmark…"
    python3 scripts/bench_python.py --device mps 2>/dev/null | tee /tmp/sleeplm_bench_py_mps.txt
    grep -E '^[A-Za-z].*,' /tmp/sleeplm_bench_py_mps.txt > figures/python_mps.csv || true
    echo "  → figures/python_mps.csv"
fi

# ── 5. Generate charts ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Generating charts…"
echo "═══════════════════════════════════════════════════════════"
python3 scripts/plot_bench.py

echo ""
echo "Done! Charts saved to figures/"
ls -la figures/*.png 2>/dev/null || echo "(no charts generated)"
