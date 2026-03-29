#!/usr/bin/env python3
"""Benchmark SleepLM inference in Python (PyTorch CPU and MPS/CUDA).

Usage:
    python3 scripts/bench_python.py [--device cpu|mps|cuda]

Outputs CSV to stdout (same format as Rust bench binary).
"""

import argparse
import sys
import os
import time

# Add SleepLM source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../SleepLM-Base/src"))

import torch
import torch.nn.functional as F


def build_model(device):
    """Build SleepLM model with random weights on the given device."""
    from open_clip import create_model

    model = create_model(
        "sleep_coca_base_dualtransformer",
        precision="fp32",
        device=device,
        num_caption_channels=5,
        prefix_len=2,
    )
    model.eval()
    return model


def bench(fn, warmup=3, iters=10):
    """Time a function, return list of ms."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def print_stats(backend, component, batch, times):
    import statistics
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0
    mn = min(times)
    mx = max(times)
    print(f"{backend},{component},{batch},{n},{n},{mean:.2f},{std:.2f},{mn:.2f},{mx:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    backend_name = f"Python-{args.device.upper()}"

    print(f"# backend: {backend_name}", file=sys.stderr)
    print(f"# building model…", file=sys.stderr)
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"# params: {n_params:,}", file=sys.stderr)

    batch_sizes = [1, 4]
    warmup = 1
    iters = 3

    print("backend,component,batch,warmup,iters,mean_ms,std_ms,min_ms,max_ms")

    for bs in batch_sizes:
        # ── Biosignal encoding ───────────────────────────────────────────
        signal = torch.randn(bs, 10, 1920, device=device)

        @torch.inference_mode()
        def bio_encode():
            return model.encode_image(signal)

        times = bench(bio_encode, warmup, iters)
        print_stats(backend_name, "bio_encode", bs, times)

        # ── Text encoding ────────────────────────────────────────────────
        text = torch.randint(0, 49408, (bs, 77), device=device)

        @torch.inference_mode()
        def text_encode():
            return model.encode_text(text)

        times = bench(text_encode, warmup, iters)
        print_stats(backend_name, "text_encode", bs, times)

        # ── Full forward ─────────────────────────────────────────────────
        ch_idx = torch.zeros(bs, 2, dtype=torch.long, device=device)

        @torch.inference_mode()
        def full_forward():
            return model(signal, text, channel_indices=ch_idx)

        times = bench(full_forward, warmup, iters)
        print_stats(backend_name, "full_forward", bs, times)

    print("# done", file=sys.stderr)


if __name__ == "__main__":
    main()
