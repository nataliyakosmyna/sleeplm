#!/usr/bin/env python3
"""Convert SleepLM PyTorch checkpoint to safetensors format.

Usage:
    python3 scripts/convert_to_safetensors.py \
        --input model_checkpoint.pt \
        --output model.safetensors

Requirements:
    pip install torch safetensors
"""

import argparse
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Convert SleepLM .pt to .safetensors")
    parser.add_argument("--input", required=True, help="Input .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .safetensors file")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    
    # Extract state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip "module." prefix if present (from DDP training)
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        # Convert to float32 for safetensors compatibility
        if value.dtype in (torch.float16, torch.bfloat16):
            value = value.float()
        cleaned[clean_key] = value

    print(f"  {len(cleaned)} tensors")
    
    # Print summary
    total_params = sum(v.numel() for v in cleaned.values())
    total_bytes = sum(v.numel() * v.element_size() for v in cleaned.values())
    print(f"  {total_params:,} parameters ({total_bytes / 1e9:.2f} GB)")

    # Save
    print(f"Saving {args.output}...")
    save_file(cleaned, args.output)
    
    import os
    size = os.path.getsize(args.output)
    print(f"  {size / 1e9:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    main()
