#!/usr/bin/env python3
"""
Prune Qwen3.8-Flash-Next (qwen4exp, 48L, 2560 embd, 176B) -> Qwen3.8-35B-A3B-NEXT
Target: 40 layers, 2048 embd, keep PLE + shared + 64 routed top-8, BF16 out.
Usage on D1 after pulling R3's BF16 shards (or Q4 dequant as fallback):
  python C:\code\llama\source-pipyakas\tools\qwen38_prune.py --src D:\llama\models\... --dst C:\code\llama\models\Qwen3.8-35B-A3B-NEXT
No training - Ornith self-run finetune recovers afterwards.
"""
import argparse, os, re, json, sys
from pathlib import Path
try:
    import gguf
    import numpy as np
except ImportError:
    print("need: pip install gguf numpy ggml-python torch -- if missing, fallback to llama-quantize path")
    sys.exit(0)

ARCH = "qwen4exp"
SRC_EMBD, DST_EMBD = 2560, 2048
SRC_LAYERS, DST_LAYERS = 48, 40
KEEP_LAYERS = list(range(40))  # 0..39, drop last 8 where entropy low; alt: even drop

def slice_embd(t, axis=0):
    # center-crop 2560->2048 (or learned proj) - simple trunc keeps RoPE heads aligned
    # head_dim 128: 2560/128=20 heads, 2048/128=16 heads -> keep first 16 heads worth
    if t.shape[axis] == SRC_EMBD:
        sl = slice(0, DST_EMBD)
        return t[sl] if axis==0 else t[:, sl] if t.ndim==2 else t
    if t.shape[axis] == SRC_EMBD * 3: # fused QKV
        return t
    return t

def rename_for_next(name):
    # 3.8 PLE already exists as blk.*.ple_* - keep; 35B adds no new tensors
    return name

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    print(f"Prune {args.src} -> {args.dst} : {SRC_EMBD}->{DST_EMBD}, {SRC_LAYERS}->{DST_LAYERS}L")
    # TODO: actual GGUF merge - D1 will run this after R3 BF16 pull
    # 1. enumerate shards, 2. for each tensor: slice embd if needed, skip dropped layers, 3. write dst GGUF
    print("STUB - run full version after BF16 shards are on D1")
