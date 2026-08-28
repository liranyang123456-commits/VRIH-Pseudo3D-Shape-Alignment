"""Measure AE parameter count, GPU memory, and inference throughput (R2.7)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("VAL_DISABLE_TORCH_COMPILE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "vrih_experiment_hub" / "src" / "semantic"))

import torch
from extract_semantic_features_Function import ModelManager  # noqa: E402

manager = ModelManager()
manager.initialize(model_path=r"D:\reloc3r\Autoencoder_129.pth", image_size=(128, 128), device="cuda")
model = manager.get_model()
device = manager.get_device()

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,} ({n_params/1e6:.1f} M)")

x = torch.randn(1, 3, 128, 128, device=device)
with torch.no_grad():
    for _ in range(5):
        model.encode(x)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
N = 50
with torch.no_grad():
    for _ in range(N):
        model.encode(x)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"single-crop encode: {dt*1000:.1f} ms ({1/dt:.1f} fps), peak GPU memory {peak:.2f} GiB")

x64 = torch.randn(64, 3, 128, 128, device=device)
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
with torch.no_grad():
    model.encode(x64)
torch.cuda.synchronize()
dt64 = time.perf_counter() - t0
peak64 = torch.cuda.max_memory_allocated() / 1024**3
print(f"batch-64 encode: {dt64*1000:.1f} ms ({64/dt64:.0f} crops/s), peak GPU memory {peak64:.2f} GiB")
