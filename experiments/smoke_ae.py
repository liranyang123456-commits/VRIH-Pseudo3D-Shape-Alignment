"""Smoke test: AE model loads and batch-encodes on GPU in py3d env."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("VAL_DISABLE_TORCH_COMPILE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "vrih_experiment_hub" / "src" / "semantic"))

import torch

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

from extract_semantic_features_Function import ModelManager  # noqa: E402

manager = ModelManager()
t0 = time.perf_counter()
manager.initialize(model_path=r"D:\reloc3r\Autoencoder_129.pth", image_size=(128, 128), device="cuda" if torch.cuda.is_available() else "cpu")
print("load time: %.1fs" % (time.perf_counter() - t0))
model = manager.get_model()
transform = manager.get_transform()
device = manager.get_device()

patch = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
x = torch.stack([transform(image=patch)["image"] for _ in range(64)]).to(device)
t0 = time.perf_counter()
with torch.no_grad():
    z = model.encode(x)
torch.cuda.synchronize() if device.type == "cuda" else None
dt = time.perf_counter() - t0
print("latent shape:", tuple(z.shape), "| 64 patches in %.3fs -> %.1f patches/s" % (dt, 64 / dt))
print("smoke OK")
