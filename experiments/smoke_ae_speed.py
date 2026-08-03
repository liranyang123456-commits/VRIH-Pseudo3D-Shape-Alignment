"""Break down AE pipeline cost: albumentations preprocess vs model.encode."""
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
transform = manager.get_transform()
device = manager.get_device()
print("transform:", transform)

patch = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

# 1) albumentations preprocessing cost
t0 = time.perf_counter()
tensors = [transform(image=patch)["image"] for _ in range(256)]
print("albu preprocess 256: %.3fs" % (time.perf_counter() - t0))

x = torch.stack(tensors).to(device)
for bs in (64, 256):
    t0 = time.perf_counter()
    with torch.no_grad():
        z = model.encode(x[:bs])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"encode batch {bs}: {dt:.3f}s -> {bs/dt:.0f} patches/s")

# 2) manual vectorized preprocessing
import cv2

t0 = time.perf_counter()
arr = np.stack([cv2.resize(patch, (128, 128)) for _ in range(256)]).astype(np.float32) / 255.0
arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
xm = torch.from_numpy(arr.transpose(0, 3, 1, 2)).float().to(device)
print("manual preprocess 256: %.3fs" % (time.perf_counter() - t0))
t0 = time.perf_counter()
with torch.no_grad():
    z2 = model.encode(xm)
torch.cuda.synchronize()
print("encode manual 256: %.3fs" % (time.perf_counter() - t0))
print("latent diff (albu vs manual):", float((z[:1] - z2[:1]).abs().max()))
