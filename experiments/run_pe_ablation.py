#!/usr/bin/env python3
"""Positional-encoding ablation for the contour autoencoder (R2.8).

Trains a compact convolution+attention autoencoder on t90 contour images
(128x128) from 3,002 self-acquired endoscopic frames under three positional
encodings that differ ONLY in the PE term:

* ``sum``     — the paper's scalar (x+y) sinusoidal encoding;
* ``sep2d``   — separate horizontal/vertical sinusoidal encodings (standard 2D);
* ``none``    — no positional encoding (control).

Each model is then evaluated on the fine-ranking protocol (49-candidate ROI
grid, two SCARED sequences x 40 pairs): top-1 accuracy and mean rank of the
true ROI by latent similarity, plus validation reconstruction loss.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FRAMES_ROOT = ROOT / "data" / "pe_ablation_frames"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG = 128
LATENT = 128
EPOCHS = 40
BATCH = 64
LR = 1e-3


def t90_contour(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    smooth = cv2.bilateralFilter(g, 7, 0.1, 3.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 90.0)
    return (mag > thr).astype(np.float32)


class ContourDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        gray = cv2.imread(str(self.paths[idx]), cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (IMG, IMG))
        c = t90_contour(gray)
        return torch.from_numpy(c[None])


def sinusoidal_1d(pos: np.ndarray, dim: int) -> np.ndarray:
    pe = np.zeros((len(pos), dim), dtype=np.float32)
    div = np.exp(np.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = np.sin(pos[:, None] * div)
    pe[:, 1::2] = np.cos(pos[:, None] * div)
    return pe


def make_pe(kind: str, grid: int, dim: int) -> torch.Tensor:
    ys, xs = np.mgrid[0:grid, 0:grid]
    if kind == "sum":
        return torch.from_numpy(sinusoidal_1d((xs + ys).ravel().astype(np.float32), dim))
    if kind == "sep2d":
        pex = sinusoidal_1d(xs.ravel().astype(np.float32), dim // 2)
        pey = sinusoidal_1d(ys.ravel().astype(np.float32), dim // 2)
        return torch.from_numpy(np.concatenate([pex, pey], axis=1))
    return torch.zeros(grid * grid, dim)


class CompactAE(nn.Module):
    def __init__(self, pe_kind: str):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
        )
        self.grid = IMG // 8
        self.register_buffer("pe", make_pe(pe_kind, self.grid, 128))
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128))
        self.to_latent = nn.Linear(128, LATENT)
        self.from_latent = nn.Linear(LATENT, 128)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        f = self.enc(x)
        b, c, h, w = f.shape
        tokens = f.flatten(2).transpose(1, 2) + self.pe[None]
        tokens = tokens + self.attn(tokens, tokens, tokens)[0]
        tokens = tokens + self.ffn(tokens)
        z = torch.tanh(self.to_latent(tokens.mean(1)))
        f2 = self.from_latent(z)[:, :, None, None].expand(-1, -1, h, w)
        return self.dec(f2), z


def train_model(pe_kind: str, train_dl, val_dl) -> tuple[CompactAE, float]:
    model = CompactAE(pe_kind).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        for x in train_dl:
            x = x.to(DEVICE)
            recon, _ = model(x)
            loss = nn.functional.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        val_loss = float(np.mean([nn.functional.mse_loss(model(x.to(DEVICE))[0], x.to(DEVICE)).item() for x in val_dl]))
    return model, val_loss


def rank_eval(model: CompactAE) -> dict[str, float]:
    """Fine-ranking on the 49-candidate grid, two SCARED sequences x 40 pairs."""
    top1, ranks = [], []
    for seq in ("scared_d1_k1", "scared_d2_k1"):
        images = sorted((ROOT / "data" / seq / "images").glob("*.png"))[:41]
        for i in range(1, len(images)):
            prev = cv2.resize(cv2.imread(str(images[i - 1]), 0), (IMG, IMG))
            curr = cv2.resize(cv2.imread(str(images[i]), 0), (IMG, IMG))
            mask_prev = t90_contour(prev)
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            ys, xs = np.nonzero(mask_prev)
            if len(xs) == 0:
                continue
            cx = int(np.clip((xs + flow[ys, xs, 0]).mean(), 0, IMG - 1))
            cy = int(np.clip((ys + flow[ys, xs, 1]).mean(), 0, IMG - 1))
            with torch.no_grad():
                z_prev = model(torch.from_numpy(mask_prev[None, None]).to(DEVICE))[1]
                sims = []
                for gy in range(0, IMG - 31, 8):
                    for gx in range(0, IMG - 31, 8):
                        patch = t90_contour(curr)[gy:gy + 32, gx:gx + 32]
                        patch = cv2.resize(patch, (IMG, IMG))
                        z_c = model(torch.from_numpy(patch[None, None]).to(DEVICE))[1]
                        sims.append(1.0 / (1.0 + float(torch.norm(z_prev - z_c))))
            sims = np.array(sims)
            centers = [(gx + 16, gy + 16) for gy in range(0, IMG - 31, 8) for gx in range(0, IMG - 31, 8)]
            true_idx = int(np.argmin([(c[0] - cx) ** 2 + (c[1] - cy) ** 2 for c in centers]))
            order = np.argsort(-sims)
            rank = int(np.where(order == true_idx)[0][0]) + 1
            top1.append(1.0 if rank == 1 else 0.0)
            ranks.append(rank)
    return {"top1": float(np.mean(top1)), "mean_rank": float(np.mean(ranks)), "n_pairs": len(ranks)}


def main() -> None:
    paths = sorted(FRAMES_ROOT.rglob("*.jpg"))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(paths))
    n_train = int(0.8 * len(paths))
    train_paths = [paths[i] for i in perm[:n_train]]
    val_paths = [paths[i] for i in perm[n_train:]]
    train_dl = DataLoader(ContourDataset(train_paths), batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(ContourDataset(val_paths), batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"train {len(train_paths)} / val {len(val_paths)} on {DEVICE}")

    report = {}
    for kind in ("sum", "sep2d", "none"):
        t0 = time.perf_counter()
        model, val_loss = train_model(kind, train_dl, val_dl)
        train_s = time.perf_counter() - t0
        torch.save(model.state_dict(), RESULTS / f"pe_ablation_{kind}.pth")
        metrics = rank_eval(model)
        report[kind] = {"val_mse": val_loss, "train_seconds": train_s, **metrics}
        print(f"{kind}: val_mse={val_loss:.5f} top1={metrics['top1']:.3f} rank={metrics['mean_rank']:.1f} ({train_s:.0f}s)", flush=True)
        (RESULTS / "pe_ablation.json").write_text(json.dumps(report, indent=1))
    (RESULTS / "pe_ablation.json").write_text(json.dumps(report, indent=1))
    print("written pe_ablation.json")


if __name__ == "__main__":
    main()
