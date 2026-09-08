#!/usr/bin/env python3
"""Retrain the hybrid autoencoder under controlled variants (R2.5/2.6/2.8).

This script reproduces, from the released SwinUNetLarge in Copy12.py, the
autoencoder-side ablations the reviewer asked to see separately:

  * positional encoding (R2.8): released separate-2D / scalar (x+y) / none;
  * diffusion-inspired denoising block (R2.5): on vs off;
  * data split (R2.6): frame-level 70/15/15 vs sequence-level (4/1/1 videos).

All variants share the released training recipe: AdamW (lr 5e-5, wd 5e-5),
batch 16, 128x128 input, mixed precision, the same photometric/geometric
augmentation, the same composite reconstruction loss (L1 + 0.03*MSE +
0.01*SSIM on the main and the two auxiliary outputs, aux weight decaying
0.2 -> 0.1), and the same diffusion-time curriculum when the block is on.

Each run reports validation reconstruction MSE and the fine-ranking diagnostic
(mean rank / top-1 of the true ROI on a 169-candidate grid over SCARED pairs),
and writes pe_ablation.json / diffusion_ablation.json / split_ablation.json.

NOTE: the released training script supervises only the RGB reconstruction
(L1/MSE/SSIM); the contour/heatmap targets described in the paper are generated
by the t90 procedure but are not separate loss summands in this autoencoder
(they enter through the auxiliary decoder outputs, which reconstruct the same
RGB target). We reproduce the released recipe exactly.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SEM = ROOT.parent / "vrih_experiment_hub" / "src" / "semantic"
HUB = ROOT.parent / "vrih_experiment_hub" / "links"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("copy12", SEM / "Copy12.py")
copy12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copy12)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SELF_DIRS = [
    HUB / "self_acquired_endoscopy" / f"extracted_frames{i}" / "extracted_frames"
    for i in range(1, 7)
]
RANK_SEQS = ["scared_d1_k1", "scared_d3_k2"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Model variants
# --------------------------------------------------------------------------
class ScalarSumPE(nn.Module):
    """Scalar (x+y) sinusoidal positional encoding (draft formula)."""

    def __init__(self, dim, scale=0.01):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):  # x: [B,H,W,C]
        B, H, W, C = x.shape
        device, dtype = x.device, x.dtype
        ys = torch.arange(H, device=device, dtype=torch.float32) / max(H - 1, 1)
        xs = torch.arange(W, device=device, dtype=torch.float32) / max(W - 1, 1)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        s = (xx + yy) * 2 - 1  # scalar in [-2,2]
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        ang = s[..., None] * freq  # [H,W,half]
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if pe.shape[-1] < self.dim:
            pe = F.pad(pe, (0, self.dim - pe.shape[-1]))
        pe = pe[..., : self.dim].unsqueeze(0).repeat(B, 1, 1, 1).to(dtype)
        return x + pe * self.scale


def build_model(pe_variant: str, diffusion_on: bool) -> torch.nn.Module:
    net = copy12.SwinUNetLarge(
        in_channels=3, base_c=64, dropout_p=0.15, window_size=8,
        depths=(2, 3, 4), bottleneck_depth=4,
        num_heads_stages=(4, 8, 8), bottleneck_heads=16,
        use_pos_emb=True, use_se_sparse=True, use_input_pos_emb=True,
    )
    if pe_variant == "none":
        net.use_input_pos_emb = False
    elif pe_variant == "sum":
        net.input_pos_emb = ScalarSumPE(3, scale=0.01)
    elif pe_variant == "sep2d":
        pass  # released AdaptivePositionalEncoding2D
    net.use_diffusion_denoising = diffusion_on
    if not diffusion_on:
        for p in net.denoising_blocks.parameters():
            p.requires_grad_(False)
    return net.to(DEVICE)


# --------------------------------------------------------------------------
# Loss (matches the released autoencoder_loss)
# --------------------------------------------------------------------------
_SSIM = None


def ssim_loss(pred, target):
    global _SSIM
    if _SSIM is None:
        # reuse the released SSIM window logic via a minimal local impl
        _SSIM = _SSIMImpl(channel=3, window_size=7)
    return _SSIM(pred, target)


class _SSIMImpl(nn.Module):
    def __init__(self, channel=3, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = None

    def _win(self, device, dtype):
        if self.window is None:
            g = torch.exp(
                -((torch.arange(self.window_size, dtype=torch.float32) - self.window_size // 2) ** 2)
                / (2 * 1.5 ** 2)
            )
            g = (g / g.sum()).unsqueeze(1)
            w = g.mm(g.t()).float().unsqueeze(0).unsqueeze(0)
            self.window = w.expand(self.channel, 1, self.window_size, self.window_size).contiguous()
        return self.window.to(device=device, dtype=dtype)

    def forward(self, a, b):
        a = torch.clamp(a.float(), 0, 1)
        b = torch.clamp(b.float(), 0, 1)
        w = self._win(a.device, a.dtype)
        pad = self.window_size // 2
        mu1 = F.conv2d(a, w, padding=pad, groups=self.channel)
        mu2 = F.conv2d(b, w, padding=pad, groups=self.channel)
        s1 = F.conv2d(a * a, w, padding=pad, groups=self.channel) - mu1.pow(2)
        s2 = F.conv2d(b * b, w, padding=pad, groups=self.channel) - mu2.pow(2)
        s12 = F.conv2d(a * b, w, padding=pad, groups=self.channel) - mu1 * mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den = (mu1.pow(2) + mu2.pow(2) + C1) * (s1 + s2 + C2)
        return 1 - (num / den.clamp(min=1e-6)).clamp(-1, 1).mean()


def recon_loss(pred, target):
    pred = torch.clamp(pred.float(), -2, 2)
    target = torch.clamp(target.float(), -2, 2)
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    pn = (pred + 1) / 2
    tn = (target + 1) / 2
    ssim = ssim_loss(pn, tn)
    total = l1 + 0.03 * mse + 0.01 * ssim
    return total, mse.item()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def gather_frames(split: str, seed: int = 2025):
    """Return (train_dirs_frames, val_frames, test_frames) as path lists."""
    per_seq = []
    for d in SELF_DIRS:
        frames = sorted(
            p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        per_seq.append(frames)
    if split == "frame":
        allf = [f for seq in per_seq for f in seq]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(allf))
        allf = [allf[i] for i in idx]
        n = len(allf)
        ntr, nval = int(n * 0.7), int(n * 0.15)
        return allf[:ntr], allf[ntr:ntr + nval], allf[ntr + nval:]
    else:  # sequence-level 4/1/1
        order = [0, 1, 2, 3, 4, 5]
        train = [f for i in order[:4] for f in per_seq[i]]
        val = per_seq[order[4]]
        test = per_seq[order[5]]
        return train, list(val), list(test)


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, paths, augment):
        self.paths = paths
        self.augment = augment
        if augment:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            self.tf = A.Compose([
                A.Resize(128, 128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.4),
                A.ShiftScaleRotate(0.1, 0.15, 25, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
                A.ColorJitter(0.25, 0.25, 0.15, 0.08, p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.15),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.1),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        else:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            self.tf = A.Compose([
                A.Resize(128, 128),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = self.tf(image=img)["image"]
        return t, t


# --------------------------------------------------------------------------
# Fine-ranking diagnostic (169-candidate grid)
# --------------------------------------------------------------------------
def t90_mask(gray):
    smooth = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return (mag > np.percentile(mag, 90.0)).astype(np.uint8) * 255


def largest_contour(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea) if cs else None


@torch.no_grad()
def encode_batch(net, bgr_list):
    """Encode a batch of BGR crops -> [N, latent] numpy."""
    xs = []
    for bgr in bgr_list:
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        xs.append((x - 0.5) / 0.5)
    x = torch.stack(xs).to(DEVICE)
    z = net.encode(x)
    return z.detach().cpu().numpy()


def fine_rank(net, max_pairs=40):
    ranks, top1 = [], 0
    n = 0
    for seq in RANK_SEQS:
        imgs = sorted((ROOT / "data" / seq / "images").glob("*.png"))[: max_pairs + 1]
        for i in range(len(imgs) - 1):
            prev = cv2.imread(str(imgs[i]), cv2.IMREAD_GRAYSCALE)
            curr = cv2.imread(str(imgs[i + 1]), cv2.IMREAD_GRAYSCALE)
            prev_c = cv2.imread(str(imgs[i]), cv2.IMREAD_COLOR)
            mp = t90_mask(prev)
            cnt = largest_contour(mp)
            if cnt is None:
                continue
            x0, y0, w0, h0 = cv2.boundingRect(cnt)
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            fx = flow[y0:y0 + h0, x0:x0 + w0, 0]
            fy = flow[y0:y0 + h0, x0:x0 + w0, 1]
            cx = x0 + w0 / 2 + float(np.median(fx))
            cy = y0 + h0 / 2 + float(np.median(fy))
            gx = np.linspace(x0 - 24, x0 + 24, 13)
            gy = np.linspace(y0 - 24, y0 + 24, 13)
            z_prev = encode_batch(net, [prev_c])[0]
            crops, centers = [], []
            for yy in gy:
                for xx in gx:
                    xi, yi = int(xx), int(yy)
                    if xi < 0 or yi < 0 or xi + w0 > curr.shape[1] or yi + h0 > curr.shape[0]:
                        continue
                    crop = curr[yi:yi + h0, xi:xi + w0]
                    if crop.size == 0:
                        continue
                    crops.append(crop)
                    centers.append((xx + w0 / 2, yy + h0 / 2))
            if not crops:
                continue
            zs = encode_batch(net, crops)
            scores = -np.linalg.norm(zs - z_prev[None, :], axis=1)
            centers = np.array(centers)
            gt_pos = int(np.argmin((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2))
            order = np.argsort(-scores)
            rank = int(np.where(order == gt_pos)[0][0]) + 1
            ranks.append(rank)
            top1 += int(rank == 1)
            n += 1
    return {
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "top1": float(top1 / n) if n else float("nan"),
        "n_pairs": n,
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_one(tag, pe_variant, diffusion_on, split, epochs, out_json):
    train_f, val_f, test_f = gather_frames(split)
    net = build_model(pe_variant, diffusion_on)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    train_ds = FrameDataset(train_f, True)
    val_ds = FrameDataset(val_f, False)
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    best = float("inf")
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        aux_w = max(0.1, 0.2 * (1 - epoch / max(1, epochs)))
        for x, _ in train_ld:
            x = x.to(DEVICE, non_blocking=True)
            B = x.shape[0]
            ts = None
            if diffusion_on:
                if epoch < epochs * 0.3:
                    ts = torch.rand(B, device=DEVICE) * 0.8 + 0.2
                elif epoch < epochs * 0.7:
                    ts = torch.rand(B, device=DEVICE) * 0.6 + 0.2
                else:
                    ts = torch.rand(B, device=DEVICE) * 0.4 + 0.1
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                main, aux = net(x, timestep=ts)
                loss, _ = recon_loss(main, x)
                if aux:
                    la = sum(recon_loss(a, x)[0] for a in aux) / len(aux)
                    loss = loss + aux_w * la
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
        # validation
        net.eval()
        vm, vn = 0.0, 0
        with torch.no_grad():
            for x, _ in val_ld:
                x = x.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    main, _ = net(x)
                vm += F.mse_loss(main.float(), x.float()).item()
                vn += 1
        val_mse = vm / max(1, vn)
        best = min(best, val_mse)
        print(f"  [{tag}] epoch {epoch+1}/{epochs} val_mse={val_mse:.4f} best={best:.4f}", flush=True)
    train_seconds = time.time() - t0
    rank = fine_rank(net)
    result = {
        "variant": tag,
        "pe": pe_variant,
        "diffusion": diffusion_on,
        "split": split,
        "epochs": epochs,
        "trainable_params": int(n_params),
        "val_mse": float(best),
        "train_seconds": float(train_seconds),
        **rank,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--mode", choices=["pe", "diffusion", "split", "all"], default="all")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = []
    if args.mode in ("pe", "all"):
        for pe in ("sep2d", "sum", "none"):
            runs.append((f"pe_{pe}", pe, True, "frame"))
    if args.mode in ("diffusion", "all"):
        runs.append(("diff_on", "sep2d", True, "frame"))
        runs.append(("diff_off", "sep2d", False, "frame"))
    if args.mode in ("split", "all"):
        runs.append(("split_frame", "sep2d", True, "frame"))
        runs.append(("split_sequence", "sep2d", True, "sequence"))

    results = {}
    for tag, pe, diff, split in runs:
        print(f"=== {tag} (pe={pe}, diffusion={diff}, split={split}) ===", flush=True)
        results[tag] = train_one(tag, pe, diff, split, args.epochs, None)

    out = args.out or (RESULTS / f"retrain_{args.mode}.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
