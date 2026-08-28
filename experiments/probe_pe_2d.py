"""Verify the implemented 2D positional encoding has no anti-diagonal collision.

The paper's R1 text erroneously described a sinusoidal PE of the coordinate
SUM (x+y), which collides on anti-diagonals. The shipped implementation
(AdaptivePositionalEncoding2D in Copy12.py) instead encodes normalized x and y
independently with separate sin/cos frequency banks and concatenates them.
This probe instantiates that module and measures the minimum encoding distance
between anti-diagonal position pairs (x+y = const) versus a random baseline.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "vrih_experiment_hub" / "src" / "semantic"))
from Copy12 import AdaptivePositionalEncoding2D  # noqa: E402

DIM = 64
H = W = 32
pe = AdaptivePositionalEncoding2D(DIM)
x = torch.zeros(1, H, W, DIM)
codes = pe(x)[0]  # [H, W, C]

# far-apart anti-diagonal pairs: (x, y) with x + y = 20 but far apart
pairs = [((2, 18), (18, 2)), ((4, 16), (16, 4)), ((6, 14), (14, 6)), ((8, 12), (12, 8))]
dists = [torch.norm(codes[y1, x1] - codes[y2, x2]).item() for (x1, y1), (x2, y2) in pairs]
print("separable PE, far-apart same-sum pair distances:", [f"{d:.3f}" for d in dists])

# reference: random far-apart pairs
import random

random.seed(0)
ref = []
for _ in range(20):
    a = (random.randrange(W), random.randrange(H))
    b = (random.randrange(W), random.randrange(H))
    ref.append(torch.norm(codes[a[1], a[0]] - codes[b[1], b[0]]).item())
print("random-pair mean distance: %.3f" % (sum(ref) / len(ref)))

# the sum-based scheme erroneously described in the R1 text: PE depends on x+y
import math

def sum_pe(x, y, dim=DIM):
    s = (x + y) / (H + W - 2) * 2 - 1
    freqs = torch.exp(torch.arange(dim // 2, dtype=torch.float32) * -math.log(10000) / (dim // 2 - 1))
    ang = s * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)])

sum_dists = [torch.norm(sum_pe(x1, y1) - sum_pe(x2, y2)).item() for (x1, y1), (x2, y2) in pairs]
print("sum-based PE (erroneous scheme), same-sum pair distances:", [f"{d:.6f}" for d in sum_dists])
print("=> separable implementation keeps same-sum positions distinct; the sum scheme collapses them.")
