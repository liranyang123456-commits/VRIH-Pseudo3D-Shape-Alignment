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

# anti-diagonal pairs: (x, y) with x + y = 20
anti = [(i, 20 - i) for i in range(5, 15)]
dists = []
for (x1, y1), (x2, y2) in zip(anti[:-1], anti[1:]):
    d = torch.norm(codes[y1, x1] - codes[y2, x2]).item()
    dists.append(d)
print("anti-diagonal adjacent-pair encoding distances:", [f"{d:.3f}" for d in dists])
print("min anti-diagonal distance: %.3f" % min(dists))

# reference: distance between two far-apart random positions
import random

random.seed(0)
ref = []
for _ in range(20):
    a = (random.randrange(W), random.randrange(H))
    b = (random.randrange(W), random.randrange(H))
    ref.append(torch.norm(codes[a[1], a[0]] - codes[b[1], b[0]]).item())
print("random-pair mean distance: %.3f" % (sum(ref) / len(ref)))
print("collision-free:", min(dists) > 1e-3)
