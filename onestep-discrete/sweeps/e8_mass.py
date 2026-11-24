"""E8 (added): can the generator move probability mass between modes?

Data zipf16: 16 deterministic codewords with Zipf weights. A finite source with K equal atoms can
only give each codeword a multiple of 1/K, and that allocation is piecewise constant in the
parameters (zero gradient). We compare the trained KL with the best allocation in which every
atom commits to one codeword (not a strict bound: atoms that spread over codewords can leak a little),
  floor(K) = min_{n_c >= 1, sum n_c = K} sum_c w_c log(w_c K / n_c)   (exact, greedy),
and with Gaussian sources whose mode masses are differentiable functions of the parameters.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
from run import RESULTS, Config, run  # noqa: E402

DATA = "zipf16"


def allocation_floor(w, K):
    """Greedy is exact: the objective is separable and convex in the integer counts n_c."""
    n = torch.ones_like(w)
    for _ in range(K - len(w)):
        gain = w * (torch.log(n + 1) - torch.log(n))
        n[gain.argmax()] += 1
    return (w * torch.log(w * K / n)).sum().item()


def configs():
    out = []
    for seed in range(3):
        for K in [16, 64, 256, 1024, 4096]:
            out.append(Config(exp="e8", data=DATA, source="categorical", K=K, enumerate=True, seed=seed,
                              steps=16000, eval_every=1000))
        for d in [4, 6, 8, 10, 12]:
            out.append(Config(exp="e8", data=DATA, source="rademacher", d=d, enumerate=True, seed=seed,
                              steps=16000, eval_every=1000))
        for d in [2, 4, 8]:
            out.append(Config(exp="e8", data=DATA, source="gaussian", d=d, N=1024, seed=seed,
                              steps=16000, eval_every=1000))
    return out


if __name__ == "__main__":
    p = D.make(DATA)
    w = p[p > 0].sort(descending=True).values
    floors = {K: allocation_floor(w, K) for K in [16, 64, 256, 1024, 4096]}
    (RESULTS / "e8").mkdir(parents=True, exist_ok=True)
    (RESULTS / "e8" / "floors.json").write_text(json.dumps(floors, indent=1))
    print("floors", floors, flush=True)
    for c in configs():
        r = run(c)
        print(r["name"], f"H={r['H_bits']} kl={r['final']['kl']:.4f}", flush=True)
