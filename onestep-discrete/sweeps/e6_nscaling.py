"""E6 (added): the MC marginal-likelihood objective needs N >~ 2^{I(X;Z)} source samples per step.

Data code<r> needs log2 r bits of shared randomness. We sweep r and the number N of source samples
per step, for a Gaussian source and a finite source with K = 4r atoms (sampled, and enumerated).

    python sweeps/e6_nscaling.py --worker 0 --nworkers 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import Config, run  # noqa: E402


def configs():
    out = []
    for r in [16, 64, 256, 1024]:
        for N in [16, 64, 256, 1024, 4096]:
            out.append(Config(exp="e6", data=f"code{r}", source="gaussian", d=8, N=N))
            out.append(Config(exp="e6", data=f"code{r}", source="categorical", K=4 * r, N=N))
        out.append(Config(exp="e6", data=f"code{r}", source="categorical", K=4 * r, enumerate=True))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    cfgs = configs()[a.worker::a.nworkers]
    for i, c in enumerate(cfgs):
        r = run(c)
        print(f"[{i + 1}/{len(cfgs)}] {r['name']} kl={r['final']['kl']:.4f} {r['wall_time']:.0f}s", flush=True)
