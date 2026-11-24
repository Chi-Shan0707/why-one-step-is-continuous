"""E3: source sweep. Same generator and objective; only P_Z changes. x-axis: H(Z) in bits.

    python sweeps/e3_sources.py --data mix16 --n_train 0 --worker 0 --nworkers 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import Config, run  # noqa: E402

H_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 32]


def source_grid():
    """(source, d, m, K) for every discrete source / entropy, plus continuous sources by d."""
    out = []
    for H in H_GRID:
        out.append(("rademacher", H, 2, 0))
        for m, b in [(4, 2), (16, 4), (256, 8)]:
            if H % b == 0:
                out.append(("quantized", H // b, m, 0))
                if m != 256:
                    out.append(("onehot", H // b, m, 0))
        if H <= 16:
            out.append(("categorical", 1, 2, 2 ** H))
    out += [("shuffled", d, 16, 0) for d in (1, 2)]
    out += [(c, d, 2, 0) for c in ("gaussian", "uniform") for d in (1, 2, 4, 8, 16, 32)]
    return out


def configs(data, n_train, seeds, steps):
    return [Config(exp="e3", data=data, n_train=n_train, source=s, d=d, m=m, K=K, seed=seed,
                   steps=steps)
            for seed in seeds for (s, d, m, K) in source_grid()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mix16")
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    cfgs = configs(a.data, a.n_train, a.seeds, a.steps)[a.worker::a.nworkers]
    for i, c in enumerate(cfgs):
        r = run(c)
        print(f"[{i + 1}/{len(cfgs)}] {r['name']} H={r['H_bits']} kl={r['final']['kl']:.4f} "
              f"best={r['best_kl']:.4f} {r['wall_time']:.0f}s", flush=True)
