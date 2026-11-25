"""E8b (second round): is atomicity alone what freezes mass allocation?

Frozen Gaussian: K samples of N(0, I_4) drawn once and kept, vs the resampled Gaussian of E8. Both
use the same MC objective with N = 1024 fresh draws per step (from the K atoms, or from N(0, I)),
so geometry, law and gradient estimator match; the only difference is whether the source has atoms.
For K equal atoms under a deterministic map, the masses are multiples of 1/K and the best KL is
achievable(K) of E8 (rounding p to the 1/K grid). For soft outputs it is an upper bound on the best KL.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
from e8_mass import DATA, allocation_floor  # noqa: E402
from run import RESULTS, Config, run  # noqa: E402

KS = [64, 1024, 2**16]

if __name__ == "__main__":
    p = D.make(DATA)
    w = p[p > 0].sort(descending=True).values
    (RESULTS / "e8b").mkdir(parents=True, exist_ok=True)
    (RESULTS / "e8b" / "floors.json").write_text(json.dumps({K: allocation_floor(w, K) for K in KS}))
    for seed in range(3):
        for K in KS:
            r = run(Config(exp="e8b", data=DATA, source="frozen", d=4, K=K, N=1024, seed=seed,
                           steps=16000, eval_every=1000))
            print(r["name"], {k: round(v, 5) for k, v in r["final"].items()}, flush=True)
