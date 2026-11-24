"""E1: fit a k-component mixture of products (latent-class model) by population EM; sweep k.

Expect KL = TC(p) at k = 1, monotone decrease, and KL ~ 0 for k >= r on mix<r> data.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "e1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def em(p, k, iters, seed):
    """Population EM on exact p. Returns the final KL(p || mixture) in nats."""
    g = torch.Generator().manual_seed(seed)
    S = D.STATES.to(DEV)
    oh = torch.nn.functional.one_hot(S, D.V).double()                 # (S, L, V)
    theta = torch.rand(k, D.L, D.V, generator=g, dtype=D.DT).to(DEV) + 0.5
    theta /= theta.sum(-1, keepdim=True)
    logpi = torch.full((k,), -torch.log(torch.tensor(float(k))).item(), dtype=D.DT, device=DEV)
    for _ in range(iters):
        logj = logpi[:, None] + D.product_log_probs(theta, S)          # (k, S) log pi_c p(x|c)
        logq = logj.logsumexp(0)
        resp = (logj - logq).exp() * p                                  # p(x) * gamma(c|x)
        mass = resp.sum(1)
        logpi = mass.clamp_min(1e-300).log()
        theta = torch.einsum("cs,slv->clv", resp, oh) / mass.clamp_min(1e-300)[:, None, None]
        theta = theta.clamp_min(1e-300)
    logq = (logpi[:, None] + D.product_log_probs(theta, S)).logsumexp(0)
    return D.kl(p, logq)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    for fam in ["mix1", "mix2", "mix4", "mix16", "mix64", "copy", "markov"]:
        out = OUT / f"{fam}.json"
        if out.exists():
            continue
        p = D.make(fam).to(DEV)
        t0, rows = time.time(), []
        for k in ks:
            kls = [em(p, k, iters=1000, seed=s) for s in range(10)]
            rows.append({"k": k, "kl_best": min(kls), "kl_all": kls})
            print(fam, k, f"{min(kls):.5f}", flush=True)
        out.write_text(json.dumps({"family": fam, "tc": D.total_correlation(p), "H": D.entropy(p),
                                   "restarts": 10, "iters": 1000, "rows": rows,
                                   "wall_time": time.time() - t0}, indent=1))


if __name__ == "__main__":
    main()
