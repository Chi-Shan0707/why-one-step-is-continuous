"""E7 (added): what a discrete-diffusion teacher's coupling can give a one-step student. Exact.

The reverse process of a Markov noising chain realizes the same joint law as the forward chain, so
a multi-step discrete teacher couples its starting noise z and its output x by the forward kernel
    pi(z, x) = p(x) prod_i k_alpha(z_i | x_i),
  uniform: k(z|x) = alpha 1[z=x] + (1-alpha)/V        (z in V^L, prior = Uniform at alpha=0)
  mask:    k(z|x) = alpha 1[z=x] + (1-alpha) 1[z=MASK] (z in (V+1)^L, prior = all-mask at alpha=0)
A one-step student fit by regression (CE) on infinitely many pairs learns q(x_i | z) = pi(x_i | z),
and generates p_G(x) = sum_z pi_Z(z) prod_i q(x_i | z). We report KL(p || p_G), I(Z;X) and H(Z).
"""
import itertools
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "e7"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = -1e4  # stands in for log 0 inside matmuls


def onehot(states, n):
    return torch.nn.functional.one_hot(states, n).flatten(1).to(D.DT)


def forward_joint(p, logK, Vz):
    """(Z, X) log pi(z, x) = log p(x) + sum_i log k(z_i | x_i); logK (V, Vz)."""
    X = D.STATES.to(DEV)
    Z = torch.tensor(list(itertools.product(range(Vz), repeat=D.L)), device=DEV)
    W = torch.block_diag(*[logK.clamp_min(NEG)] * D.L)                 # (L*V, L*Vz)
    return onehot(Z, Vz) @ W.T @ onehot(X, D.V).T + p.clamp_min(1e-300).log()[None]


def student_from_joint(p, logj):
    """Regression-optimal one-step student for the coupling exp(logj) (Z, X).

    Returns KL(p || p_G), I(Z;X) bits and H(Z) bits, where p_G = sum_z pi(z) prod_i pi(x_i | z)."""
    ohx = onehot(D.STATES.to(DEV), D.V)
    logpz, logpx = logj.logsumexp(1), logj.logsumexp(0)
    post = (logj - logpz[:, None]).exp()                              # pi(x | z)
    logq = (post @ ohx).clamp_min(1e-300).log()                       # (Z, L*V) log pi(x_i | z)
    logpg = (logq @ ohx.T + logpz[:, None]).logsumexp(0)
    mi = (logj.exp() * (logj - logpz[:, None] - logpx[None])).nansum().item()
    hz = -(logpz.exp() * logpz).nansum().item()
    return D.kl(p, logpg), mi / math.log(2), hz / math.log(2)


def student_law(p, logK, Vz):
    return student_from_joint(p, forward_joint(p, logK, Vz))


def kernel(kind, a):
    eye = torch.eye(D.V, dtype=D.DT, device=DEV)
    if kind == "uniform":
        return (a * eye + (1 - a) / D.V).log(), D.V
    K = torch.cat([a * eye, torch.full((D.V, 1), 1 - a, dtype=D.DT, device=DEV)], 1)
    return K.log(), D.V + 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alphas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    for fam in ["mix16", "markov", "copy"]:
        p = D.make(fam).to(DEV)
        rows = []
        for kind in ["uniform", "mask"]:
            for a in alphas:
                kl, mi, hz = student_law(p, *kernel(kind, a))
                rows.append({"kernel": kind, "alpha": a, "kl": kl, "I_bits": mi, "HZ_bits": hz})
                print(fam, rows[-1], flush=True)
        (OUT / f"{fam}.json").write_text(json.dumps(
            {"family": fam, "tc": D.total_correlation(p), "H_bits": D.entropy(p) / math.log(2),
             "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
