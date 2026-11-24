"""E7b (added after review): the coupling of a Discrete Flow Matching teacher, computed exactly.

E7 uses the time-reversed noising chain, whose coupling with its prior is independent. A DFM teacher
(Campbell et al. 2024, Gat et al. 2024) instead runs the *marginal* CTMC of the mixture path
    p_t(x) = sum_{x1} p(x1) prod_i [ t 1[x_i = x1_i] + (1 - t)/V ],   X_0 ~ Uniform(V^L),
with rate x -> x' (x'_i = y != x_i, other positions equal) of P(X1_i = y | X_t = x) / (1 - t).
It has the same marginals but a different joint law of (X_0, X_1). We integrate the forward
Kolmogorov equation for the joint P(X_0 = z, X_t = x) over all 4096 x 4096 pairs. In time
s = -log(1 - t) the rates are just the posteriors, so Euler steps in s are well conditioned.
Then the regression-optimal one-step student is analysed exactly as in E7.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
from e7_teacher_coupling import OUT, onehot, student_from_joint  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log_joint_t(p, t, ohx):
    """(x_t, x1) log p(x1) prod_i k_t(x_t,i | x1_i) of the mixture path at time t < 1."""
    logk = torch.full((D.V, D.V), math.log((1 - t) / D.V), dtype=D.DT, device=DEV)
    logk.fill_diagonal_(math.log(t + (1 - t) / D.V))
    return ohx @ torch.block_diag(*[logk] * D.L) @ ohx.T + p.clamp_min(1e-300).log()[None]


def posterior_marginals(p, t, ohx):
    """(S, L, V): P(X1_i = y | X_t = x) under the mixture path at time t."""
    logj = log_joint_t(p, t, ohx)
    post = (logj - logj.logsumexp(1, keepdim=True)).exp()
    return (post @ ohx).view(-1, D.L, D.V)


def dfm_joint(p, s_max=12.0, ds=0.0025, check_every=20):
    """Joint P(X_0, X_T) of the marginal CTMC, T = 1 - exp(-s_max). Returns it and a marginal check."""
    S = len(p)
    ohx = onehot(D.STATES.to(DEV), D.V)
    P = torch.eye(S, dtype=D.DT, device=DEV) / S                       # X_0 uniform, X_0 = X_t at t=0
    shape = [S] + [D.V] * D.L
    n_steps, worst_tv = int(round(s_max / ds)), 0.0
    for k in range(n_steps):
        t = 1 - math.exp(-k * ds)
        r = posterior_marginals(p, t, ohx)                            # rate * dt / ds
        dP = torch.zeros_like(P)
        for i in range(D.L):
            ri = r[:, i, :].reshape([D.V] * D.L + [D.V])              # [x..., y]
            ri = ri * (1 - torch.nn.functional.one_hot(D.STATES[:, i].to(DEV), D.V)
                       .to(D.DT).reshape([D.V] * D.L + [D.V]))       # no self-jumps
            G = P.view(shape)[..., None] * ri[None]                   # flow x -> (x_i := y)
            out = G.sum(-1)
            inflow = G.movedim(1 + i, -1).sum(-1).movedim(-1, 1 + i)  # sum over old value v
            dP += (inflow - out).reshape(S, S)
        P = P + ds * dP
        if k % check_every == 0:  # the marginal of X_t must equal p_t
            pt = log_joint_t(p, 1 - math.exp(-(k + 1) * ds), ohx).logsumexp(1).exp()  # time after the step
            worst_tv = max(worst_tv, D.tv(pt, P.sum(0)))
    return P, worst_tv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in ["mix16", "markov", "copy"]:
        out = OUT / f"{fam}_dfm.json"
        if out.exists():
            continue
        p = D.make(fam).to(DEV)
        P, worst_tv = dfm_joint(p)
        kl, mi, hz = student_from_joint(p, P.clamp_min(1e-300).log())
        P2, _ = dfm_joint(p, ds=0.005, check_every=10**9)  # coarser step: Euler error estimate
        kl2, mi2, _ = student_from_joint(p, P2.clamp_min(1e-300).log())
        res = {"family": fam, "tc": D.total_correlation(p), "H_bits": D.entropy(p) / math.log(2),
               "student_kl": kl, "I_bits": mi, "HZ_bits": hz, "mass": P.sum().item(),
               "final_tv_to_p": D.tv(p, P.sum(0)), "max_tv_marginal_vs_pt": worst_tv,
               "euler_diff_kl": kl2 - kl, "euler_diff_I_bits": mi2 - mi}  # ds vs 2*ds
        print(res, flush=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
