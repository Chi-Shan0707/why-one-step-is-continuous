"""E5 (toy version of the plan's E5): do correlated positions flip together when z crosses a boundary?

For trained one-step generators we move z a little and record which positions change their argmax:
  gaussian d=4:     spherical interpolation z(t) = cos(pi t/2) z0 + sin(pi t/2) z1, T steps
  rademacher d=8:   single-bit flips of a random corner
lift_ij = P(j flips | i flips at the same move) / P(j flips). H3 predicts lift_ij > 1 and that
lift_ij grows with the mutual information I(x_i; x_j) of the data.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
from model import MLP  # noqa: E402
from run import RESULTS, Config, run  # noqa: E402
from sources import make_source  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = RESULTS / "e5"


def pairwise_mi(p):
    """(L, L) mutual information I(x_i; x_j) in nats (diagonal = 0)."""
    S = D.STATES.to(p.device)
    mi = torch.zeros(D.L, D.L, dtype=D.DT)
    for i in range(D.L):
        for j in range(i + 1, D.L):
            pij = torch.zeros(D.V * D.V, dtype=D.DT, device=p.device).index_add_(0, S[:, i] * D.V + S[:, j], p)
            pij = pij.view(D.V, D.V)
            ind = pij.sum(1, keepdim=True) * pij.sum(0, keepdim=True)
            m = pij > 0
            mi[i, j] = mi[j, i] = (pij[m] * (pij[m] / ind[m]).log()).sum().item()
    return mi


@torch.no_grad()
def flips(model, kind, d, n=1000, T=256, seed=0):
    """(moves, L) bool: which positions change argmax in each small move of z."""
    g = torch.Generator(DEV).manual_seed(seed)
    if kind == "gaussian":
        z0, z1 = torch.randn(2, n, d, generator=g, device=DEV)
        t = torch.linspace(0, 1, T, device=DEV)[:, None, None] * math.pi / 2
        a = model((t.cos() * z0 + t.sin() * z1).reshape(-1, d)).argmax(-1).view(T, n, D.L)
        return (a[1:] != a[:-1]).reshape(-1, D.L)
    z = (torch.randint(2, (n, d), generator=g, device=DEV) * 2 - 1).float()
    a0 = model(z).argmax(-1)
    zf = z[:, None, :].repeat(1, d, 1)
    zf[:, torch.arange(d), torch.arange(d)] *= -1
    return (model(zf.view(-1, d)).argmax(-1).view(n, d, D.L) != a0[:, None]).reshape(-1, D.L)


def lift(F):
    F = F.double()
    rate = F.mean(0)
    joint = (F.T @ F) / len(F)
    out = joint / (rate[:, None] * rate[None, :]).clamp_min(1e-12)
    out.fill_diagonal_(float("nan"))
    return out.cpu(), rate.cpu()


def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    iu = torch.triu_indices(D.L, D.L, 1)
    for fam in ["markov", "mix16"]:
        p = D.make(fam).to(DEV)
        mi = pairwise_mi(p)
        for kind, d in [("gaussian", 4), ("rademacher", 8)]:
            for seed in range(3):
                cfg = Config(exp="e5", data=fam, source=kind, d=d, seed=seed, save_model=True)
                res = run(cfg)
                model = MLP(make_source(kind, d).dim).to(DEV)
                model.load_state_dict(torch.load(RESULTS / "e5" / f"{cfg.name()}.pt"))
                model.eval()
                L_, rate = lift(flips(model, kind, d, seed=seed))
                lv, mv = L_[iu[0], iu[1]], mi[iu[0], iu[1]]
                dist = (iu[1] - iu[0]).double()
                row = {"family": fam, "source": kind, "d": d, "seed": seed, "kl": res["final"]["kl"],
                       "lift": L_.tolist(), "mi": mi.tolist(), "flip_rate": rate.tolist(),
                       "spearman_lift_mi": spearman(lv, mv),
                       "mean_lift_adjacent": lv[dist == 1].mean().item(),
                       "mean_lift_distant": lv[dist >= 3].mean().item()}
                print(fam, kind, seed, {k: round(v, 3) for k, v in row.items()
                                        if isinstance(v, float)}, flush=True)
                (OUT / f"{fam}_{kind}_s{seed}_flips.json").write_text(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
