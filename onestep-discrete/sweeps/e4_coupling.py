"""E4: coupling quality. Fit G by supervised CE on fixed (z, x) pairs; test on held-out z.

Sources (what G sees):
  gauss   z ~ N(0, I_24); held-out z are fresh draws; deployed law uses fresh z (MC).
  code    K = 4096 symbols = strings s in {0..3}^6, fed as per-digit one-hot (factorial geometry).
  onehot  K = 4096 symbols fed as a plain one-hot (no geometry at all).
  For code/onehot, held-out z are unseen symbols and the deployed law averages all K symbols.
Couplings (how targets are attached to z):
  random  z_j paired with an independent sample x_j ~ p.
  ot      optimal assignment between the M source points and M samples of p:
          gauss: squared Euclidean to one-hot(x);  code/onehot: Hamming(s, x) (plan's "Hamming OT").
  ode     gauss only: x = argmax of the ODE of a one-hot flow-matching model started at z.
  vae     gauss only: z ~ q(z | x) of a VAE with factorized decoder (Coupling Models Stage A, simplified).
Knothe-Rosenblatt cells (coupling `kr`: x_i = inverse CDF of p(x_i | x_<i) at u_i, u in [0,1]^6):
  unif    u ~ U[0,1]^6 continuous.
  quant   u on a grid of Q levels per dim (ordered discrete source).
  shuf    the same grid and the same coupling, but G sees the levels permuted per dim. The source
          law is identical to `quant`; only the alignment between source order and coupling differs.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
from exact import log_marginal, metrics  # noqa: E402
from losses import regression_ce  # noqa: E402
from model import MLP  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "e4"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DZ, M, FRAC_TRAIN, Q = D.L * D.V, 4096, 0.8, 64
CELLS = [("gauss", "random"), ("gauss", "ot"), ("gauss", "ode"), ("gauss", "vae"),
         ("code", "random"), ("code", "ot"), ("onehot", "random"), ("onehot", "ot"),
         ("unif", "kr"), ("quant", "kr"), ("shuf", "kr")]
IN_DIM = {"gauss": DZ, "code": DZ, "onehot": M, "unif": D.L, "quant": D.L, "shuf": D.L}
SHUF = torch.stack([torch.randperm(Q, generator=torch.Generator().manual_seed(i)) for i in range(D.L)])


def onehot_x(x):
    return torch.nn.functional.one_hot(x, D.V).flatten(1).float()


def fit(model, inputs, targets, steps, lr=1e-3, batch=1024, loss_fn=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(len(inputs), (batch,), device=DEV)
        loss = loss_fn(model, inputs[idx], targets[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model


# ---------- Gaussian couplings that need a model ----------

def fm_loss(net, x1, _):
    """Flow matching from N(0, I) to one-hot(x) on the linear path; velocity MLP on [x_t, t]."""
    z = torch.randn_like(x1)
    t = torch.rand(len(x1), 1, device=DEV)
    return (net(torch.cat([(1 - t) * z + t * x1, t], 1))[:, 0] - (x1 - z)).pow(2).mean()


def train_fm(p, seed, steps=10000):
    torch.manual_seed(seed)
    x1 = onehot_x(D.STATES.to(DEV)[torch.multinomial(p, 2**18, replacement=True)])
    return fit(MLP(DZ + 1, L=1, V=DZ, width=512).to(DEV), x1, x1, steps, loss_fn=fm_loss)


@torch.no_grad()
def ode_decode(net, z, n_steps=100):
    """Euler ODE from t=0 (noise) to t=1, then per-position argmax."""
    x = z.clone()
    for k in range(n_steps):
        t = torch.full((len(x), 1), k / n_steps, device=DEV)
        x = x + net(torch.cat([x, t], 1))[:, 0] / n_steps
    return x.view(-1, D.L, D.V).argmax(-1)


class VAE(torch.nn.Module):
    """Encoder q(z|x) = N(mu, sigma^2) in R^24 and a factorized decoder."""

    def __init__(self):
        super().__init__()
        self.enc, self.dec = MLP(DZ, L=1, V=2 * DZ), MLP(DZ)

    def encode(self, x):
        return self.enc(onehot_x(x))[:, 0].chunk(2, -1)  # mu, log sigma


def vae_loss(vae, x, _):
    mu, logs = vae.encode(x)
    z = mu + logs.exp() * torch.randn_like(mu)
    rec = torch.nn.functional.cross_entropy(vae.dec(z).transpose(1, 2), x, reduction="none").sum(1)
    klz = 0.5 * (mu.pow(2) + (2 * logs).exp() - 2 * logs - 1).sum(1)
    return (rec + klz).mean()


def train_vae(p, seed, steps=10000):
    torch.manual_seed(seed)
    x = D.STATES.to(DEV)[torch.multinomial(p, 2**18, replacement=True)]
    return fit(VAE().to(DEV), x, x, steps, loss_fn=vae_loss)


# ---------- building pairs ----------

def all_codes():
    return D.STATES.to(DEV)  # symbol k <-> string s_k in {0..3}^6


def grid_u(idx):
    return (idx + 0.5) / Q


def source_input(src, z):
    """What G sees. z: gauss/unif floats, code/onehot symbol ids, quant/shuf (n, L) level ids."""
    if src == "gauss":
        return z
    if src == "code":
        return onehot_x(all_codes()[z])
    if src == "onehot":
        return torch.nn.functional.one_hot(z, M).float()
    if src == "shuf":
        z = SHUF.to(DEV).T[z, torch.arange(D.L, device=DEV)]  # level j of dim i -> SHUF[i, j]
    u = z if src == "unif" else grid_u(z)
    return (u - 0.5) * 12 ** 0.5


def sample_z(src, n, g):
    if src == "gauss":
        return torch.randn(n, DZ, generator=g, device=DEV)
    if src == "unif":
        return torch.rand(n, D.L, generator=g, device=DEV)
    return torch.randint(Q, (n, D.L), generator=g, device=DEV)


def kr_decode(p, u):
    """Knothe-Rosenblatt / autoregressive inverse-CDF map [0,1]^L -> V^L for the exact p."""
    pt, x = p.reshape([D.V] * D.L), torch.zeros(len(u), D.L, dtype=torch.long, device=DEV)
    for i in range(D.L):
        m = pt.sum(list(range(i + 1, D.L))) if i < D.L - 1 else pt  # p(x_<=i), shape V^(i+1)
        cond = m[tuple(x[:, j] for j in range(i))]                   # (n, V) unnormalized
        cdf = (cond / cond.sum(-1, keepdim=True)).cumsum(-1)
        xi = torch.searchsorted(cdf, u[:, i:i + 1].double(), right=True)[:, 0]
        last = D.V - 1 - (cond.flip(-1) > 0).long().argmax(-1)       # last token with mass
        x[:, i] = torch.minimum(xi, last)
    return x


def assign(cost):
    """Row i gets column perm[i] (min-cost perfect matching)."""
    _, perm = linear_sum_assignment(cost.double().cpu().numpy())
    return torch.as_tensor(perm, device=DEV)


def make_pairs(src, coupling, p, seed, cache):
    """Returns (z_train, x_train, z_test, x_test)."""
    g = torch.Generator(DEV).manual_seed(seed)
    S = D.STATES.to(DEV)
    x = S[torch.multinomial(p, M, replacement=True, generator=g)]
    if src == "gauss":
        z = sample_z(src, M, g)
        if coupling == "ot":
            x = x[assign(torch.cdist(z, onehot_x(x)).pow(2))]
        elif coupling == "ode":
            x = ode_decode(cache["fm"], z)
        elif coupling == "vae":
            with torch.no_grad():
                mu, logs = cache["vae"].encode(x)
                z = mu + logs.exp() * torch.randn(mu.shape, generator=g, device=DEV)
    elif coupling == "kr":
        z = sample_z(src, M, g)
        x = kr_decode(p, z if src == "unif" else grid_u(z))
    else:
        z = torch.randperm(M, generator=g, device=DEV)  # symbol indices
        if coupling == "ot":
            ham = (all_codes()[z][:, None, :] != x[None, :, :]).sum(-1)
            x = x[assign(ham)]
    perm = torch.randperm(M, generator=g, device=DEV)  # matched pairs are not exchangeable (LSA order)
    z, x, n_tr = z[perm], x[perm], int(FRAC_TRAIN * M)
    return z[:n_tr], x[:n_tr], z[n_tr:], x[n_tr:]


@torch.no_grad()
def evaluate(G, src, p, z_te, x_te, z_tr, x_tr, seed):
    def acc(z, x):
        pred = G(source_input(src, z)).argmax(-1)
        return (pred == x).float().mean().item(), (pred == x).all(1).float().mean().item()

    out = {}
    out["train_tok_acc"], out["train_seq_acc"] = acc(z_tr, x_tr)
    out["test_tok_acc"], out["test_seq_acc"] = acc(z_te, x_te)
    out["test_ce"] = regression_ce(G(source_input(src, z_te)), x_te).item()
    if src in ("code", "onehot"):
        # main metric: law over held-out symbols only; also report the law over all K symbols,
        # 80% of which are memorized training inputs
        out.update(metrics(p, log_marginal(G, source_input(src, z_te))))
        out["kl_all_symbols"] = metrics(p, log_marginal(G, source_input(src, torch.arange(M, device=DEV))))["kl"]
    else:  # fresh z: all held out
        z_dep = source_input(src, sample_z(src, 2**16, torch.Generator(DEV).manual_seed(10_000 + seed)))
        out.update(metrics(p, log_marginal(G, z_dep)))
    return out


def target_tv(p, x):
    emp = torch.bincount((x * (D.V ** torch.arange(D.L - 1, -1, -1, device=DEV))).sum(1),
                         minlength=len(p)).double() / len(x)
    return D.tv(p, emp)


def main(families=("mix16", "markov"), seeds=(0, 1, 2), g_steps=4000):
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in families:
        p = D.make(fam).to(DEV)
        for seed in seeds:
            cache = {}
            for src, coupling in CELLS:
                out = OUT / f"{fam}_{src}_{coupling}_s{seed}.json"
                if out.exists():
                    continue
                t0 = time.time()
                if coupling == "ode" and "fm" not in cache:
                    cache["fm"] = train_fm(p, seed)
                if coupling == "vae" and "vae" not in cache:
                    cache["vae"] = train_vae(p, seed)
                z_tr, x_tr, z_te, x_te = make_pairs(src, coupling, p, seed, cache)
                torch.manual_seed(seed)
                G = MLP(IN_DIM[src]).to(DEV)
                fit(G, source_input(src, z_tr), x_tr, g_steps,
                    loss_fn=lambda G, z, x: regression_ce(G(z), x))
                res = {"family": fam, "source": src, "coupling": coupling, "seed": seed,
                       "tc": D.total_correlation(p),
                       "target_tv": target_tv(p, torch.cat([x_tr, x_te])),
                       **evaluate(G, src, p, z_te, x_te, z_tr, x_tr, seed),
                       "wall_time": time.time() - t0}
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
                out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
