"""E2: steps as shared randomness. Masked diffusion on toy data, k-step samplers evaluated exactly.

A k-step sampler unmasks a uniformly random ordered partition of the L positions into k blocks of
size L/k; tokens inside a block are sampled independently given the tokens already committed. Its
exact law is  p_k(x) = mean over ordered partitions of prod_s prod_{i in S_s} q(x_i | x_{S_<s}).
We evaluate it for the trained model and for the exact conditionals of p (oracle).
"""
import itertools
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
from model import MLP  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "e2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MASK = D.V
SUBSETS = range(2 ** D.L)  # bitmask U of unmasked positions


def masked_input(x, unmasked):
    """x (B, L), unmasked (B, L) bool -> (B, L*(V+1)) one-hot with MASK token elsewhere."""
    tok = torch.where(unmasked, x, torch.full_like(x, MASK))
    return torch.nn.functional.one_hot(tok, D.V + 1).flatten(1).float()


def bits(U):
    return torch.tensor([(U >> i) & 1 for i in range(D.L)], dtype=torch.bool)


def train_mdm(p, seed, steps=6000, batch=4096, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP(D.L * (D.V + 1)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    S = D.STATES.to(DEV)
    for _ in range(steps):
        x = S[torch.multinomial(p, batch, replacement=True)]
        t = torch.rand(batch, 1, device=DEV)
        unmasked = torch.rand(batch, D.L, device=DEV) > t
        ce = torch.nn.functional.cross_entropy(model(masked_input(x, unmasked)).transpose(1, 2), x,
                                               reduction="none")
        loss = (ce * ~unmasked).sum() / (~unmasked).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def model_cond(model):
    """(2^L, S, L) log q(x_i | x_U) for every unmasked set U, state x, position i."""
    S = D.STATES.to(DEV)
    out = []
    for U in SUBSETS:
        um = bits(U).to(DEV).expand(len(S), -1)
        logp = model(masked_input(S, um)).double().log_softmax(-1)
        out.append(logp.gather(-1, S[..., None])[..., 0])
    return torch.stack(out)


def oracle_cond(p):
    """Same as model_cond but with the exact conditionals of p."""
    pt = p.reshape([D.V] * D.L)

    def marg(keep):  # p(x_keep), broadcast back to every state
        drop = [j for j in range(D.L) if j not in keep]
        m = pt.sum(drop, keepdim=True) if drop else pt
        return m.expand_as(pt).reshape(-1)

    out = torch.zeros(2 ** D.L, len(p), D.L, dtype=D.DT, device=p.device)
    for U in SUBSETS:
        keep = [j for j in range(D.L) if (U >> j) & 1]
        den = marg(keep)
        for i in range(D.L):
            if i not in keep:
                q = marg(keep + [i]) / den
                out[U, :, i] = torch.where(den > 0, q, 1 / D.V).log()  # p(x_U)=0: never reached
    return out


def ordered_partitions(k):
    """All ordered partitions of range(L) into k blocks of size L/k (as lists of bitmasks)."""
    size, seen = D.L // k, set()
    for perm in itertools.permutations(range(D.L)):
        blocks = tuple(sum(1 << j for j in perm[s * size:(s + 1) * size]) for s in range(k))
        seen.add(blocks)
    return sorted(seen)


def k_step_logprob(cond, k):
    """Exact log p_k(x) of the k-step sampler driven by conditionals `cond`."""
    terms = []
    for blocks in ordered_partitions(k):
        lp, U = 0, 0
        for B in blocks:
            lp = lp + cond[U][:, bits(B).to(cond.device)].sum(-1)
            U |= B
        terms.append(lp)
    return torch.stack(terms).logsumexp(0) - math.log(len(terms))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ks = [1, 2, 3, 6]
    for fam in ["markov", "mix16", "copy"]:
        p = D.make(fam).to(DEV)
        orc = oracle_cond(p)
        for seed in range(3):
            out = OUT / f"{fam}_s{seed}.json"
            if out.exists():
                continue
            t0 = time.time()
            cond = model_cond(train_mdm(p, seed))
            rows = []
            for k in ks:
                lm, lo = k_step_logprob(cond, k), k_step_logprob(orc, k)
                rows.append({"k": k, "kl_model": D.kl(p, lm), "kl_oracle": D.kl(p, lo),
                             "tv_model": D.tv(p, lm.exp()), "tc_model": D.total_correlation(lm.exp())})
                print(fam, seed, rows[-1], flush=True)
            out.write_text(json.dumps({"family": fam, "seed": seed, "tc": D.total_correlation(p),
                                       "rows": rows, "wall_time": time.time() - t0}, indent=1))


if __name__ == "__main__":
    main()
