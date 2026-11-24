"""log p(x | z) for every state, and the exact (or large-N MC) marginal of a one-step generator."""
import math

import torch

from data import STATES, V, kl, tv, total_correlation


def onehot_states(states=STATES, V=V):
    """(S, L*V) one-hot rows, so that log p(x | z) = logp(z).flatten() @ onehot(x)."""
    return torch.nn.functional.one_hot(states, V).flatten(1)


def log_px_given_z(logits, onehot):
    """logits (N, L, V), onehot (S, L*V) -> (N, S) log prod_i p(x_i | z)."""
    logp = logits.log_softmax(-1).flatten(1)
    return logp @ onehot.to(logp.dtype).T


@torch.no_grad()
def log_marginal(model, z, chunk=4096):
    """log p_theta(x) = log mean_n prod_i p(x_i | z_n) over the given z (uniform weights), (S,) float64."""
    oh = onehot_states().to(z.device)
    acc = None
    for zc in z.split(chunk):
        lp = log_px_given_z(model(zc).double(), oh).logsumexp(0)
        acc = lp if acc is None else torch.logaddexp(acc, lp)
    return acc - math.log(len(z))


@torch.no_grad()
def eval_z(source, n_mc, gen, device, enum_cap=2**16):
    """Atoms of the source if it is finite with K <= enum_cap (exact), else n_mc MC samples."""
    if source.finite and source.K <= enum_cap:
        return source.atoms().to(device), True
    return source.sample(n_mc, gen, device), False


def metrics(p, log_q):
    """Exact KL(p||q), TV, and the model's own TC, all in nats.

    KL splits exactly as kl = kl_alloc + leak_nats: kl_alloc = KL(p || q restricted to supp p and
    renormalized) is how mass is shared among the data states; leak_nats = -log q(supp p) is the
    mass lost to states outside the support (both are 0 / kl for full-support p)."""
    p = p.to(log_q.device)
    q = log_q.exp()
    log_in = log_q[p > 0].logsumexp(0)
    return {"kl": kl(p, log_q), "tv": tv(p, q), "tc_model": total_correlation(q / q.sum()),
            "kl_alloc": kl(p, log_q - log_in), "leak_nats": -log_in.item()}
