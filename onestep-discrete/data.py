"""Toy data families over V^L states, with exact probabilities.

All distributions are float64 vectors of length V**L, indexed like `all_states()`.
Entropies / KL / TC are in nats unless a name says `_bits`.
"""
import itertools
import math

import torch

V, L = 4, 6
DT = torch.float64


def all_states(V=V, L=L):
    """(V**L, L) long tensor of every sequence, in itertools.product order."""
    return torch.tensor(list(itertools.product(range(V), repeat=L)), dtype=torch.long)


STATES = all_states()


def product_log_probs(theta, states=STATES):
    """theta: (C, L, V) per-component categoricals -> (C, V**L) log p(x | c)."""
    logt = theta.log()
    pos = torch.arange(states.shape[1], device=states.device)
    return logt[:, pos, states].sum(-1)


def mixture_of_products(r, alpha=0.2, seed=0):
    """Uniform mixture of r random product distributions (rows ~ Dirichlet(alpha))."""
    g = torch.Generator().manual_seed(seed)
    gam = torch._standard_gamma(torch.full((r, L, V), alpha, dtype=DT), generator=g)
    theta = gam / gam.sum(-1, keepdim=True)
    return product_log_probs(theta).exp().mean(0)


def copy_dist():
    """x_1 = ... = x_L uniform over V."""
    p = (STATES == STATES[:, :1]).all(1).to(DT)
    return p / p.sum()


def markov_chain(alpha=0.5, seed=0):
    """Uniform x_1, then x_{i+1} | x_i ~ T[x_i], rows of T ~ Dirichlet(alpha)."""
    g = torch.Generator().manual_seed(seed)
    gam = torch._standard_gamma(torch.full((V, V), alpha, dtype=DT), generator=g)
    T = gam / gam.sum(-1, keepdim=True)
    logp = torch.full((len(STATES),), -math.log(V), dtype=DT)
    for i in range(L - 1):
        logp = logp + T.log()[STATES[:, i], STATES[:, i + 1]]
    return logp.exp()


def codebook(r, seed=0, zipf=False):
    """r random codewords of the parity code sum(x) = 0 mod V (min Hamming distance 2), uniform or
    with Zipf weights 1/rank. Distance 2 makes the nonnegative rank exactly r: a product component
    covering two codewords would also cover a non-codeword at distance 1 from one of them."""
    g = torch.Generator().manual_seed(seed)
    code = torch.nonzero(STATES.sum(1) % V == 0)[:, 0]
    w = 1 / torch.arange(1, r + 1, dtype=DT) if zipf else torch.ones(r, dtype=DT)
    p = torch.zeros(len(STATES), dtype=DT)
    p[code[torch.randperm(len(code), generator=g)[:r]]] = w / w.sum()
    return p


def make(family, seed=0):
    """'mix<r>' (e.g. 'mix16'), 'code<r>', 'zipf<r>' (Zipf-weighted code), 'copy', or 'markov'."""
    if family.startswith("zipf"):
        return codebook(int(family[4:]), seed=seed, zipf=True)
    if family.startswith("code"):
        return codebook(int(family[4:]), seed=seed)
    if family.startswith("mix"):
        return mixture_of_products(int(family[3:]), seed=seed)
    if family == "copy":
        return copy_dist()
    if family == "markov":
        return markov_chain(seed=seed)
    raise ValueError(family)


def empirical(p, n, seed):
    """Empirical distribution of n iid samples from p."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.multinomial(p, n, replacement=True, generator=g)
    return torch.bincount(idx, minlength=len(p)).to(DT) / n


# ---------- exact information quantities ----------

def entropy(p):
    p = p[p > 0]
    return -(p * p.log()).sum().item()


def marginals(p, states=STATES):
    """(L, V) single-site marginals."""
    m = torch.zeros(states.shape[1], V, dtype=DT, device=p.device)
    for i in range(states.shape[1]):
        m[i].index_add_(0, states[:, i].to(p.device), p)
    return m


def total_correlation(p):
    """TC(p) = sum_i H(p_i) - H(p) = KL(p || product of its marginals)."""
    return sum(entropy(mi) for mi in marginals(p)) - entropy(p)


def kl(p, log_q):
    """KL(p || q) with q given by log-probabilities."""
    m = p > 0
    return (p[m] * (p[m].log() - log_q[m])).sum().item()


def tv(p, q):
    return 0.5 * (p - q).abs().sum().item()
