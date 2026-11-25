"""Source distributions P_Z. All are standardized (≈ zero mean, unit variance per input coordinate).

A finite source is a product of `d` independent uniform factors; factor j picks one row of its
embedding table, and the rows of all factors are concatenated. This covers every discrete case:
  quantized (d, m)   d factors, table = m standardized Gaussian quantile levels (ordered grid)
  rademacher (d)     = quantized (d, 2)
  shuffled (d, m)    quantized with levels permuted per factor. NOTE: this is the same uniform
                     distribution over the same point set, so it is identical to `quantized` as a
                     source; it is kept only as a sanity check.
  onehot (d, m)      d factors, table = standardized one-hot of m levels (factorial, no order)
  categorical (K)    1 factor, table = K fixed random N(0, I_32) embeddings (no structure)
  constant           1 factor with a single all-zero row (deterministic source, K = 1)
  frozen (d, K)      1 factor, table = K fixed N(0, I_d) draws, not re-standardized: the Gaussian with
                     its atoms exposed
"""
import math
from statistics import NormalDist

import torch

CAT_DIM = 32


def gaussian_levels(m):
    """Conditional means of m equal-mass bins of N(0,1), rescaled to unit variance."""
    nd = NormalDist()
    edges = [-math.inf] + [nd.inv_cdf(k / m) for k in range(1, m)] + [math.inf]
    pdf = [0.0 if math.isinf(e) else nd.pdf(e) for e in edges]
    lv = torch.tensor([(pdf[k] - pdf[k + 1]) * m for k in range(m)], dtype=torch.float32)
    return lv / lv.pow(2).mean().sqrt()


class Source:
    def __init__(self, name, dim, tables=None, dist=None):
        self.name, self.dim = name, dim
        self.tables = tables  # list of (m_j, e_j) tensors for finite sources, else None
        self.dist = dist      # 'gaussian' | 'uniform' for continuous sources

    @property
    def finite(self):
        return self.tables is not None

    @property
    def K(self):
        return math.prod(t.shape[0] for t in self.tables) if self.finite else math.inf

    @property
    def entropy_bits(self):
        return sum(math.log2(t.shape[0]) for t in self.tables) if self.finite else math.inf

    def to(self, device):
        if self.finite:
            self.tables = [t.to(device) for t in self.tables]
        return self

    def sample(self, n, gen, device):
        if self.dist == "gaussian":
            return torch.randn(n, self.dim, generator=gen, device=device)
        if self.dist == "uniform":
            return (torch.rand(n, self.dim, generator=gen, device=device) * 2 - 1) * math.sqrt(3)
        return torch.cat([t[torch.randint(t.shape[0], (n,), generator=gen, device=device)]
                          for t in self.tables], 1)

    def atoms(self):
        """All K atoms (K, dim), each with probability 1/K. Only for finite sources."""
        z = self.tables[0]
        for t in self.tables[1:]:
            z = torch.cat([z.repeat_interleave(t.shape[0], 0), t.repeat(z.shape[0], 1)], 1)
        return z


def make_source(kind, d=1, m=2, K=2, seed=0):
    g = torch.Generator().manual_seed(1000 + seed)
    if kind in ("gaussian", "uniform"):
        return Source(f"{kind}_d{d}", d, dist=kind)
    if kind == "rademacher":
        kind, m = "quantized", 2
    if kind == "quantized":
        return Source(f"quantized_d{d}_m{m}", d, [gaussian_levels(m)[:, None]] * d)
    if kind == "shuffled":
        lv = gaussian_levels(m)
        return Source(f"shuffled_d{d}_m{m}", d,
                      [lv[torch.randperm(m, generator=g)][:, None] for _ in range(d)])
    if kind == "onehot":
        e = (torch.eye(m) - 1 / m) / math.sqrt((1 / m) * (1 - 1 / m))
        return Source(f"onehot_d{d}_m{m}", d * m, [e] * d)
    if kind == "categorical":
        e = torch.randn(K, CAT_DIM, generator=g)
        e = (e - e.mean(0)) / e.std(0, unbiased=False) if K > 1 else torch.zeros(1, CAT_DIM)
        return Source(f"categorical_K{K}", CAT_DIM, [e])
    if kind == "frozen":
        return Source(f"frozen_d{d}_K{K}", d, [torch.randn(K, d, generator=g)])
    if kind == "constant":
        return Source("constant", 1, [torch.zeros(1, 1)])
    raise ValueError(kind)
