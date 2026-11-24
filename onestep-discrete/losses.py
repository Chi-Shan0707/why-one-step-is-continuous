"""Training objectives."""
import math

import torch

from exact import log_px_given_z


def mc_nll(logits, w, onehot):
    """-sum_x w(x) log (1/N) sum_n prod_i p(x_i | z_n).

    logits (N, L, V) for N source samples; w (S,) data weights over the states in `onehot` (S, L*V).
    With w = p this is the population objective; with w = empirical counts / n it is the sample NLL.
    """
    lp = log_px_given_z(logits, onehot).logsumexp(0) - math.log(logits.shape[0])
    return -(w * lp).sum()


def regression_ce(logits, x):
    """Supervised CE for fixed (z, x) pairs; logits (B, L, V), x (B, L) long."""
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), x.flatten())
