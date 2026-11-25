"""E7c (second round): the continuous-teacher point for E7, measured rather than asserted.

Teacher T = argmax of the ODE of E4's one-hot flow-matching model, z ~ N(0, I_24). T is deterministic,
so I(Z;X) = H(T(Z)) (plug-in over the pool) and E_z TC(X | Z=z) = 0. The student is trained by CE on
a pool of 2^22 (z, T(z)) pairs (~1300x E4's 3,276) and tested on fresh z. Its law p_G is evaluated like every
continuous source (2^16 MC draws). We report KL(p_T || p_G) against the teacher's own law p_T (the
quantity E7 reports, there p_T = p) and KL(p || p_G), which adds the teacher's own error.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
from e4_coupling import DEV, DZ, fit, ode_decode, teacher  # noqa: E402
from e7_teacher_coupling import OUT  # noqa: E402
from exact import log_marginal, metrics  # noqa: E402
from losses import regression_ce  # noqa: E402
from model import MLP  # noqa: E402

RADIX = D.V ** torch.arange(D.L - 1, -1, -1, device=DEV)


def law(x):
    return torch.bincount((x * RADIX).sum(1), minlength=D.V ** D.L).double() / len(x)


def main(families=("markov", "mix16"), seeds=(0, 1, 2), steps=10000, batch=4096):
    for fam in families:
        p = D.make(fam).to(DEV)
        for seed in seeds:
            out = OUT / f"{fam}_ode_s{seed}.json"
            if out.exists():
                continue
            T = teacher("fm", p, fam, seed)
            g = torch.Generator(DEV).manual_seed(seed)
            z_pool = torch.randn(2**22, DZ, generator=g, device=DEV)
            x_pool = torch.cat([ode_decode(T, zc) for zc in z_pool.split(2**16)])
            pT = law(x_pool)
            torch.manual_seed(seed)
            G = fit(MLP(DZ).to(DEV), z_pool, x_pool, steps, batch=batch,
                    loss_fn=lambda G, z, x: regression_ce(G(z), x))
            z = torch.randn(2**16, DZ, generator=g, device=DEV)
            lq = log_marginal(G, z)
            with torch.no_grad():
                acc = (G(z[:8192]).argmax(-1) == ode_decode(T, z[:8192])).float().mean().item()
            res = {"family": fam, "seed": seed, "I_bits": D.entropy(pT) / math.log(2),
                   "H_bits": D.entropy(p) / math.log(2), "bound": 0.0, "kl_to_pi": D.kl(pT, lq),
                   "teacher_tv": D.tv(p, pT), "test_tok_acc": acc, **metrics(p, lq)}
            print(res, flush=True)
            out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
