"""E4b (second round): neighbor agreement nu, a discrete Lipschitz proxy for the coupling T.

For each E4 cell, rebuild its M = 4096 (z, x) pairs, find each z's nearest other z in the geometry
G sees (Euclidean on G's input; ties broken at random), and let nu = mean fraction of positions on
which the two targets agree. No training: the held-out accuracy is read from E4's JSONs.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
from e4_coupling import CELLS, DEV, OUT, make_pairs, source_input, target_tv, teacher  # noqa: E402


def nu(z, x, g):
    d = torch.cdist(z, z) + 1e-6 * torch.rand(len(z), len(z), generator=g, device=DEV)
    nn = d.fill_diagonal_(float("inf")).argmin(1)
    return (x == x[nn]).float().mean().item(), (x == x[nn]).all(1).float().mean().item()


def main(families=("mix16", "markov"), seeds=(0, 1, 2)):
    rows = []
    for fam in families:
        p = D.make(fam).to(DEV)
        for seed in seeds:
            cache = {"fm": teacher("fm", p, fam, seed), "vae": teacher("vae", p, fam, seed)}
            for src, coupling in CELLS:
                z_tr, x_tr, z_te, x_te = make_pairs(src, coupling, p, seed, cache)
                z, x = torch.cat([z_tr, z_te]), torch.cat([x_tr, x_te])
                e4 = json.loads((OUT / f"{fam}_{src}_{coupling}_s{seed}.json").read_text())
                nu_tok, nu_seq = nu(source_input(src, z).float(), x, torch.Generator(DEV).manual_seed(seed))
                rows.append({"family": fam, "source": src, "coupling": coupling, "seed": seed,
                             "nu_tok": nu_tok, "nu_seq": nu_seq, "test_tok_acc": e4["test_tok_acc"],
                             "test_seq_acc": e4["test_seq_acc"],
                             "pairs_match_e4": abs(target_tv(p, x) - e4["target_tv"]) < 1e-9})
                print(rows[-1], flush=True)
    (OUT.parent / "e4b.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
