"""Train one one-step generator with the unified MC marginal-likelihood objective; write one JSON.

    python run.py --data mix16 --source gaussian --d 4
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import data as D
from exact import eval_z, log_marginal, metrics, onehot_states
from losses import mc_nll
from model import MLP
from sources import make_source

RESULTS = Path(__file__).parent / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    exp: str = "debug"
    data: str = "mix16"
    n_train: int = 0          # 0 = population objective (w = p); else n iid samples
    source: str = "gaussian"
    d: int = 1
    m: int = 2
    K: int = 2
    seed: int = 0
    steps: int = 4000
    lr: float = 1e-3
    N: int = 256              # source samples per step
    enumerate: bool = False   # finite sources: use all K atoms instead of N samples (K <= 4096)
    eval_every: int = 500
    n_eval: int = 2**16       # MC samples for evaluation when the source cannot be enumerated

    def name(self):
        src = make_source(self.source, self.d, self.m, self.K).name
        n = self.n_train or "pop"
        return f"{self.data}_{n}_{src}_N{'enum' if self.enumerate else self.N}_s{self.seed}"


def run(cfg: Config, force=False):
    out = RESULTS / cfg.exp / f"{cfg.name()}.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(DEVICE).manual_seed(cfg.seed)
    p = D.make(cfg.data).to(DEVICE)
    w = (p if not cfg.n_train else D.empirical(p.cpu(), cfg.n_train, seed=cfg.seed).to(DEVICE)).float()
    src = make_source(cfg.source, cfg.d, cfg.m, cfg.K, seed=cfg.seed).to(DEVICE)
    model = MLP(src.dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    oh = onehot_states().to(DEVICE)
    if cfg.enumerate:
        assert src.finite and src.K <= 4096
        z_all = src.atoms()

    eval_gen = torch.Generator(DEVICE).manual_seed(12345)
    z_eval, exact_eval = eval_z(src, cfg.n_eval, eval_gen, DEVICE)

    def evaluate():
        model.eval()
        m = metrics(p, log_marginal(model, z_eval))
        model.train()
        return m

    traj = []
    for step in range(1, cfg.steps + 1):
        z = z_all if cfg.enumerate else src.sample(cfg.N, gen, DEVICE)
        loss = mc_nll(model(z), w, oh)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            traj.append({"step": step, "loss": loss.item(), **evaluate()})

    final = traj[-1]
    res = {
        "config": asdict(cfg), "name": cfg.name(), "source_name": src.name,
        "H_bits": src.entropy_bits, "K": src.K, "dim": src.dim, "exact_eval": exact_eval,
        "tc_data": D.total_correlation(p), "H_data": D.entropy(p),
        "final": final, "best_kl": min(t["kl"] for t in traj), "traj": traj,
    }
    if not exact_eval:  # MC bias check: the same model evaluated with 4x fewer source samples
        res["kl_quarter_n_eval"] = metrics(p, log_marginal(model, z_eval[: cfg.n_eval // 4]))["kl"]
    if cfg.n_train:     # fit to the training sample itself
        res["train_kl_to_empirical"] = D.kl(w.double(), log_marginal(model, z_eval))
    res["wall_time"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f, v in asdict(Config()).items():
        ap.add_argument(f"--{f}", type=type(v) if not isinstance(v, bool) else lambda s: s == "1", default=v)
    ap.add_argument("--force", action="store_true")
    a = vars(ap.parse_args())
    force = a.pop("force")
    r = run(Config(**a), force=force)
    print(r["name"], {k: round(v, 4) for k, v in r["final"].items()}, f"{r['wall_time']:.1f}s")
