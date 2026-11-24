"""E3b: is the plateau of smooth sources an optimization-budget or MC-N artifact? (mix16, population)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import Config, run  # noqa: E402

SRC = [("gaussian", 4, 2, 0), ("quantized", 2, 16, 0), ("quantized", 4, 2, 0), ("quantized", 32, 2, 0),
       ("categorical", 1, 2, 256), ("onehot", 2, 16, 0)]
if __name__ == "__main__":
    for steps, N in [(16000, 256), (4000, 1024), (16000, 1024)]:
        for s, d, m, K in SRC:
            r = run(Config(exp=f"e3b_steps{steps}", data="mix16", source=s, d=d, m=m, K=K, N=N, steps=steps,
                           eval_every=1000))
            print(r["name"], steps, f"kl={r['final']['kl']:.4f}", flush=True)
