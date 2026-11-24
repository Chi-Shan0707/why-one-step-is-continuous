"""All figures, generated from results/*.json only.   python plots.py"""
import collections
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).parent / "results"
FIG = Path(__file__).parent / "figures"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 6, "font.size": 10, "legend.frameon": False, "axes.titlesize": 11,
})
FLOOR = 1e-4  # KL values below this are drawn at the floor on log axes
FLOOR_NOTE = "KL < 1e-4 drawn at 1e-4"


def fl(v):
    return np.maximum(v, FLOOR)


def load(pattern):
    return [json.loads(f.read_text()) for f in sorted(R.glob(pattern))]


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def fig_e1():
    rs = {r["family"]: r for r in load("e1/*.json")}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for c, fam in zip(SLOTS, ["mix2", "mix4", "mix16", "mix64"]):
        if fam not in rs:
            continue
        r = rs[fam]
        k = [x["k"] for x in r["rows"]]
        kl = [max(x["kl_best"], FLOOR) for x in r["rows"]]
        ax.plot(k, kl, "-o", color=c, label=f"{fam}  (TC = {r['tc']:.2f})")
        ax.axvline(int(fam[3:]), color=c, lw=1, ls=":")
    ax.set(xscale="log", yscale="log", xlabel="mixture components k (EM, best of 10)",
           ylabel="KL(p || p_k)  [nats]", title=f"E1: EM's KL vanishes at k ~ r (dotted: k = r; {FLOOR_NOTE})")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], [1, 2, 4, 8, 16, 32, 64, 128])
    ax.legend()
    save(fig, "e1_rank.png")


def fig_e2():
    rs = load("e2/*.json")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for c, fam in zip(SLOTS, ["markov", "mix16", "copy"]):
        rows = [r for r in rs if r["family"] == fam]
        if not rows:
            continue
        ks = [x["k"] for x in rows[0]["rows"]]
        m = fl(np.mean([[x["kl_model"] for x in r["rows"]] for r in rows], 0))
        o = [max(x["kl_oracle"], FLOOR) for x in rows[0]["rows"]]
        ax.plot(ks, m, "-o", color=c, label=f"{fam} MDM (TC = {rows[0]['tc']:.2f})")
        ax.plot(ks, o, "--", color=c, lw=1.2, alpha=0.8)
    ax.plot([], [], "--", color=INK2, lw=1.2, label="exact conditionals (oracle)")
    ax.set(yscale="log", xlabel="sampling steps k (L/k tokens per step)", ylabel="KL(p || p_k)  [nats]",
           title=f"E2: committed tokens act as shared randomness ({FLOOR_NOTE})")
    ax.set_xticks([1, 2, 3, 6])
    ax.legend()
    save(fig, "e2_steps.png")


E3_SERIES = [("quantized_m2", "Rademacher (hypercube)"), ("onehot_m4", "one-hot levels, m=4"),
             ("onehot_m16", "one-hot levels, m=16"), ("categorical", "categorical, random emb."),
             ("quantized_m4", "quantized Gaussian, m=4"), ("quantized_m16", "quantized Gaussian, m=16"),
             ("quantized_m256", "quantized Gaussian, m=256")]


def e3_table(prefix):
    """{series: {H: [final kl per seed]}} and {continuous source: {d: [...]}}."""
    disc, cont = collections.defaultdict(lambda: collections.defaultdict(list)), \
        collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load(f"e3/{prefix}_*.json"):
        n = r["source_name"]
        if n.startswith(("gaussian", "uniform")):
            cont[n.split("_")[0]][r["dim"]].append(r["final"]["kl"])
        elif not n.startswith("shuffled"):
            disc[re.sub(r"_d\d+|_K\d+", "", n)][r["H_bits"]].append(r["final"]["kl"])
    return disc, cont


def fig_e3():
    panels = [("mix16_pop", "mix16, population objective"), ("mix16_500", "mix16, n = 500 samples"),
              ("mix16_5000", "mix16, n = 5000 samples"), ("markov_pop", "Markov, population objective")]
    panels = [p for p in panels if list(R.glob(f"e3/{p[0]}_*.json"))]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4), squeeze=False)
    for ax, (prefix, title) in zip(axes[0], panels):
        disc, cont = e3_table(prefix)
        for c, (key, label) in zip(SLOTS, E3_SERIES):
            if key not in disc:
                continue
            H = sorted(disc[key])
            y = fl([np.mean(disc[key][h]) for h in H])
            lo = fl([min(disc[key][h]) for h in H])
            hi = fl([max(disc[key][h]) for h in H])
            ax.errorbar(H, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="-o", color=c,
                        label=label, capsize=0, elinewidth=1)
        for ls, name in [("--", "gaussian"), (":", "uniform")]:
            if name in cont:
                best = min(np.mean(v) for v in cont[name].values())
                ax.axhline(fl(best), color=INK2, ls=ls, lw=1.2, label=f"{name} (best d_z)")
        if prefix.startswith("mix16"):
            ax.axvline(4, color=INK2, lw=0.8)
            ax.text(4.3, 0.97, "log2 r", color=INK2, fontsize=8, va="top",
                    transform=ax.get_xaxis_transform())
        ax.set(xscale="log", yscale="log", xlabel="source entropy H(Z)  [bits]", title=title)
        ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
    axes[0][0].set_ylabel("KL(p || p_theta)  [nats]")
    fig.text(0.0, -0.06, "Points: mean over 3 seeds, bars: min-max. KL is exact for sources with K <= 2^16 atoms "
             f"and a 2^16-sample MC estimate otherwise (bias < 1%). {FLOOR_NOTE}.", fontsize=8, color=INK2)
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("E3: same generator and objective, only P_Z changes", y=1.02)
    save(fig, "e3_sources.png")


E4_CELLS = [("gauss", "random"), ("gauss", "vae"), ("gauss", "ot"), ("gauss", "ode"), ("unif", "kr"),
            ("code", "random"), ("code", "ot"), ("quant", "kr"),
            ("onehot", "random"), ("onehot", "ot"), ("shuf", "kr")]
E4_GROUP = {"gauss": 0, "unif": 0, "code": 1, "quant": 1, "onehot": 2, "shuf": 2}
E4_LEGEND = ["continuous source", "discrete source with geometry", "discrete source without geometry"]


def fig_e4():
    rs = load("e4/*.json")
    if not rs:
        return
    fams = sorted({r["family"] for r in rs})
    metrics_ = [("test_tok_acc", "held-out token accuracy"), ("test_seq_acc", "held-out sequence accuracy")]
    fig, axes = plt.subplots(len(fams), 2, figsize=(12, 3.3 * len(fams)), squeeze=False)
    for i, fam in enumerate(fams):
        for j, (key, label) in enumerate(metrics_):
            ax = axes[i][j]
            vals = [[r[key] for r in rs if r["family"] == fam and (r["source"], r["coupling"]) == c]
                    for c in E4_CELLS]
            ax.bar(range(len(E4_CELLS)), [np.mean(v) if v else np.nan for v in vals],
                   color=[SLOTS[E4_GROUP[c[0]]] for c in E4_CELLS], edgecolor=SURF, linewidth=2)
            for x, v in enumerate(vals):
                ax.plot([x] * len(v), v, "o", color=INK, ms=3)
            ax.set_xticks(range(len(E4_CELLS)), [f"{s}\n{c}" for s, c in E4_CELLS], fontsize=8)
            ax.set(title=f"{fam}: {label}", ylim=(0, 1))
            ax.grid(axis="x", visible=False)
    axes[0][0].legend(handles=[Patch(color=SLOTS[k], label=lab) for k, lab in enumerate(E4_LEGEND)],
                      fontsize=8, loc="upper left")
    fig.suptitle("E4: fit G by regression on 3,277 fixed (z, x) pairs; test on 819 held-out z "
                 "(bars: mean, dots: 3 seeds)", y=1.0)
    fig.tight_layout()
    save(fig, "e4_coupling.png")


def fig_e6():
    rs = load("e6/*.json")
    if not rs:
        return
    # code1024 is the full parity code: every source stays at the product solution (no low-order
    # gradient signal), a separate phenomenon reported in the text, so it is not plotted here
    rvals = sorted({int(r["config"]["data"][4:]) for r in rs} - {1024})
    fig, axes = plt.subplots(1, len(rvals), figsize=(3.6 * len(rvals), 3.4), sharey=True, squeeze=False)
    for ax, rv in zip(axes[0], rvals):
        sub = [r for r in rs if r["config"]["data"] == f"code{rv}"]
        for c, src, label in [(SLOTS[0], "gaussian", "Gaussian d=8, N samples"),
                              (SLOTS[1], "categorical", f"categorical K={4 * rv}, N samples")]:
            by_n = collections.defaultdict(list)
            for r in sub:
                if r["config"]["source"] == src and not r["config"]["enumerate"]:
                    by_n[r["config"]["N"]].append(r["final"]["kl"])
            if by_n:
                ns = sorted(by_n)
                ax.plot(ns, fl([np.mean(by_n[n]) for n in ns]), "-o", color=c, label=label)
        enum = [r["final"]["kl"] for r in sub if r["config"]["enumerate"]]
        if enum:
            ax.axhline(fl(np.mean(enum)), color=SLOTS[1], ls="--", lw=1.2, label="categorical, all K enumerated")
        ax.axvline(rv, color=INK2, lw=0.8)
        ax.set(xscale="log", yscale="log", title=f"code{rv} (exact fit needs {int(math.log2(rv))} bits)",
               xlabel="source samples per step N")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("KL(p || p_theta)  [nats]")
    fig.suptitle("E6: the MC marginal-likelihood objective vs N (vertical line: N = r)", y=1.03)
    save(fig, "e6_nscaling.png")


def fig_e8():
    rs = load("e8/zipf16_*.json")
    if not rs:
        return
    floors = {int(k): v for k, v in json.loads((R / "e8" / "floors.json").read_text()).items()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    # left: final mass-allocation error vs H(Z)
    for c, src, label in [(SLOTS[1], "categorical", "categorical, random emb."),
                          (SLOTS[0], "rademacher", "Rademacher hypercube")]:
        by_h = collections.defaultdict(list)
        for r in rs:
            if r["config"]["source"] == src:
                by_h[r["H_bits"]].append(r["traj"][-1]["kl_alloc"])
        H = sorted(by_h)
        y = fl([np.mean(by_h[h]) for h in H])
        a1.errorbar(H, y, yerr=[y - fl([min(by_h[h]) for h in H]), fl([max(by_h[h]) for h in H]) - y],
                    fmt="-o", color=c, label=label + " (exact objective)", elinewidth=1)
    H = sorted(floors)
    a1.plot([math.log2(k) for k in H], fl([floors[k] for k in H]), "--", color=INK2, lw=1.2,
            label="best allocation, one codeword per atom")
    g = [r["traj"][-1]["kl_alloc"] for r in rs if r["config"]["source"] == "gaussian"]
    a1.axhline(fl(np.mean(g)), color=SLOTS[2], lw=1.5, label="Gaussian (d = 2, 4, 8)")
    a1.set(yscale="log", xlabel="source entropy H(Z) = log2 K  [bits]",
           ylabel="mass-allocation error KL_alloc  [nats]", title="after 16k steps")
    a1.legend(fontsize=8)
    # right: allocation vs leakage over training, three representative sources
    for c, (src, key, val), label in [(SLOTS[1], ("categorical", "K", 1024), "categorical K=1024"),
                                      (SLOTS[0], ("rademacher", "d", 12), "Rademacher d=12"),
                                      (SLOTS[2], ("gaussian", "d", 2), "Gaussian d=2")]:
        sub = [r for r in rs if r["config"]["source"] == src and r["config"][key] == val]
        steps = [t["step"] for t in sub[0]["traj"]]
        for metric, ls in [("kl_alloc", "-"), ("leak_nats", ":")]:
            a2.plot(steps, fl(np.mean([[t[metric] for t in r["traj"]] for r in sub], 0)), ls, color=c,
                    label=f"{label}: {'allocation' if metric == 'kl_alloc' else 'leakage'}")
    a2.set(yscale="log", xlabel="training step", ylabel="nats", title="during training (mean of 3 seeds)")
    a2.legend(fontsize=7, ncol=1)
    fig.suptitle("E8: 16 codewords with Zipf weights. KL = allocation error + leakage. "
                 f"Atoms cannot move mass; a continuous source can. ({FLOOR_NOTE})", y=1.02)
    fig.tight_layout()
    save(fig, "e8_mass.png")


def fig_e7():
    rs = load("e7/*.json")
    if not rs:
        return
    fig, axes = plt.subplots(1, len(rs), figsize=(4.2 * len(rs), 3.6), squeeze=False)
    for ax, r in zip(axes[0], rs):
        for c, kind, label in [(SLOTS[0], "uniform", "uniform-noise teacher (prior H(Z) = 12 bits)"),
                               (SLOTS[1], "mask", "masking teacher (prior H(Z) = 0 bits)")]:
            rows = [x for x in r["rows"] if x["kernel"] == kind]
            ax.plot([x["I_bits"] for x in rows], fl([x["kl"] for x in rows]), "-o", color=c, label=label)
        ax.axhline(r["tc"], color=INK2, ls=":", lw=1)
        ax.set(yscale="log", xlabel="I(Z; X) of the teacher coupling  [bits]",
               title=f"{r['family']} (dotted: TC = {r['tc']:.2f})")
    axes[0][0].set_ylabel("KL of best regression student")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"E7: a discrete teacher's coupling carries I(Z;X) = 0 at its prior ({FLOOR_NOTE})", y=1.03)
    save(fig, "e7_teacher_coupling.png")


if __name__ == "__main__":
    for f in [fig_e1, fig_e2, fig_e3, fig_e4, fig_e6, fig_e7, fig_e8]:
        f()
