"""Collect every experiment's per-fold metrics into tables and figures.

Reads results/<exp>/<rat>/metrics.json (and results/baseline, results/dropping).
Writes results/summary_folds.csv, results/summary_experiments.csv and figures/results_*.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reference import STATE_COLORS, TYPE_COLORS  # noqa: E402
from src.dataset import CLASS_NAMES  # noqa: E402

RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
REGION = {"B22": "mPFC", "BWRat17": "ACC", "BWRat18": "M2", "BWRat19": "M2", "BWRat20": "ACC",
          "BWRat21": "mPFC", "Bogey": "mPFC", "Dino": "mPFC+ACC", "Rizzo": "OFC", "Splinter": "OFC"}
LABEL = {"decoder": "CNN (full)", "baseline": "logistic regression on E/I traces",
         "no_relational": "CNN without relational layer", "no_population": "CNN without population conv layers",
         "E_only": "E cells only", "I_only": "I cells only", "E_matched": "E cells, as many as I cells",
         "control_rawcounts": "CNN on raw counts (control)"}


def collect() -> pd.DataFrame:
    rows = []
    for mfile in RES.glob("*/*/metrics.json"):
        exp, rat = mfile.parts[-3], mfile.parts[-2]
        m = json.loads(mfile.read_text())
        if not m.get("rat"):
            continue
        r = m["rat"]
        rows.append(dict(experiment=exp, rat=rat, region=REGION.get(rat, "?"), n_labeled_s=r["n"],
                         balanced_accuracy=r["balanced_accuracy"], kappa=r["kappa"], accuracy=r["accuracy"],
                         **{f"recall_{k}": v for k, v in r["recall"].items()},
                         epochs=m.get("epochs"), val_bal_acc=m.get("best_val_bal_acc"),
                         train_min=m.get("train_minutes"), confusion=json.dumps(r["confusion"])))
    df = pd.DataFrame(rows).sort_values(["experiment", "rat"])
    df.to_csv(RES / "summary_folds.csv", index=False)
    g = df.groupby("experiment").agg(n_rats=("rat", "count"), bal_acc_mean=("balanced_accuracy", "mean"),
                                     bal_acc_sd=("balanced_accuracy", "std"), kappa_mean=("kappa", "mean"),
                                     kappa_sd=("kappa", "std")).round(3)
    g.to_csv(RES / "summary_experiments.csv")
    return df


def fig_main(df: pd.DataFrame):
    """Per-rat balanced accuracy: CNN vs baseline, plus pooled confusion matrix."""
    d = df[df.experiment.isin(["decoder", "baseline"])]
    if d.empty:
        return
    rats = sorted(d.rat.unique())
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    w = 0.38; x = np.arange(len(rats))
    for i, (exp, color) in enumerate((("decoder", "#2a78d6"), ("baseline", "#b0b0b0"))):
        v = [d[(d.experiment == exp) & (d.rat == r)].balanced_accuracy.mean() for r in rats]
        ax.bar(x + (i - 0.5) * w, v, w, color=color, label=LABEL[exp], edgecolor="white")
    ax.axhline(1 / 3, color="#888888", ls="--", lw=1); ax.text(len(rats) - 0.5, 0.345, "chance", ha="right", fontsize=8, color="#666666")
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\n{REGION[r]}" for r in rats], fontsize=8)
    ax.set_ylabel("balanced accuracy (held-out rat)"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.set_title("Decoding each rat with a model trained on the others", fontsize=11)
    dec = d[d.experiment == "decoder"]
    if dec.empty:
        fig.tight_layout(); fig.savefig(FIG / "results_decoder.png", dpi=150); return
    cm = np.sum([np.array(json.loads(c)) for c in dec.confusion], axis=0)
    cmn = cm / cm.sum(1, keepdims=True)
    axc.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            axc.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=10)
    axc.set_xticks(range(3)); axc.set_xticklabels(CLASS_NAMES); axc.set_yticks(range(3)); axc.set_yticklabels(CLASS_NAMES)
    axc.set_xlabel("predicted"); axc.set_ylabel("authors' label"); axc.set_title("CNN confusion (all held-out rats)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "results_decoder.png", dpi=150)


def fig_ablations(df: pd.DataFrame):
    order = ["decoder", "no_relational", "no_population", "E_only", "E_matched", "I_only", "baseline", "control_rawcounts"]
    d = df[df.experiment.isin(order)]
    exps = [e for e in order if e in set(d.experiment)]
    if len(exps) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(exps) + 1.5))
    for i, e in enumerate(exps):
        v = d[d.experiment == e].balanced_accuracy.values
        color = {"E_only": TYPE_COLORS["E"], "E_matched": TYPE_COLORS["E"], "I_only": TYPE_COLORS["I"]}.get(e, "#2a78d6" if e == "decoder" else "#8a8a8a")
        ax.barh(i, v.mean(), color=color, alpha=0.85, height=0.6)
        ax.errorbar(v.mean(), i, xerr=v.std(), color="black", capsize=3, lw=1)
        ax.scatter(v, np.full(v.size, i), color="black", s=10, zorder=3)
        ax.text(0.02, i, f"n = {v.size} rats", va="center", fontsize=8, color="white")
    ax.set_yticks(range(len(exps))); ax.set_yticklabels([LABEL[e] for e in exps], fontsize=9); ax.invert_yaxis()
    ax.axvline(1 / 3, color="#888888", ls="--", lw=1); ax.set_xlim(0, 1); ax.set_xlabel("balanced accuracy on held-out rats (mean ± sd; dots = rats)")
    ax.spines[["top", "right"]].set_visible(False); ax.set_title("Ablations: what carries the state information?", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "results_ablations.png", dpi=150)


def fig_dropping():
    f = RES / "dropping" / "summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f); d["kk"] = d.k.replace("all", 1000).astype(int)
    g = d.groupby("kk").balanced_accuracy.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.fill_between(range(len(g)), g["mean"] - g["std"], g["mean"] + g["std"], color="#2a78d6", alpha=0.15, lw=0)
    ax.plot(range(len(g)), g["mean"], "-o", color="#2a78d6", lw=2, ms=6)
    ax.set_xticks(range(len(g))); ax.set_xticklabels([str(k) if k < 1000 else "all" for k in g.kk])
    ax.axhline(1 / 3, color="#888888", ls="--", lw=1); ax.set_ylim(0.3, 1)
    ax.set_xlabel("neurons given to the trained decoder (random subsets, 20 draws)"); ax.set_ylabel("balanced accuracy (held-out rats)")
    ax.spines[["top", "right"]].set_visible(False); ax.set_title("Neuron-dropping curve", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "results_dropping.png", dpi=150)


if __name__ == "__main__":
    df = collect()
    print(pd.read_csv(RES / "summary_experiments.csv").to_string(index=False))
    fig_main(df); fig_ablations(df); fig_dropping()
    print("figures written")
