"""Figure: E rate, I rate and synchrony around the four kinds of state boundary.
Reads results/transitions/curves.json (written by src.transitions); writes figures/transitions.png."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reference import TYPE_COLORS  # noqa: E402

curves = json.loads((ROOT / "results" / "transitions" / "curves.json").read_text())
kinds = ["MA onset", "WI onset", "WAKE->SLEEP", "SLEEP->WAKE"]
titles = {"MA onset": "Microarousal onset", "WI onset": "Wake-interruption onset",
          "WAKE->SLEEP": "WAKE → SLEEP", "SLEEP->WAKE": "SLEEP → WAKE"}
traces = [("E rate", TYPE_COLORS["E"]), ("I rate", TYPE_COLORS["I"]), ("synchrony", "#555555")]
t = np.arange(-60, 61)
fig, axes = plt.subplots(3, 4, figsize=(13, 7.5), sharex=True)
for j, kind in enumerate(kinds):
    for i, (name, color) in enumerate(traces):
        ax = axes[i, j]
        key = f"{kind}|{name}"
        if key not in curves:
            ax.set_visible(False); continue
        c = curves[key]; m = 100 * (np.array(c["median"]) - 1); lo = 100 * (np.array(c["q25"]) - 1); hi = 100 * (np.array(c["q75"]) - 1)
        ax.fill_between(t, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(t, m, color=color, lw=2)
        ax.axvline(0, color="#999999", lw=1, ls="--"); ax.axhline(0, color="#cccccc", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeeeee")
        if i == 0:
            ax.set_title(f"{titles[kind]}\n(n = {c['n_sessions']} sessions)", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{name}\n% change from baseline", fontsize=10)
        if i == 2:
            ax.set_xlabel("time from boundary (s)")
fig.suptitle("Firing around state boundaries (mean ± sd across sessions; baseline = −60 to −30 s)", fontsize=12)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "transitions.png", dpi=150)
print("wrote figures/transitions.png")
