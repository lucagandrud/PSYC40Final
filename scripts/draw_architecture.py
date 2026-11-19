"""Architecture diagram of the decoder -> figures/architecture.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
BLUE, GREY, GREEN, PURPLE = "#2a78d6", "#5b6b7a", "#008300", "#6250d6"

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")


def box(x, y, w, h, title, body, color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=color, lw=1.8))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)
    ax.text(x + w / 2, y + (h - 0.45) / 2 - 0.05, body, ha="center", va="center", fontsize=8.6, color="#222222", linespacing=1.35)


def arrow(x0, x1, y=2.3, label=""):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.4))
    if label:
        ax.text((x0 + x1) / 2, y + 0.22, label, ha="center", va="bottom", fontsize=7.8, color=GREY)


box(0.15, 1.1, 2.0, 2.4, "INPUT", "one labeled second\n+ 0.63 s each side\n\nN neurons × 226 bins\n(10 ms, counts relative\nto each unit's mean)\n+ E / I tag", GREY)
arrow(2.2, 2.65, label="")
box(2.7, 1.1, 2.2, 2.4, "PER-NEURON", "3 conv layers\nkernel 3, dilation 1, 2, 4\nreach 150 ms\n\nsame filters on\nevery neuron's row")
arrow(4.95, 5.4)
box(5.45, 1.1, 2.2, 2.4, "RELATIONAL", "attention across\nneurons at every\n10 ms step\n\ncompares activity,\nnever row numbers")
arrow(7.7, 8.15)
box(8.2, 1.1, 1.75, 2.4, "POOLING", "2 learned queries\nweight neurons by\nwhat they are doing\n\nN disappears")
arrow(10.0, 10.45)
box(10.5, 1.1, 2.35, 2.4, "POPULATION → OUT", "3 conv layers\ndilation 8, 16, 32\ntotal reach 1.27 s\n\nmean over the second\n→ WAKE / nonREM / REM")
ax.text(6.5, 0.45, "10 weight layers, 6.5k weights. Reach set by the paper's OFF periods (75–1250 ms). Weights are shared across neurons, so the trained network runs on any rat with any number of cells.",
        ha="center", fontsize=8.6, color="#333333")
ax.text(6.5, 4.25, "Decoder: a 1D convolutional network that reads the raster directly", ha="center", fontsize=12, fontweight="bold")
fig.savefig(ROOT / "figures" / "architecture.png", dpi=170, bbox_inches="tight")
print("wrote figures/architecture.png")
