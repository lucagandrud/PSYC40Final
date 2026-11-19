"""Architecture figure of the decoder -> figures/architecture.png

One row per stage, top to bottom: a small drawing of what the stage does on the left, a short
explanation on the right, and the tensor shape on the arrow between stages. Every number in
the drawing is the real one (kernel 3, dilations 1/2/4 and 8/16/32, 16 filters, 2 queries,
226 -> 212 -> 100 steps, 3 classes).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reference import STATE_COLORS, TYPE_COLORS  # noqa: E402

E, I = TYPE_COLORS["E"], TYPE_COLORS["I"]
INK, MUTED, LINE = "#1f2933", "#5b6b7a", "#9aa5b1"
CARD, BAND = "#f6f8fa", "#dbe9fb"
BLUE = "#2a78d6"
rng = np.random.default_rng(3)

W, H = 10.6, 11.5
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

ROW_H, GAP, TOP = 1.42, 0.30, 10.65
PIC_X0, PIC_X1, TXT_X = 0.55, 4.55, 4.85
rows_y = [TOP - i * (ROW_H + GAP) - ROW_H for i in range(6)]       # bottom y of each card


def card(y):
    ax.add_patch(FancyBboxPatch((0.3, y), W - 0.6, ROW_H, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=CARD, ec="#e3e8ee", lw=1))


def text(y, title, body):
    ax.text(TXT_X, y + ROW_H - 0.22, title, fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(TXT_X, y + ROW_H - 0.54, body, fontsize=9.5, color=INK, va="top", linespacing=1.42)


def between(y_top_card_bottom, shape):
    """Arrow from one card to the next with the tensor shape beside it."""
    x = 2.55
    ax.add_patch(FancyArrowPatch((x, y_top_card_bottom - 0.02), (x, y_top_card_bottom - GAP + 0.04),
                                 arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.4))
    ax.text(x + 0.18, y_top_card_bottom - GAP / 2, shape, fontsize=9, color=MUTED, va="center", family="monospace")


def strip(x0, y0, w, h, n, cmap="Blues", seed=0):
    v = np.random.default_rng(seed).random(n)
    ax.imshow(v[None, :], extent=(x0, x0 + w, y0, y0 + h), cmap=cmap, vmin=-0.3, vmax=1.3, aspect="auto", interpolation="nearest", zorder=2)


def feature_stack(x0, y0, w, n_feat, total_h, seed=0, label=None):
    h = total_h / n_feat
    for k in range(n_feat):
        strip(x0, y0 + k * h, w, h * 0.85, 24, seed=seed + k)
    if label:
        ax.text(x0 + w / 2, y0 - 0.08, label, fontsize=8, color=MUTED, ha="center", va="top")


def dilated_tree(xc, y_base, dilations, unit, layer_h, color=BLUE, node=0.045):
    """Receptive-field drawing: output node on top, three taps per layer going down."""
    taps = {len(dilations): [0]}
    for li in range(len(dilations) - 1, -1, -1):
        d = dilations[li]
        taps[li] = sorted({t + k * d for t in taps[li + 1] for k in (-1, 0, 1)})
    for li in range(len(dilations), -1, -1):
        y = y_base + li * layer_h
        for t in taps[li]:
            ax.add_patch(Circle((xc + t * unit, y), node, fc=color if li == len(dilations) else "white", ec=color, lw=1.1, zorder=3))
        if li < len(dilations):
            d = dilations[li]
            for t in taps[li + 1]:
                for k in (-1, 0, 1):
                    ax.plot([xc + t * unit, xc + (t + k * d) * unit], [y + layer_h, y], color=color, lw=0.8, alpha=0.8, zorder=2)
    reach = max(taps[0]) - min(taps[0]) + 1
    return reach, min(taps[0]), max(taps[0])


def bracket(x0, x1, y, label, color=MUTED, fs=8.5, below=True):
    s = -1 if below else 1
    ax.plot([x0, x0, x1, x1], [y - s * 0.05, y, y, y - s * 0.05], color=color, lw=0.9)
    ax.text((x0 + x1) / 2, y + s * 0.07, label, fontsize=fs, color=color, ha="center", va="top" if below else "bottom")


# ---------------------------------------------------------------- title
ax.text(0.3, H - 0.32, "The decoder: a 1D convolutional network over the spike trains of all recorded neurons",
        fontsize=13.5, fontweight="bold", color=INK, va="top")

# ---------------------------------------------------------------- 1. input
y = rows_y[0]; card(y)
n_rows, n_e = 11, 8
rx0, rx1 = PIC_X0 + 0.45, PIC_X1 - 0.1
ry0, ry1 = y + 0.42, y + ROW_H - 0.22
ax.add_patch(Rectangle((rx0 + (rx1 - rx0) * 63 / 226, ry0 - 0.03), (rx1 - rx0) * 100 / 226, ry1 - ry0 + 0.06, fc=BAND, ec="none", zorder=1))
rh = (ry1 - ry0) / n_rows
for r in range(n_rows):
    col = E if r < n_e else I
    rate = rng.uniform(0.01, 0.05) if r < n_e else rng.uniform(0.06, 0.14)
    t = np.flatnonzero(rng.random(226) < rate)
    yy = ry1 - (r + 1) * rh
    ax.vlines(rx0 + (rx1 - rx0) * t / 226, yy + 0.12 * rh, yy + 0.88 * rh, color=col, lw=0.9, zorder=2)
    ax.add_patch(Rectangle((rx0 - 0.14, yy + 0.2 * rh), 0.08, 0.6 * rh, fc=col, ec="none"))
ax.text(rx0 - 0.42, (ry0 + ry1) / 2, "N\nneurons", fontsize=8.5, color=MUTED, ha="center", va="center")
bracket(rx0, rx0 + (rx1 - rx0) * 63 / 226, ry0 - 0.05, "0.63 s")
bracket(rx0 + (rx1 - rx0) * 63 / 226, rx0 + (rx1 - rx0) * 163 / 226, ry0 - 0.05, "labeled second (100 bins)", color=BLUE)
bracket(rx0 + (rx1 - rx0) * 163 / 226, rx1, ry0 - 0.05, "0.63 s")
ax.text(rx0 - 0.14, ry1 + 0.04, "tag", fontsize=7.5, color=MUTED, ha="center", va="bottom")
text(y, "1  Input: the spike trains of one second, with context",
     "N neurons × 226 bins of 10 ms. Each bin holds the unit's spike count divided by\n"
     "that unit's mean count over the recording; a second channel tags the neuron as\n"
     "excitatory (green) or inhibitory (purple). N differs between rats, and the order\n"
     "of the neurons carries no meaning.")

# ---------------------------------------------------------------- 2. per-neuron convolutions
between(y, "N × 2 × 226")
y = rows_y[1]; card(y)
unit = 0.115; xc = PIC_X0 + 1.3
reach, lo, hi = dilated_tree(xc, y + 0.42, (1, 2, 4), unit, 0.27)
ax.vlines(xc + np.arange(lo, hi + 1) * unit, y + 0.2, y + 0.33, color=E, lw=1.2)        # the spike train under the taps
for t in range(lo - 3, lo):
    ax.vlines(xc + t * unit, y + 0.2, y + 0.33, color=E, lw=1.2, alpha=0.3)
for t in range(hi + 1, hi + 4):
    ax.vlines(xc + t * unit, y + 0.2, y + 0.33, color=E, lw=1.2, alpha=0.3)
bracket(xc + lo * unit - 0.04, xc + hi * unit + 0.04, y + 0.17, f"{reach} bins = 150 ms seen by one output step")
for li, d in enumerate((1, 2, 4)):
    ax.text(xc + (hi + 2.2) * unit, y + 0.42 + li * 0.27 + 0.135, f"dilation {d}", fontsize=8, color=MUTED, va="center")
feature_stack(PIC_X0 + 3.25, y + 0.32, 0.72, 16, 0.9, seed=10, label="16 features per step")
text(y, "2  Per-neuron convolutions (3 layers, kernel 3)",
     "The same three filters slide along every neuron's spike train, with dilations\n"
     "1, 2 and 4 so the third layer sees 150 ms. Output: 16 features per neuron per\n"
     "10 ms step. Because the filters are shared across neurons, a network trained on\n"
     "nine rats runs on a tenth with a different set of cells.")

# ---------------------------------------------------------------- 3. attention across neurons
between(y, "N × 16 × 212")
y = rows_y[2]; card(y)
n = 9; ys = np.linspace(y + 0.28, y + ROW_H - 0.28, n)
xl, xr = PIC_X0 + 0.9, PIC_X0 + 2.9
cols = [E] * 6 + [I] * 3
for k, (yy, c) in enumerate(zip(ys, cols)):
    for yy2 in ys:
        ax.plot([xl, xr], [yy, yy2], color=LINE, lw=0.4, alpha=0.35, zorder=1)
target = 3
wts = rng.dirichlet(np.ones(n) * 0.6)
for k, (yy, c) in enumerate(zip(ys, cols)):
    ax.plot([xl, xr], [yy, ys[target]], color=BLUE, lw=0.4 + 3.2 * wts[k], alpha=0.9, zorder=2)
for k, (yy, c) in enumerate(zip(ys, cols)):
    ax.add_patch(Circle((xl, yy), 0.065, fc=c, ec="white", lw=0.8, zorder=3))
    ax.add_patch(Circle((xr, yy), 0.065, fc=c if k == target else "white", ec=c, lw=1.2, zorder=3))
ax.text(xl, y + ROW_H - 0.1, "neurons at one 10 ms step", fontsize=8, color=MUTED, ha="center", va="top")
ax.text(xr, y + ROW_H - 0.1, "updated", fontsize=8, color=MUTED, ha="center", va="top")
ax.text(xr + 0.15, ys[target], "weights depend on\nwhat each neuron is\ndoing right now", fontsize=8, color=MUTED, va="center")
text(y, "3  Attention across neurons (1 layer)",
     "At every step, each neuron's 16 features are updated with a weighted sum of\n"
     "all other neurons' features, so the network can read which cells fire together,\n"
     "including which excitatory cells fire with which inhibitory cells. The weights\n"
     "come from the features, never from the neuron's row number.")

# ---------------------------------------------------------------- 4. pooling
between(y, "N × 16 × 212")
y = rows_y[3]; card(y)
ys = np.linspace(y + 0.28, y + ROW_H - 0.28, n)
xl, xq = PIC_X0 + 0.6, PIC_X0 + 2.2
qy = [y + ROW_H * 0.64, y + ROW_H * 0.36]
for qi, yq in enumerate(qy):
    w = rng.dirichlet(np.ones(n) * 0.8)
    for k, yy in enumerate(ys):
        ax.plot([xl, xq - 0.12], [yy, yq], color=BLUE, lw=0.3 + 2.8 * w[k], alpha=0.8, zorder=2)
    ax.add_patch(FancyBboxPatch((xq - 0.12, yq - 0.12), 0.5, 0.24, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=BLUE, lw=1.3, zorder=3))
    ax.text(xq + 0.13, yq, f"query {qi + 1}", fontsize=8.5, color=BLUE, ha="center", va="center", zorder=4)
for k, (yy, c) in enumerate(zip(ys, cols)):
    ax.add_patch(Circle((xl, yy), 0.065, fc=c, ec="white", lw=0.8, zorder=3))
ax.text(xl, y + ROW_H - 0.1, "N neurons", fontsize=8, color=MUTED, ha="center", va="top")
for qi, yq in enumerate(qy):
    ax.add_patch(FancyArrowPatch((xq + 0.42, yq), (xq + 0.75, yq), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
    feature_stack(xq + 0.8, yq - 0.14, 0.8, 16, 0.28, seed=20 + qi)
ax.text(xq + 1.2, y + 0.16, "2 × 16 numbers per step", fontsize=8, color=MUTED, ha="center", va="top")
text(y, "4  Attention pooling (2 learned queries)",
     "Each query scores every neuron by its current features and takes a weighted\n"
     "average of them (weights sum to 1). The neuron dimension disappears here: from\n"
     "this point on there is one population signal per step, whatever the number of\n"
     "cells.")

# ---------------------------------------------------------------- 5. population convolutions
between(y, "32 × 212")
y = rows_y[4]; card(y)
unit = 0.0235; xc = PIC_X0 + 2.0
reach, lo, hi = dilated_tree(xc, y + 0.46, (8, 16, 32), unit, 0.26, node=0.035)
strip(xc + (lo - 20) * unit, y + 0.22, (hi - lo + 40) * unit, 0.14, 160, seed=30)
ax.add_patch(Rectangle((xc + lo * unit, y + 0.21), (hi - lo) * unit, 0.16, fc="none", ec=BLUE, lw=1.1, zorder=3))
bracket(xc + lo * unit, xc + hi * unit, y + 0.18, f"{reach} steps = 1.27 s of input seen by one output step")
for li, d in enumerate((8, 16, 32)):
    ax.text(xc + (hi + 6) * unit, y + 0.46 + li * 0.26 + 0.13, f"dilation {d}", fontsize=8, color=MUTED, va="center")
text(y, "5  Population convolutions (3 layers, kernel 3)",
     "The same kind of filters, now over the population signal, with dilations 8, 16\n"
     "and 32: each output step sees 113 steps of that signal, 1.27 s of input in all.\n"
     "Convolutions are unpadded, so 226 input bins leave exactly the 100 steps of the\n"
     "labeled second, each read with its full reach.")

# ---------------------------------------------------------------- 6. readout
between(y, "16 × 100")
y = rows_y[5]; card(y)
sx0, sw = PIC_X0 + 0.1, 1.3
feature_stack(sx0, y + 0.38, sw, 16, 0.7, seed=40)
bracket(sx0, sx0 + sw, y + 0.33, "mean over the 100 steps")
ax.add_patch(FancyArrowPatch((sx0 + sw + 0.1, y + 0.72), (sx0 + sw + 0.45, y + 0.72), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.1))
vx = sx0 + sw + 0.5
for k in range(16):
    ax.add_patch(Rectangle((vx, y + 0.3 + k * 0.052), 0.12, 0.045, fc=plt.cm.Blues(0.3 + 0.6 * rng.random()), ec="none"))
ax.text(vx + 0.06, y + 0.22, "16", fontsize=8, color=MUTED, ha="center", va="top")
ax.add_patch(FancyArrowPatch((vx + 0.2, y + 0.72), (vx + 0.55, y + 0.72), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.1))
ax.text(vx + 0.37, y + 0.8, "linear", fontsize=7.5, color=MUTED, ha="center", va="bottom")
bx = vx + 0.62
for k, (name, p) in enumerate((("WAKE", 0.18), ("nonREM", 0.70), ("REM", 0.12))):
    yy = y + ROW_H - 0.42 - k * 0.34
    ax.add_patch(Rectangle((bx + 0.62, yy - 0.1), 0.7 * p, 0.2, fc=STATE_COLORS[name], ec="none"))
    ax.text(bx + 0.56, yy, name, fontsize=8.5, color=INK, ha="right", va="center")
    ax.text(bx + 0.62 + 0.7 * p + 0.05, yy, f"{p:.2f}", fontsize=8, color=MUTED, va="center")
text(y, "6  Readout",
     "The 100 steps of the labeled second are averaged and a linear layer turns the\n"
     "16 numbers into probabilities for WAKE, nonREM and REM. Trained with cross-\n"
     "entropy against the authors' label of that second.")

ax.text(0.3, 0.2, "10 weight layers, 6,467 weights in total. Reach: 150 ms per neuron, 1.27 s in all, chosen to cover the paper's range of "
        "nonREM OFF periods (75–1250 ms).", fontsize=9, color=MUTED, va="bottom")

fig.savefig(ROOT / "figures" / "architecture.png", dpi=165)
print("wrote figures/architecture.png")
