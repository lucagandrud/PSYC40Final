"""Build the rotatable 3D model (Plotly HTML) and the mp4 for one session across a WAKE -> SLEEP switch.

Showcase: BWRat20_101513, one probe in the anterior cingulate cortex (ACC) of rat BWRat20, across
the authors' WAKE -> SLEEP switch at 16578 s. BWRat20 was the held-out rat for the predictions
shown. The window was chosen (scripts/pick: scan of every WAKE -> SLEEP boundary in every session)
as the one where the decoder agrees best with the authors on both sides of the switch.
Each unit's marker swells with the spikes it fired in the current 100 ms frame (decaying over three
frames). Colors: excitatory neurons green, inhibitory neurons purple (the authors' labels).
Outputs: results/brain3d/brain3d.html, results/brain3d/brain3d.mp4, results/brain3d/showcase.json,
figures/brain3d_still.png
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.brain3d import OUT, REGION_MAP, sample_region_points, place_units, spike_frames, state_strip  # noqa: E402
from src.reference import TYPE_COLORS, STATE_COLORS  # noqa: E402

SESSION, RAT, REGION = "BWRat20_101513", "BWRat20", "ACC"
REGION_NAME = {"ACC": "anterior cingulate cortex", "mPFC": "medial prefrontal cortex", "M2": "secondary motor cortex", "OFC": "orbitofrontal cortex"}
STRUCT_NAME = {"Cg1": "cingulate area 1", "Cg2": "cingulate area 2", "PrL": "prelimbic area", "IL": "infralimbic area",
               "M2": "secondary motor area", "VO": "ventral orbital area", "LO": "lateral orbital area", "MO": "medial orbital area"}
BOUNDARY = 16578.0                              # first second after the authors' last WAKE second
T0, T1, STEP = BOUNDARY - 50, BOUNDARY + 60, 0.1
STATE_NAME = {0: "unscored", 1: "WAKE", 2: "nonREM", 3: "REM", 4: "microarousal", 5: "wake interruption", 9: "ambiguous"}
STATE_HEX = {1: STATE_COLORS["WAKE"], 2: STATE_COLORS["nonREM"], 3: STATE_COLORS["REM"], 4: STATE_COLORS["MA"], 5: STATE_COLORS["WI"], 0: "#dddddd", 9: "#dddddd"}
TYPE_NAME = {"E": "excitatory neuron", "I": "inhibitory neuron"}


def discrete_colorscale():
    """Plotly colorscale that maps integer state codes 0..5 to STATE_HEX."""
    n = 6; cs = []
    for k in range(n):
        cs += [[k / n, STATE_HEX[k]], [(k + 1) / n, STATE_HEX[k]]]
    return cs


def main():
    pts = np.load(sample_region_points())
    units = place_units(SESSION, pts[f"pts_{REGION}"])
    times, X = spike_frames(SESSION, T0, T1, STEP)
    pulse = X.astype(float).copy()
    for k in (1, 2):                                                  # decay over three frames
        pulse[k:] += X[:-k] * (0.5 ** k)
    size = 4 + 5 * np.minimum(pulse, 3)
    strip = state_strip(SESSION, T0, T1)
    secs = strip["seconds"]; tr = strip["true"]; pr = strip["pred"]
    colors = units.type.map(TYPE_COLORS).to_numpy()
    # facts for the notebook
    m = np.isin(tr, (1, 2, 3)); before = secs < BOUNDARY
    q = np.isin(pr, (2, 3)).astype(int); run = np.convolve(q, np.ones(5, int), "valid")
    first_sleep = next((int(secs[i]) for i in range(len(run)) if secs[i] >= BOUNDARY - 10 and run[i] == 5), None)
    facts = dict(session=SESSION, rat=RAT, region=REGION, region_name=REGION_NAME[REGION], t0=T0, t1=T1, boundary=BOUNDARY,
                 n_units=int(len(units)), n_E=int((units.type == "E").sum()), n_I=int((units.type == "I").sum()),
                 wake_correct=float((pr[m & before] == 1).mean()), sleep_correct=float(np.isin(pr[m & ~before], (2, 3)).mean()),
                 decoder_sleep_lag_s=None if first_sleep is None else first_sleep - BOUNDARY,
                 ma_seconds=int((tr == 4).sum()), ma_called_wake=float((pr[tr == 4] == 1).mean()) if (tr == 4).any() else None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "showcase.json").write_text(json.dumps(facts, indent=1))
    print(facts)
    # ---------------------------------------------------------------- Plotly: 3D scene + state strip, animated together
    fig = make_subplots(rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "xy"}]], row_heights=[0.8, 0.2], vertical_spacing=0.09)
    for ac in REGION_MAP[REGION]:
        v, f = pts[f"mesh_{ac}_v"], pts[f"mesh_{ac}_f"]
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 2], z=-v[:, 1], i=f[:, 0], j=f[:, 1], k=f[:, 2], opacity=0.12, color="#c9a27e",
                                name=f"{REGION_NAME[REGION]}: {STRUCT_NAME[ac]} ({ac})", hoverinfo="name", showlegend=True), row=1, col=1)
    hover = [f"unit {r.unit_id}: {TYPE_NAME[r.type]}, shank {r.shank}" for r in units.itertuples()]
    fig.add_trace(go.Scatter3d(x=units.ap, y=units.rl, z=-units.si, mode="markers", name="units",
                               marker=dict(size=size[0], color=colors, opacity=0.95, line=dict(width=0)), text=hover, hoverinfo="text", showlegend=False), row=1, col=1)
    units_idx = len(fig.data) - 1
    for name, col in (("excitatory neurons", TYPE_COLORS["E"]), ("inhibitory neurons", TYPE_COLORS["I"])):
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=8, color=col), name=name, showlegend=True), row=1, col=1)
    z = np.vstack([pr, tr]).astype(float)
    fig.add_trace(go.Heatmap(z=z, x=secs + 0.5, y=[0, 1], zmin=0, zmax=6, colorscale=discrete_colorscale(), showscale=False,
                             hovertemplate="%{y}: %{customdata}<extra></extra>",
                             customdata=np.vectorize(STATE_NAME.get)(z.astype(int)), xgap=0, ygap=2), row=2, col=1)
    fig.add_trace(go.Scatter(x=[T0, T0], y=[-0.5, 1.5], mode="lines", line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"), row=2, col=1)
    cursor_idx = len(fig.data) - 1
    for k, (name, code) in enumerate((("WAKE", 1), ("nonREM", 2), ("REM", 3), ("microarousal", 4))):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=10, color=STATE_HEX[code]), name=name, showlegend=True), row=2, col=1)
    pl_frames = []
    for i, t in enumerate(times):
        j = int(np.floor(t)) - int(T0)
        pl_frames.append(go.Frame(data=[go.Scatter3d(marker=dict(size=size[i], color=colors)), go.Scatter(x=[t, t], y=[-0.5, 1.5])],
                                  traces=[units_idx, cursor_idx], name=f"{t:.1f}",
                                  layout=go.Layout(title_text=f"t = {t - T0:5.1f} s     authors: {STATE_NAME[int(tr[j])]}     decoder: {STATE_NAME[int(pr[j])]}")))
    fig.frames = pl_frames
    fig.update_layout(
        title=dict(text=f"{SESSION}: {facts['n_units']} {REGION_NAME[REGION]} units across the authors' WAKE → SLEEP switch "
                        "(positions inside the atlas region are schematic)", font=dict(size=14)),
        scene=dict(xaxis_title="anterior–posterior (µm)", yaxis_title="right–left (µm)", zaxis_title="depth (µm)", aspectmode="data",
                   camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))),
        legend=dict(x=0.01, y=0.99, font=dict(size=11)), margin=dict(l=0, r=0, t=70, b=0), height=820,
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.3,
                          buttons=[dict(label="▶ play", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                                   dict(label="■ pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.1, y=0.31, len=0.85, currentvalue=dict(prefix="time in window (s): "),
                      steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=f"{float(f.name) - T0:.0f}") for f in pl_frames[::10]])])
    fig.update_xaxes(title_text="session time (s)", tickformat="d", row=2, col=1)
    fig.update_yaxes(range=[-0.5, 1.5], tickvals=[0, 1], ticktext=["decoder", "authors"], fixedrange=True, row=2, col=1)
    fig.write_html(OUT / "brain3d.html", include_plotlyjs="cdn", auto_play=False)
    # ---------------------------------------------------------------- mp4 via matplotlib frames
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fdir = OUT / "frames"; fdir.mkdir(exist_ok=True)
    for f in fdir.glob("*.png"):
        f.unlink()
    p = pts[f"pts_{REGION}"][::2]
    still_i = int((BOUNDARY + 5 - T0) / STEP)
    for i, t in enumerate(times):
        if "--still-only" in sys.argv and i != still_i:
            continue
        j = int(np.floor(t)) - int(T0)
        fig2 = plt.figure(figsize=(10, 7.8))
        ax = fig2.add_axes([0.0, 0.21, 1.0, 0.72], projection="3d")
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1], s=9, color="#d4a97a", alpha=0.08, linewidths=0, depthshade=False)
        ax.scatter(units.ap, units.rl, -units.si, s=size[i] ** 2 * 0.35, c=colors, alpha=0.95, linewidths=0, depthshade=False)
        ax.view_init(elev=22, azim=-60 + 0.25 * i)
        ax.set_xlabel("anterior–posterior (µm)"); ax.set_ylabel("right–left (µm)"); ax.set_zlabel("depth (µm)")
        ax.set_title(f"{SESSION}: {facts['n_units']} units in the {REGION_NAME[REGION]} (tan volume), t = {t - T0:5.1f} s", fontsize=12, pad=0)
        fig2.text(0.5, 0.905, f"authors: {STATE_NAME[int(tr[j])]}", ha="right", fontsize=12, color=STATE_HEX.get(int(tr[j]), "#555555"), fontweight="bold")
        fig2.text(0.52, 0.905, f"decoder: {STATE_NAME[int(pr[j])]}", ha="left", fontsize=12, color=STATE_HEX.get(int(pr[j]), "#555555"), fontweight="bold")
        axs = fig2.add_axes([0.1, 0.085, 0.82, 0.1]); axs.set_xlim(T0, T1); axs.set_ylim(0, 2); axs.set_yticks([0.5, 1.5])
        axs.set_yticklabels(["decoder", "authors"], fontsize=9); axs.set_xlabel("session time (s)"); axs.spines[["top", "right"]].set_visible(False)
        for jj, sec in enumerate(secs):
            axs.add_patch(Rectangle((sec, 1.05), 1, 0.9, color=STATE_HEX.get(int(tr[jj]), "#dddddd"), lw=0))
            axs.add_patch(Rectangle((sec, 0.05), 1, 0.9, color=STATE_HEX.get(int(pr[jj]), "#dddddd"), lw=0))
        axs.axvline(t, color="black", lw=1.5)
        for k, (name, code) in enumerate((("WAKE", 1), ("nonREM", 2), ("REM", 3), ("microarousal", 4))):
            fig2.text(0.08 + 0.11 * k, 0.012, f"■ {name}", color=STATE_HEX[code], fontsize=9)
        fig2.text(0.56, 0.012, "● excitatory neurons", color=TYPE_COLORS["E"], fontsize=9)
        fig2.text(0.75, 0.012, "● inhibitory neurons", color=TYPE_COLORS["I"], fontsize=9)
        fig2.savefig(fdir / f"f{i:04d}.png", dpi=90)
        if i == still_i:
            fig2.savefig(ROOT / "figures" / "brain3d_still.png", dpi=120)
        plt.close(fig2)
        if "--still-only" in sys.argv:
            return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", str(fdir / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(OUT / "brain3d.mp4")], check=True)
    for f in fdir.glob("*.png"):
        f.unlink()
    fdir.rmdir()
    units.to_csv(OUT / "unit_positions.csv", index=False)
    print("wrote", OUT / "brain3d.html", OUT / "brain3d.mp4")


if __name__ == "__main__":
    main()
