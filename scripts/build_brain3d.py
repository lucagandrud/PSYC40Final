"""Build the rotatable 3D model (Plotly HTML) and the mp4 for the Dino_061914 ACC + mPFC pair.

Window: 90 s across the authors' WAKE -> SLEEP switch in this pair (WAKE until ~10767 s,
then sleep), at 100 ms frames. Each unit's marker swells with the spikes it fired in the
current frame (decaying over three frames). Colors: E green, I purple (the authors' labels).
Outputs: results/brain3d/brain3d.html, results/brain3d/brain3d.mp4, figures/brain3d_still.png
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.brain3d import OUT, REGION_MAP, sample_region_points, place_units, spike_frames, state_strip  # noqa: E402
from src.reference import TYPE_COLORS, STATE_COLORS  # noqa: E402

PAIR = {"Dino_061914_mPFC": "mPFC", "Dino_061914_ACC": "ACC"}
T0, T1, STEP = 10727.0, 10817.0, 0.1          # 40 s of WAKE, 50 s of sleep (switch at 10767 s)
STATE_NAME = {0: "unscored", 1: "WAKE", 2: "nonREM", 3: "REM", 4: "MA", 5: "WI", 9: "ambiguous"}
STATE_HEX = {1: STATE_COLORS["WAKE"], 2: STATE_COLORS["nonREM"], 3: STATE_COLORS["REM"], 4: STATE_COLORS["MA"], 5: STATE_COLORS["WI"], 0: "#dddddd", 9: "#dddddd"}


def main():
    pts = np.load(sample_region_points())
    units, frames, times = [], [], None
    for s, region in PAIR.items():
        u = place_units(s, pts[f"pts_{region}"]); u["session"] = s
        t, x = spike_frames(s, T0, T1, STEP)
        units.append(u); frames.append(x); times = t
    units = pd.concat(units, ignore_index=True); X = np.concatenate(frames, axis=1)     # (frames, units)
    pulse = X.astype(float).copy()
    for k in (1, 2):                                                  # decay over three frames
        pulse[k:] += X[:-k] * (0.5 ** k)
    size = 4 + 5 * np.minimum(pulse, 3)
    strip = {s: state_strip(s, T0, T1) for s in PAIR}
    # ---------------------------------------------------------------- Plotly
    fig = go.Figure()
    for region, acronyms in REGION_MAP.items():
        if region not in PAIR.values():
            continue
        for ac in acronyms:
            v, f = pts[f"mesh_{ac}_v"], pts[f"mesh_{ac}_f"]
            keep = v[:, 2] > np.median(pts["pts_mPFC"][:, 2]) - 3000       # right hemisphere
            fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 2], z=-v[:, 1], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                    opacity=0.12, color="#9aa6b2" if region == "mPFC" else "#c9a27e", name=f"{region}: {ac}",
                                    hoverinfo="name", showlegend=True))
    colors = units.type.map(TYPE_COLORS).to_numpy()
    hover = [f"{r.session}<br>unit {r.unit_id} ({r.type}), shank {r.shank}, {r.region}" for r in units.itertuples()]
    fig.add_trace(go.Scatter3d(x=units.ap, y=units.rl, z=-units.si, mode="markers", name="units",
                               marker=dict(size=size[0], color=colors, opacity=0.95, line=dict(width=0)),
                               text=hover, hoverinfo="text"))
    pl_frames = []
    for i, t in enumerate(times):
        sec = int(np.floor(t))
        st = {s: STATE_NAME[int(strip[s]["true"][sec - int(T0)])] for s in PAIR}
        pl_frames.append(go.Frame(data=[go.Scatter3d(marker=dict(size=size[i], color=colors))], traces=[len(fig.data) - 1], name=f"{t:.1f}",
                                  layout=go.Layout(title_text=f"t = {t - T0:5.1f} s   authors' state: mPFC {st['Dino_061914_mPFC']}, ACC {st['Dino_061914_ACC']}")))
    fig.frames = pl_frames
    fig.update_layout(
        title="Dino_061914: mPFC + ACC units across a WAKE → SLEEP switch (schematic positions inside atlas regions)",
        scene=dict(xaxis_title="anterior ← AP (µm)", yaxis_title="RL (µm)", zaxis_title="depth (µm)", aspectmode="data",
                   camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))),
        legend=dict(x=0.01, y=0.99), margin=dict(l=0, r=0, t=60, b=0), height=720,
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02,
                          buttons=[dict(label="▶ play", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                                   dict(label="■ pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.1, len=0.85, currentvalue=dict(prefix="time (s): "),
                      steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=f"{float(f.name) - T0:.0f}") for f in pl_frames[::10]])])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT / "brain3d.html", include_plotlyjs="cdn", auto_play=False)
    # ---------------------------------------------------------------- mp4 via matplotlib frames
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fdir = OUT / "frames"; fdir.mkdir(exist_ok=True)
    for f in fdir.glob("*.png"):
        f.unlink()
    secs = np.arange(int(T0), int(T1))
    for i, t in enumerate(times):
        fig2 = plt.figure(figsize=(10, 7.6))
        ax = fig2.add_axes([0.0, 0.2, 1.0, 0.74], projection="3d")
        for region, acronyms in REGION_MAP.items():
            if region not in PAIR.values():
                continue
            p = pts[f"pts_{region}"][::2]
            ax.scatter(p[:, 0], p[:, 2], -p[:, 1], s=9, color="#8fa3b8" if region == "mPFC" else "#d4a97a", alpha=0.08, linewidths=0, depthshade=False)
        ax.scatter(units.ap, units.rl, -units.si, s=size[i] ** 2 * 0.35, c=colors, alpha=0.95, linewidths=0, depthshade=False)
        ax.view_init(elev=22, azim=-60 + 0.25 * i)
        ax.set_xlabel("AP (µm)"); ax.set_ylabel("RL (µm)"); ax.set_zlabel("depth (µm)")
        ax.set_title(f"Dino_061914: mPFC (blue-grey volume) + ACC (tan volume) units, t = {t - T0:5.1f} s", fontsize=12, pad=0)
        axs = fig2.add_axes([0.08, 0.075, 0.84, 0.1]); axs.set_xlim(T0, T1); axs.set_ylim(0, 2); axs.set_yticks([0.5, 1.5])
        axs.set_yticklabels(["decoder", "authors"]); axs.set_xlabel("session time (s)"); axs.spines[["top", "right"]].set_visible(False)
        for s in PAIR:
            tr = strip[s]["true"]; pr = strip[s].get("pred")
            for j, sec in enumerate(secs):
                axs.add_patch(Rectangle((sec, 1.05), 1, 0.9, color=STATE_HEX.get(int(tr[j]), "#dddddd"), lw=0))
                if pr is not None:
                    axs.add_patch(Rectangle((sec, 0.05), 1, 0.9, color=STATE_HEX.get(int(pr[j]), "#dddddd"), lw=0))
            break                                                        # same labels for both probes within this window
        axs.axvline(t, color="black", lw=1.5)
        for k, (name, hexc) in enumerate((("WAKE", STATE_HEX[1]), ("nonREM", STATE_HEX[2]), ("REM", STATE_HEX[3]))):
            fig2.text(0.08 + 0.1 * k, 0.005, f"■ {name}", color=hexc, fontsize=9)
        fig2.text(0.6, 0.005, "● E", color=TYPE_COLORS["E"], fontsize=10); fig2.text(0.66, 0.005, "● I", color=TYPE_COLORS["I"], fontsize=10)
        fig2.savefig(fdir / f"f{i:04d}.png", dpi=90)
        if i == 400:
            fig2.savefig(ROOT / "figures" / "brain3d_still.png", dpi=120)
        if i == 400 and "--still-only" in sys.argv:
            plt.close(fig2); return
        plt.close(fig2)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", str(fdir / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(OUT / "brain3d.mp4")], check=True)
    for f in fdir.glob("*.png"):
        f.unlink()
    fdir.rmdir()
    units.to_csv(OUT / "unit_positions.csv", index=False)
    print("wrote", OUT / "brain3d.html", OUT / "brain3d.mp4", "pred strip:", all("pred" in v for v in strip.values()))


if __name__ == "__main__":
    main()
