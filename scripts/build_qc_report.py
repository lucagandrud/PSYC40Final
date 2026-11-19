"""Phase 1d: cross-session QC report (reports/qc_report.html) from reports/qc_*.csv.

Usage: python -I scripts/build_qc_report.py
Regenerated automatically after every session is processed (scripts/on_session_ready.sh).
"""
from __future__ import annotations

import base64
import html
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.reference import TABLE_S1, STATUS_COLORS, STATE_COLORS, TYPE_COLORS

REPORTS, FIGS, PROC = ROOT / "reports", ROOT / "figures" / "qc", ROOT / "data" / "processed"
SHORT = {
    "Files: files match authors' file list": "files",
    "Units: E/I labels consistent across authors' files": "E/I labels",
    "Units: inhibitory units >= 3": "≥3 I units",
    "Spikes: sorted, non-negative, within recording, no duplicates": "spike times",
    "Spikes: refractory period (authors' sorting criterion: >1 ms)": "refractory",
    "Spikes: units with zero spikes in an analyzed state": "silent units",
    "Spikes: longest stretch with no spikes from any unit": "data gaps",
    "Spikes: spike data covers the recording files": "spike coverage",
    "States: intervals well-formed (start < stop, non-negative)": "intervals",
    "States: scored states lie within the spike recording": "states in recording",
    "States: state types do not overlap": "overlaps",
    "States: usable WAKE and SLEEP time exists": "usable time",
    "States: transitions and events": "transitions",
    "Physiology: median rates vs Watson 2016 (comparison only)": "rates vs paper",
    "Anatomy: every unit maps to one region via its shank": "region map",
    "Anatomy: region agrees with paper Table S1": "region vs Table S1",
    "Flags: authors' notes applied": "authors' notes",
}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})


def heatmap(tab: pd.DataFrame, out: Path):
    cols = [c for c in SHORT if c in tab.columns]
    code = {"PASS": 0, "WARN": 1, "FAIL": 2}
    M = np.array([[code.get(v, 3) for v in row] for row in tab[cols].astype(str).to_numpy()], float)
    fig, ax = plt.subplots(figsize=(1.2 + 0.62 * len(cols), 0.9 + 0.32 * len(tab)), facecolor="#fcfcfb")
    from matplotlib.colors import ListedColormap
    ax.imshow(M, cmap=ListedColormap([STATUS_COLORS["PASS"], STATUS_COLORS["WARN"], STATUS_COLORS["FAIL"], "#e6e5e0"]),
              vmin=0, vmax=3, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, ["✓", "!", "✗", "–"][int(M[i, j])], ha="center", va="center", fontsize=9,
                    color="#0b0b0b" if M[i, j] in (1, 3) else "white", fontweight="bold")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([SHORT[c] for c in cols], rotation=40, ha="right")
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"{s}  ({r})" for s, r in zip(tab.session, tab.regions)])
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(tab)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", lw=2); ax.tick_params(which="both", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("QC checks per session   ✓ PASS   ! WARN   ✗ FAIL", loc="left", fontsize=10)
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)


def overview(tab: pd.DataFrame, out: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.9 + 0.32 * len(tab)), sharey=True, facecolor="#fcfcfb")
    y = np.arange(len(tab))
    a1.barh(y, tab.n_E, color=TYPE_COLORS["E"], height=0.7, label="excitatory")
    a1.barh(y, tab.n_I, left=tab.n_E + 0.4, color=TYPE_COLORS["I"], height=0.7, label="inhibitory")
    for yi, (e, i) in enumerate(zip(tab.n_E, tab.n_I)):
        a1.text(e + i + 1.5, yi, f"{e} E / {i} I", va="center", fontsize=8, color="#52514e")
    a1.set_yticks(y); a1.set_yticklabels([f"{s}  ({r})" for s, r in zip(tab.session, tab.regions)])
    a1.invert_yaxis(); a1.set_xlabel("stable units"); a1.legend(frameon=False, loc="lower right")
    a1.set_title("Units per session (authors' stable set)", loc="left", fontsize=10)
    left = np.zeros(len(tab))
    for k in ["WAKE", "nonREM", "REM"]:
        v = tab[f"min_{k}"].to_numpy(float)
        a2.barh(y, v, left=left, color=STATE_COLORS[k], height=0.7, label=k)
        left += v + 0.6
    for yi, (w, sl) in enumerate(zip(tab.min_usable_WAKE, tab.min_usable_SLEEP)):
        a2.text(left[yi] + 2, yi, f"{w:.0f} / {sl:.0f} min", va="center", fontsize=8, color="#52514e")
    a2.set_xlabel("usable minutes (after exclusions)"); a2.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.04), ncol=3)
    a2.set_title("Usable scored time: WAKE / SLEEP (nonREM + REM)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)


def table_s1_compare(tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rat, g in tab.groupby("rat"):
        ref = TABLE_S1.get(rat)
        if not ref: continue
        complete = len(g) == ref["n_sessions"]
        mE, mI = g.n_E.mean(), g.n_I.mean()
        if complete:
            verdict = "MATCH" if abs(mE - ref["pE"]) < 0.06 and abs(mI - ref["pI"]) < 0.06 else "MISMATCH"
        else:
            verdict = f"partial ({len(g)} of {ref['n_sessions']} sessions)"
        rows.append(dict(rat=rat, sessions=", ".join(g.session), **{"E/session (mine)": round(mE, 1),
                    "E/session (Table S1)": ref["pE"], "I/session (mine)": round(mI, 1),
                    "I/session (Table S1)": ref["pI"], "verdict": verdict}))
    return pd.DataFrame(rows)


def simultaneity(tab: pd.DataFrame) -> list[str]:
    out = []
    sess = tab.session.tolist()
    for s in sess:
        if not s.endswith("_ACC"): continue
        m = s[:-4] + "_mPFC"
        if m not in sess: continue
        ma, mb = (json.loads((PROC / x / "meta.json").read_text()) for x in (s, m))
        sa, sb = (np.load(PROC / x / "states.npz") for x in (s, m))
        rfi_same = np.allclose(np.asarray(ma["recording_file_intervals"], float),
                               np.asarray(mb["recording_file_intervals"], float), equal_nan=True) \
            if np.shape(ma["recording_file_intervals"]) == np.shape(mb["recording_file_intervals"]) else False
        dur_same = abs(ma["duration_s"] - mb["duration_s"]) < 1
        # direct test: same clock => population activity co-fluctuates at ~zero lag
        sp = {}
        bw, T = 0.025, min(ma["duration_s"], mb["duration_s"])
        for x in (s, m):
            z = np.load(PROC / x / "spikes.npz")["spike_times"]
            c = np.bincount((z[z < T] / bw).astype(int), minlength=int(T / bw))[:int(T / bw)].astype(float)
            sp[x] = c - c.mean()
        pa, pb = sp[s], sp[m]
        den = np.sqrt(pa @ pa * (pb @ pb))
        lags = np.arange(-40, 41)
        cc = np.array([pa[max(0, l):len(pa) + min(0, l)] @ pb[max(0, -l):len(pb) - max(0, l)] / den for l in lags])
        i = int(np.argmax(cc)); far = max(cc[0], cc[-1], cc[20], cc[60])
        peak_ok = abs(lags[i] * bw) <= 0.05 and cc[i] > 3 * far
        la = np.load(PROC / s / "labels_1s.npz")["labels"]; lb = np.load(PROC / m / "labels_1s.npz")["labels"]
        n = min(len(la), len(lb)); both = (la[:n] > 0) & (lb[:n] > 0)
        agree = 100 * float((la[:n][both] == lb[:n][both]).mean()) if both.any() else float("nan")
        ok = rfi_same and dur_same and peak_ok
        out.append(f"<b>{s} + {m}</b>: {'CONFIRMED shared clock' if ok else 'NOT confirmed'} "
                   f"(recording-file intervals identical: {rfi_same}; last spike {ma['duration_s']:.2f} vs "
                   f"{mb['duration_s']:.2f} s; population cross-correlation peak r={cc[i]:.3f} at "
                   f"{lags[i]*bw*1000:+.0f} ms vs ≤{far:.3f} at ±0.5–1 s). The authors scored each probe "
                   f"separately: labels agree on {agree:.1f}% of seconds scored in both.")
    return out


def main():
    tab = pd.read_csv(REPORTS / "qc_table.csv")
    chk = pd.read_csv(REPORTS / "qc_checks.csv")
    tab["order"] = tab.rat.astype(str) + tab.session
    tab = tab.sort_values("order").reset_index(drop=True)
    style()
    heatmap(tab, FIGS / "_checks_heatmap.png")
    overview(tab, FIGS / "_overview.png")
    s1 = table_s1_compare(tab)
    sim = simultaneity(tab)
    cleared = tab[tab.cleared_for_modeling]
    flags = chk[chk.status != "PASS"].merge(tab[["session", "rat"]], on="session").sort_values(["rat", "session"])

    def tiles():
        items = [("sessions cleared", f"{len(cleared)} / {len(tab)}"), ("rats", f"{cleared.rat.nunique()}"),
                 ("excitatory units", f"{int(cleared.n_E.sum())}"), ("inhibitory units", f"{int(cleared.n_I.sum())}"),
                 ("usable WAKE", f"{cleared.min_usable_WAKE.sum()/60:.1f} h"),
                 ("usable SLEEP", f"{cleared.min_usable_SLEEP.sum()/60:.1f} h")]
        return "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in items)

    summ_cols = ["session", "rat", "regions", "n_E", "n_I", "min_usable_WAKE", "min_nonREM", "min_REM",
                 "n_WAKE_to_SLEEP", "n_SLEEP_to_WAKE", "n_MA", "n_WI", "pct_seconds_scored", "cleared_for_modeling"]
    summ = tab[summ_cols].rename(columns={"n_E": "E", "n_I": "I", "min_usable_WAKE": "WAKE min",
                                          "min_nonREM": "nonREM min", "min_REM": "REM min",
                                          "n_WAKE_to_SLEEP": "WAKE→SLEEP", "n_SLEEP_to_WAKE": "SLEEP→WAKE",
                                          "n_MA": "MA onsets", "n_WI": "WI onsets",
                                          "pct_seconds_scored": "% s scored", "cleared_for_modeling": "cleared"})
    notes = tab[["session", "authors_note"] + (["note_verdict"] if "note_verdict" in tab else [])].fillna("")

    flag_html = "".join(
        f'<li><span class="badge {r.status}">{r.status}</span> <b>{html.escape(r.session)}</b> · '
        f'{html.escape(r.area)}: {html.escape(r.check)} — {html.escape(str(r.detail))}</li>'
        for r in flags.itertuples()) or "<li>None.</li>"
    panels = "".join(
        f'<details><summary><b>{r.session}</b> · rat {r.rat} · {r.regions} · {r.n_E} E / {r.n_I} I · '
        f'{"cleared" if r.cleared_for_modeling else "BLOCKED"}</summary>'
        f'<img src="data:image/png;base64,{b64(FIGS / (r.session + ".png"))}"/>'
        f'{chk[chk.session == r.session][["area","check","status","detail"]].to_html(index=False, classes="t")}'
        f'</details>' for r in tab.itertuples() if (FIGS / (r.session + ".png")).exists())

    css = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--card:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--card:#232322}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--card:#232322}
body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:24px 16px;max-width:1280px;margin:auto}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--ink2)} .tiles{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:120px}
.tile .v{font-size:22px;font-weight:600} .tile .k{color:var(--ink2);font-size:12px}
img{max-width:100%;background:#fcfcfb;border-radius:6px;border:1px solid var(--line)}
table.t{border-collapse:collapse;font-size:12.5px;margin:8px 0;width:100%} .t th,.t td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
.badge{display:inline-block;padding:0 6px;border-radius:4px;font-size:11px;font-weight:700;color:#fff}
.badge.WARN{background:#fab219;color:#0b0b0b} .badge.FAIL{background:#d03b3b} .badge.PASS{background:#0ca30c}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:8px 0}
summary{cursor:pointer} ul{padding-left:20px} li{margin:3px 0}
"""
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSYC40 QC Report</title><style>{css}</style></head><body>
<h1>Data quality report: CRCNS fcx-1 (Watson et al. 2016)</h1>
<div class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} · {len(tab)} sessions processed · checks never drop units; exclusions come only from the authors' notes.</div>
<div class="tiles">{tiles()}</div>
<h2>What needs attention</h2><ul>{flag_html}</ul>
<h2>Check matrix</h2><img src="data:image/png;base64,{b64(FIGS / '_checks_heatmap.png')}"/>
<h2>Units and usable time</h2><img src="data:image/png;base64,{b64(FIGS / '_overview.png')}"/>
<h2>Unit counts vs paper Table S1 (per rat)</h2>{s1.to_html(index=False, classes="t") if len(s1) else "<p>none yet</p>"}
<h2>Sessions</h2>{summ.to_html(index=False, classes="t")}
<h2>Authors' notes and exclusions</h2>{notes.to_html(index=False, classes="t")}
<h2>Same-day ACC/mPFC pairs (shared clock?)</h2><ul>{''.join(f'<li>{x}</li>' for x in sim) or '<li>No complete pair downloaded yet.</li>'}</ul>
<h2>Per-session panels</h2><p class="sub">Each panel: authors' scoring over the session, mean rate per cell, 120 s raster across a scored WAKE→SLEEP boundary (E green, I violet), and per-state firing-rate distributions with the paper's medians (dotted).</p>
{panels}
</body></html>"""
    (REPORTS / "qc_report.html").write_text(doc)
    print(f"wrote {REPORTS / 'qc_report.html'}: {len(tab)} sessions, {len(cleared)} cleared, {len(flags)} flags")


if __name__ == "__main__":
    main()
