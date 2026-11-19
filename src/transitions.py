"""Firing around state transitions, per cell type, with the paper's synchrony measure.

For every session I compute three 1 s traces from the authors' spikes:
  E rate      mean spikes per excitatory cell per second
  I rate      mean spikes per inhibitory cell per second
  synchrony   Watson et al. 2016 definition: fraction of excitatory cells firing at least
              one spike in a 50 ms window, stepped every 25 ms, averaged within the second
and align them (+/- 60 s) to four kinds of boundary taken from the authors' scoring:
  microarousal onset, wake-interruption onset (many events), and the full WAKE->SLEEP
  and SLEEP->WAKE switches (few events; descriptive only).

Each session's event-average is expressed relative to its own pre-event baseline
(the mean from -60 to -30 s) so sessions with different firing rates can be averaged;
sessions are then averaged with equal weight. "Lead" is read from the time at which each
trace first reaches half of its eventual change, with a bootstrap over sessions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import SessionData, load_all, ROOT, BINS_PER_S

OUT = ROOT / "results" / "transitions"
HALF = 60                      # seconds each side of the boundary
BASE = (-60, -30)              # baseline window, seconds relative to the boundary
SYNC_WIN, SYNC_STEP = 0.050, 0.025
EVENT_KINDS = ["MA onset", "WI onset", "WAKE->SLEEP", "SLEEP->WAKE"]
TRACES = ["E rate", "I rate", "synchrony"]


def rate_traces(sd: SessionData) -> dict[str, np.ndarray]:
    """Per-second E and I rates (spikes per cell per s) from the 10 ms bins."""
    n_sec = sd.bins.shape[1] // BINS_PER_S
    out = {}
    for name, code in (("E rate", 0), ("I rate", 1)):
        rows = np.where(sd.types == code)[0]
        if rows.size == 0:
            out[name] = np.full(n_sec, np.nan); continue
        per_sec = np.zeros(n_sec)
        for r in rows:                                   # one row at a time keeps memory small
            per_sec += np.asarray(sd.bins[r, :n_sec * BINS_PER_S]).reshape(n_sec, BINS_PER_S).sum(1)
        out[name] = per_sec / rows.size
    return out


def synchrony_trace(session: str, sd: SessionData) -> np.ndarray:
    """Paper synchrony per 25 ms step, then averaged within each second."""
    sp = np.load(ROOT / "data" / "processed" / session / "spikes.npz")
    off, times = sp["unit_offsets"], sp["spike_times"]
    n_sec = sd.bins.shape[1] // BINS_PER_S
    n_win = int(n_sec / SYNC_STEP)
    e_rows = np.where(sd.types == 0)[0]
    active = np.zeros(n_win, dtype=np.int16)
    for r in e_rows:
        t = times[off[r]:off[r + 1]]
        k = np.floor(t / SYNC_STEP).astype(np.int64)      # a 50 ms window covers 2 steps
        hit = np.zeros(n_win, dtype=bool)
        for kk in (k, k - 1):
            kk = kk[(kk >= 0) & (kk < n_win)]
            hit[kk] = True
        active += hit
    frac = active / max(e_rows.size, 1)
    per_step = int(round(1 / SYNC_STEP))                 # 40 steps per second
    return frac[:n_sec * per_step].reshape(n_sec, per_step).mean(1)


def boundaries(sd: SessionData) -> dict[str, np.ndarray]:
    """Event onsets (s) from the authors' scoring."""
    st = dict(np.load(ROOT / "data" / "processed" / sd.session / "states.npz"))
    lab = sd.labels
    wake = lab == 1; sleep = np.isin(lab, (2, 3))
    # full switches: a WAKE second followed within 1 s by a sleep second, and the reverse
    ws = np.where(wake[:-2] & (sleep[1:-1] | sleep[2:]))[0] + 1
    sw = np.where(sleep[:-2] & (wake[1:-1] | wake[2:]))[0] + 1
    def first_of_runs(idx):                               # keep one onset per switch
        return idx[np.r_[True, np.diff(idx) > 2]] if idx.size else idx
    return {"MA onset": st["MA"][:, 0] if st["MA"].size else np.zeros(0),
            "WI onset": st["WI"][:, 0] if st["WI"].size else np.zeros(0),
            "WAKE->SLEEP": first_of_runs(ws).astype(float),
            "SLEEP->WAKE": first_of_runs(sw).astype(float)}


def align(trace: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """(n_events, 121) windows at 1 s resolution; events too close to the edges are dropped."""
    out = []
    for t0 in onsets:
        c = int(np.floor(t0))
        if c - HALF >= 0 and c + HALF + 1 <= trace.size:
            out.append(trace[c - HALF:c + HALF + 1])
    return np.array(out) if out else np.zeros((0, 2 * HALF + 1))


def half_change_time(mean_curve: np.ndarray) -> float:
    """First time (s, relative to onset) the curve covers half the distance from baseline to its
    post-onset extreme. Baseline = mean over BASE; extreme = max |deviation| in [0, +30] s."""
    t = np.arange(-HALF, HALF + 1)
    base = mean_curve[(t >= BASE[0]) & (t < BASE[1])].mean()
    post = mean_curve[(t >= 0) & (t <= 30)]
    dev = post - base
    k = np.argmax(np.abs(dev)); target = base + dev[k] / 2
    seg = mean_curve[(t >= -30) & (t <= t[(t >= 0) & (t <= 30)][k])]
    tt = t[(t >= -30) & (t <= t[(t >= 0) & (t <= 30)][k])]
    crossed = np.where((seg - target) * np.sign(dev[k]) >= 0)[0]
    return float(tt[crossed[0]]) if crossed.size else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    per_session = {}          # kind -> trace -> list of (session, normalized mean curve)
    counts = []
    for s, sd in data.items():
        tr = rate_traces(sd); tr["synchrony"] = synchrony_trace(s, sd)
        bd = boundaries(sd)
        for kind, onsets in bd.items():
            n_used = 0
            for name in TRACES:
                w = align(tr[name], onsets)
                if w.shape[0] == 0 or np.isnan(w).any():
                    continue
                m = w.mean(0); n_used = w.shape[0]
                t = np.arange(-HALF, HALF + 1)
                base = m[(t >= BASE[0]) & (t < BASE[1])].mean()
                if base <= 0:
                    continue
                per_session.setdefault(kind, {}).setdefault(name, []).append((s, m / base))
            counts.append(dict(session=s, rat=sd.rat, kind=kind, n_events=int(n_used)))
    pd.DataFrame(counts).to_csv(OUT / "event_counts.csv", index=False)
    rng = np.random.default_rng(0)
    summary, curves = [], {}
    for kind in EVENT_KINDS:
        for name in TRACES:
            items = per_session.get(kind, {}).get(name, [])
            if not items:
                continue
            M = np.array([c for _, c in items])
            mean = M.mean(0); sd_ = M.std(0)
            curves[f"{kind}|{name}"] = dict(mean=mean.tolist(), sd=sd_.tolist(), n_sessions=len(items))
            lat = half_change_time(mean)
            boot = [half_change_time(M[rng.integers(0, len(M), len(M))].mean(0)) for _ in range(500)]
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            t = np.arange(-HALF, HALF + 1)
            post = mean[(t >= 0) & (t <= 30)]; base = mean[(t >= BASE[0]) & (t < BASE[1])].mean()
            change = post[np.argmax(np.abs(post - base))] / base - 1
            summary.append(dict(event=kind, trace=name, n_sessions=len(items), peak_change_pct=100 * change,
                                half_change_s=lat, ci_low=lo, ci_high=hi))
    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "curves.json").write_text(json.dumps(curves))
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
