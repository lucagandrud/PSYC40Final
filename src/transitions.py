"""Firing around state transitions, per cell type, with the paper's synchrony measure.

For every session I compute three 1 s traces from the authors' spikes:
  E rate      mean spikes per excitatory cell per second
  I rate      mean spikes per inhibitory cell per second
  synchrony   Watson et al. 2016 definition: fraction of excitatory cells firing at least
              one spike in a 50 ms window, stepped every 25 ms, averaged within the second
and align them (+/- 60 s) to four kinds of boundary taken from the authors' scoring:
  microarousal onset, wake-interruption onset, and the full WAKE->SLEEP and SLEEP->WAKE
  switches (few events; descriptive only).
Only microarousals and wake interruptions that directly follow scored sleep count (a
nonREM or REM second within 2 s before onset, and at least 80% of the preceding 30 s
scored as sleep): 227 of 1738 MAs and 288 of 1769 WIs. The rest sit in stretches the
authors left unscored, where no baseline state is defined.

Each session's event-average is expressed relative to its own pre-event baseline
(the mean from -60 to -30 s) so sessions with different firing rates can be compared;
the figure shows the median and interquartile range across sessions (a median, because
a few sessions have near-silent inhibitory cells and a tiny baseline). Onset latency is
the first second, from -30 s on, at which the median curve leaves its baseline by more
than twice the baseline's own standard deviation for three consecutive seconds.
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
    def after_sleep(onsets):
        keep = []
        for t0 in onsets:
            a = int(np.floor(t0))
            if a >= 30 and sleep[a - 2:a].any() and sleep[a - 30:a].mean() >= 0.8:
                keep.append(t0)
        return np.array(keep)
    return {"MA onset": after_sleep(st["MA"][:, 0]) if st["MA"].size else np.zeros(0),
            "WI onset": after_sleep(st["WI"][:, 0]) if st["WI"].size else np.zeros(0),
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


def onset_latency(curve: np.ndarray) -> float:
    """First second (from -30 s on) at which the curve leaves its baseline by more than
    2 x the baseline's own sd for 3 consecutive seconds; NaN if it never does."""
    t = np.arange(-HALF, HALF + 1)
    seg = curve[(t >= BASE[0]) & (t < BASE[1])]
    base, noise = seg.mean(), seg.std()
    out = np.abs(curve - base) > 2 * max(noise, 1e-9)
    for i in np.where(t >= -30)[0]:
        if out[i:i + 3].all():
            return float(t[i])
    return float("nan")


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
            med = np.median(M, 0); q25, q75 = np.percentile(M, [25, 75], axis=0)
            curves[f"{kind}|{name}"] = dict(median=med.tolist(), q25=q25.tolist(), q75=q75.tolist(), n_sessions=len(items))
            lat = onset_latency(med)
            boot = [onset_latency(np.median(M[rng.integers(0, len(M), len(M))], 0)) for _ in range(300)]
            lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan)
            t = np.arange(-HALF, HALF + 1)
            change_0_10 = med[(t >= 0) & (t < 10)].mean() - 1
            summary.append(dict(event=kind, trace=name, n_sessions=len(items), median_change_0_10s_pct=100 * change_0_10,
                                onset_latency_s=lat, ci_low=lo, ci_high=hi))
    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "curves.json").write_text(json.dumps(curves))
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
