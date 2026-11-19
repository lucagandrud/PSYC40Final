"""Turn a loaded session into the processed format used by every later step.

What this does (and nothing more):
- unit table: the authors' type, shank and spike count per unit, plus the region lookup
  and firing rate per state (spike count / seconds the authors scored as that state).
- 1 s label vector: the authors' state intervals written out one entry per second.
  Seconds no interval covers stay UNSCORED. Seconds covered by two state types are
  marked AMBIGUOUS (no precedence rule) and are never used for modeling.
- exclusion mask: time windows taken from the authors' own notes (see reference.NOTE_WINDOWS).
- usable mask: WAKE / nonREM / REM seconds that are not excluded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .reference import NOTE_WINDOWS, REGION_ALIASES

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

LABEL_CODES = {"UNSCORED": 0, "WAKE": 1, "nonREM": 2, "REM": 3, "MA": 4, "WI": 5, "AMBIGUOUS": 9}
MODEL_STATES = ("WAKE", "nonREM", "REM")


def recording_file_end(s: dict) -> float | None:
    """End of the last recording file (RecordingFileIntervals), if finite."""
    rfi = np.asarray(s["meta"]["recording_file_intervals"], float)
    return float(rfi[:, 1].max()) if rfi.size and np.isfinite(rfi[:, 1].max()) else None


def session_duration(s: dict) -> float:
    """End of the spike data in seconds = the last spike from any unit.
    Seconds after it have no spike data (in some sessions the authors' state scoring and
    even the recording files run longer than the spike data), so they are never used.
    QC compares this with the recording-file end and reports any shortfall."""
    return float(max(t.max() for t in s["spikes"] if t.size))


def states_beyond_recording(s: dict, end: float) -> dict:
    """Seconds of each scored state that lie after the spike recording ends."""
    out = {}
    for k, iv in s["states"].items():
        if len(iv):
            n = float(np.clip(iv[:, 1] - np.maximum(iv[:, 0], end), 0, None).sum())
            if n > 0:
                out[k] = n
    return out


def interval_seconds(iv: np.ndarray, n_sec: int) -> np.ndarray:
    """Boolean per-second mask: second k (covering [k, k+1)) is in an interval [a, b)
    if a <= k and k + 1 <= b. The authors' intervals are whole seconds, so this is exact."""
    m = np.zeros(n_sec, bool)
    for a, b in iv:
        lo, hi = int(np.ceil(a)), int(np.floor(b))
        if hi > lo:
            m[max(lo, 0):min(hi, n_sec)] = True
    return m


def label_vector(s: dict, n_sec: int) -> tuple[np.ndarray, dict]:
    masks = {k: interval_seconds(s["states"][k], n_sec) for k in ["WAKE", "nonREM", "REM", "MA", "WI"]}
    n_cover = sum(m.astype(int) for m in masks.values())
    lab = np.zeros(n_sec, np.int8)
    for k, m in masks.items():
        lab[m & (n_cover == 1)] = LABEL_CODES[k]
    lab[n_cover > 1] = LABEL_CODES["AMBIGUOUS"]
    overlaps = {}
    names = list(masks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = int((masks[a] & masks[b]).sum())
            if n:
                overlaps[f"{a}&{b}"] = n
    return lab, overlaps


def note_exclusion(s: dict, n_sec: int) -> tuple[np.ndarray, dict]:
    """Exclusion mask from the authors' notes, plus a record of what was applied."""
    sess = s["session"]
    excl = np.zeros(n_sec, bool)
    rec = dict(applied=False, rule=None, cut_s=None, why=None, seconds_excluded=0)
    w = NOTE_WINDOWS.get(sess)
    if not w:
        return excl, rec
    rec["why"] = w["why"]
    if "exclude_from_s" in w:
        cut = w["exclude_from_s"]
        rec["rule"] = f"exclude t >= {cut:g} s"
    else:
        a, b = w["injection_between_files"]
        rfi = np.asarray(s["meta"]["recording_file_intervals"], float)
        if len(rfi) < a:
            rec["rule"] = f"injection between files {a}&{b}, but only {len(rfi)} file intervals exist"
            rec["problem"] = True
            return excl, rec
        cut = float(rfi[a - 1, 1])  # end of recording file a
        rec["rule"] = f"injection between files {a}&{b}: exclude t >= end of file {a} ({cut:.1f} s)"
    rec["cut_s"] = float(cut)
    excl[int(np.floor(cut)):] = True
    rec["applied"] = True
    rec["seconds_excluded"] = int(excl.sum())
    return excl, rec


def counts_per_second(spikes: list[np.ndarray], n_sec: int) -> np.ndarray:
    out = np.zeros((len(spikes), n_sec), np.int32)
    for i, t in enumerate(spikes):
        k = np.floor(t).astype(int)
        k = k[(k >= 0) & (k < n_sec)]
        out[i] = np.bincount(k, minlength=n_sec)[:n_sec]
    return out


def unit_table(s: dict, counts: np.ndarray, lab: np.ndarray, usable_any: np.ndarray) -> pd.DataFrame:
    n = len(s["spikes"])
    ids = s["cellids"]
    E, I = set(ids["EAll"]), set(ids["IAll"])
    Edef, Idef = set(ids["EDefinite"]), set(ids["IDefinite"])
    rows = []
    for u in range(1, n + 1):
        typ = "E" if u in E else ("I" if u in I else "unclassified")
        src = ("definite (waveform + synaptic interaction)" if u in Edef | Idef
               else "waveform only" if typ != "unclassified" else "")
        sh = int(s["shank"][u - 1])
        chans = s["shank_channels"].get(sh, [])
        file_regs = sorted({s["ch_region"].get(c, "") for c in chans} - {""})
        regs = sorted({REGION_ALIASES.get(r, r) for r in file_regs})
        t = s["spikes"][u - 1]
        isi = np.diff(t)
        row = dict(session=s["session"], rat=s["meta"]["rat"], unit_id=u, type=typ, type_source=src,
                   shank=sh, cluster_on_shank=int(s["cell_ix"][u - 1]),
                   shank_channels=" ".join(map(str, chans)),
                   region=regs[0] if len(regs) == 1 else ("UNMAPPED" if not regs else "AMBIGUOUS:" + "/".join(regs)),
                   region_file_label="/".join(file_regs),
                   region_source="shank -> .xml spike group channels -> ChannelAnatomy.csv",
                   n_spikes=int(t.size),
                   isi_lt_1ms_frac=float((isi < 0.001).mean()) if isi.size else 0.0,
                   isi_lt_1p5ms_frac=float((isi < 0.0015).mean()) if isi.size else 0.0)
        for k in ["WAKE", "nonREM", "REM", "MA", "WI"]:
            m = (lab == LABEL_CODES[k]) & usable_any
            row[f"rate_{k}_hz"] = float(counts[u - 1, m].sum() / m.sum()) if m.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def preprocess(s: dict) -> dict:
    dur = session_duration(s)
    n_sec = int(np.ceil(dur))
    lab, overlaps = label_vector(s, n_sec)
    excl, note = note_exclusion(s, n_sec)
    usable = np.isin(lab, [LABEL_CODES[k] for k in MODEL_STATES]) & ~excl
    counts = counts_per_second(s["spikes"], n_sec)
    units = unit_table(s, counts, lab, ~excl)
    beyond = states_beyond_recording(s, dur)
    allsp = np.sort(np.concatenate([t for t in s["spikes"] if t.size]))
    gaps = np.diff(np.r_[0.0, allsp])          # start gap + internal gaps (data end = last spike)
    g = int(np.argmax(gaps))
    longest_gap = dict(seconds=float(gaps[g]), start_s=float(np.r_[0.0, allsp][g]))
    return dict(duration_s=dur, rec_file_end_s=recording_file_end(s), n_sec=n_sec,
                labels=lab, overlaps=overlaps, excluded=excl,
                note=note, usable=usable, units=units, counts=counts,
                states_beyond=beyond, longest_gap=longest_gap)


def save(s: dict, pp: dict, out_root: Path = PROCESSED) -> Path:
    out = out_root / s["session"]
    out.mkdir(parents=True, exist_ok=True)
    lens = np.array([t.size for t in s["spikes"]])
    np.savez_compressed(out / "spikes.npz",
                        spike_times=np.concatenate(s["spikes"]) if lens.sum() else np.zeros(0),
                        unit_offsets=np.concatenate([[0], np.cumsum(lens)]))
    pp["units"].to_csv(out / "units.csv", index=False)
    np.savez_compressed(out / "labels_1s.npz", labels=pp["labels"], excluded=pp["excluded"],
                        usable=pp["usable"], codes=json.dumps(LABEL_CODES))
    np.savez_compressed(out / "states.npz", **{k: v for k, v in s["states"].items()},
                        **{k: v for k, v in s["episodes"].items()})
    np.savez_compressed(out / "events.npz", **s["events"])
    meta = dict(s["meta"], duration_s=pp["duration_s"], recording_file_end_s=pp["rec_file_end_s"],
                n_seconds=pp["n_sec"],
                state_overlap_seconds=pp["overlaps"], note_exclusion=pp["note"],
                state_seconds_after_recording_end=pp["states_beyond"],
                longest_gap_without_spikes=pp["longest_gap"],
                label_codes=LABEL_CODES,
                regions=sorted(pp["units"].region.unique().tolist()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return out


def load_processed(session: str, root: Path = PROCESSED) -> dict:
    """Read back a processed session: spike trains, unit table, 1 s labels, meta."""
    d = root / session
    sp = np.load(d / "spikes.npz")
    off = sp["unit_offsets"]
    trains = [sp["spike_times"][off[i]:off[i + 1]] for i in range(len(off) - 1)]
    lab = np.load(d / "labels_1s.npz")
    return dict(session=session, spikes=trains, units=pd.read_csv(d / "units.csv"),
                labels=lab["labels"], excluded=lab["excluded"], usable=lab["usable"],
                states=dict(np.load(d / "states.npz")), events=dict(np.load(d / "events.npz")),
                meta=json.loads((d / "meta.json").read_text()))
