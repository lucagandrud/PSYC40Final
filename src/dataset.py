"""Model input: each neuron's spike counts in 10 ms bins, kept as a time series.

For every processed session this module writes `bins10ms.npy`, a (neurons x bins) uint8
array of spike counts covering the whole recording (from 0 s to the last spike), and
`bins10ms_types.npy`, the authors' E/I label for each row (0 = E, 1 = I). Nothing else is
derived from the spikes: the model sees the raster itself.

One example is one labeled second. The network needs 0.63 s of real spikes on either
side of that second so that every 10 ms step inside it is read with its full
1.27 s reach, so the input window is 226 bins (2.26 s) and the prediction is for the
middle 100 bins. Seconds whose window would run past the recording are not used.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BIN_S = 0.010                 # 10 ms bins
BINS_PER_S = 100
HALF_REACH = 63               # bins of context each side (reach 127 bins = 1.27 s)
WINDOW = BINS_PER_S + 2 * HALF_REACH   # 226 bins per example

MODEL_CODES = {"WAKE": 1, "nonREM": 2, "REM": 3}
CLASS_NAMES = ["WAKE", "nonREM", "REM"]
TYPE_CODES = {"E": 0, "I": 1}


def sessions() -> list[str]:
    return sorted(p.name for p in PROCESSED.iterdir() if (p / "meta.json").exists())


def build_bins(session: str, force: bool = False) -> Path:
    """Bin the authors' spike times (unchanged) into 10 ms counts; one row per unit."""
    d = PROCESSED / session
    out = d / "bins10ms.npy"
    if out.exists() and not force:
        return out
    sp = np.load(d / "spikes.npz")
    off = sp["unit_offsets"]
    times = sp["spike_times"]
    meta = json.loads((d / "meta.json").read_text())
    n_bins = int(np.ceil(meta["duration_s"] / BIN_S)) + 1
    units = pd.read_csv(d / "units.csv")
    n_units = len(off) - 1
    assert len(units) == n_units, f"{session}: units.csv has {len(units)} rows, spikes.npz {n_units}"
    bins = np.zeros((n_units, n_bins), dtype=np.uint8)
    for i in range(n_units):
        t = times[off[i]:off[i + 1]]
        idx = np.floor(t / BIN_S).astype(np.int64)
        counts = np.bincount(idx, minlength=n_bins)
        assert counts.max() <= 255
        bins[i] = counts
    assert int(bins.sum()) == times.size, f"{session}: binned {bins.sum()} spikes, expected {times.size}"
    np.save(out, bins)
    np.save(d / "bins10ms_types.npy", units["type"].map(TYPE_CODES).to_numpy(np.int8))
    return out


class SessionData:
    """One session's raster (memory-mapped), labels and metadata."""

    def __init__(self, session: str):
        d = PROCESSED / session
        self.session = session
        self.meta = json.loads((d / "meta.json").read_text())
        self.rat = self.meta["rat"]
        self.region = self.meta["regions"][0]
        self.bins = np.load(d / "bins10ms.npy", mmap_mode="r")      # (N, T) uint8
        self.types = np.load(d / "bins10ms_types.npy")              # (N,) 0=E 1=I
        lab = np.load(d / "labels_1s.npz")
        self.labels = lab["labels"].astype(np.int64)                # authors' code per second
        self.usable = lab["usable"]                                 # WAKE/nonREM/REM, not excluded
        self.n_units = self.bins.shape[0]
        n_bins = self.bins.shape[1]
        # seconds whose 226-bin window fits inside the recording
        sec = np.arange(self.labels.size)
        self.window_ok = (sec * BINS_PER_S - HALF_REACH >= 0) & (sec * BINS_PER_S + BINS_PER_S + HALF_REACH <= n_bins)
        # class index (0,1,2) for usable seconds, -1 elsewhere
        self.y = np.full(self.labels.size, -1, dtype=np.int64)
        for name, code in MODEL_CODES.items():
            self.y[(self.labels == code) & self.usable] = CLASS_NAMES.index(name)
        self.labeled_seconds = np.where((self.y >= 0) & self.window_ok)[0]
        self.all_seconds = np.where(self.window_ok)[0]
        # each unit's mean spike count per bin over the whole recording (label-free);
        # used only when the input mode is "relative" (counts divided by the unit's own mean)
        counts = pd.read_csv(d / "units.csv")["n_spikes"].to_numpy(float)
        self.unit_mean_per_bin = counts / n_bins
        self.input_mode = "counts"

    def window(self, seconds: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """Raster windows for the given seconds: (B, N, 226) float32 spike counts
        (or counts relative to each unit's session mean when input_mode == "relative")."""
        starts = seconds * BINS_PER_S - HALF_REACH
        idx = starts[:, None] + np.arange(WINDOW)[None, :]          # (B, 226)
        src = self.bins if rows is None else self.bins[rows]
        x = np.asarray(src[:, idx]).transpose(1, 0, 2).astype(np.float32)   # (B, N, 226)
        if self.input_mode == "relative":
            m = self.unit_mean_per_bin if rows is None else self.unit_mean_per_bin[rows]
            x = (x / np.maximum(m, 1e-9)[None, :, None]).astype(np.float32)
        return x

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y[self.labeled_seconds], minlength=3)


def load_all(session_names: list[str] | None = None) -> dict[str, SessionData]:
    return {s: SessionData(s) for s in (session_names or sessions())}


def rats_of(data: dict[str, SessionData]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s, sd in data.items():
        out.setdefault(sd.rat, []).append(s)
    return out


if __name__ == "__main__":
    import sys, time
    names = sys.argv[1:] or sessions()
    for s in names:
        t0 = time.time()
        p = build_bins(s)
        b = np.load(p, mmap_mode="r")
        print(f"{s:18s} {b.shape[0]:4d} units x {b.shape[1]:9d} bins  {b.nbytes/1e6:7.1f} MB  {time.time()-t0:5.1f} s")
