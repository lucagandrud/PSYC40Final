"""Load one CRCNS fcx-1 session exactly as the authors provide it.

Nothing here infers or alters data. The one derived field is each unit's region,
which is a lookup: unit shank -> that shank's channels (.xml) -> channel region
(ChannelAnatomy.csv). Only plain-array formats (CellFormat, TimePairFormat) are used.

Loading traps handled:
- scipy's squeeze_me collapses a 1-element cell array into its contents, so cell
  arrays of spike trains are always read with squeeze_me=False.
- a single interval comes back as shape (2,), so intervals are reshaped to (n, 2).
- single-element index lists come back as scalars, so they are wrapped with atleast_1d.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from .reference import rat_of

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# our state name -> authors' variable prefix in _WSRestrictedIntervals.mat
STATE_VARS = {"WAKE": "Wake", "nonREM": "SWSPacket", "REM": "REM",
              "MA": "MA", "WI": "WakeInterruption"}
EPISODE_VARS = {"SLEEP_episode": "Sleep", "WakeSleep": "WakeSleep",
                "nonREM_episode": "SWSEpisode", "REM_episode": "REMEpisode"}


def _mat(path: Path, squeeze: bool = True) -> dict:
    return sio.loadmat(path, squeeze_me=squeeze, struct_as_record=False)


def _idx(x) -> np.ndarray:
    """MATLAB 1-based index list -> int array (handles scalar / empty)."""
    a = np.atleast_1d(np.asarray(x)).ravel()
    return a.astype(int) if a.size else np.zeros(0, int)


def _pairs(x) -> np.ndarray:
    """TimePairFormat -> float (n, 2) array of [start, stop] in seconds.
    Some sessions store WakeSleepTimePairFormat as a cell of 2x2 [wake; sleep] blocks;
    those blocks are stacked in order (rows alternate wake, sleep)."""
    raw = np.asarray(x)
    if raw.dtype == object:
        blocks = [np.asarray(e, dtype=float).reshape(-1, 2) for e in raw.ravel() if np.size(e)]
        return np.vstack(blocks) if blocks else np.zeros((0, 2))
    a = raw.astype(float)
    return np.zeros((0, 2)) if a.size == 0 else a.reshape(-1, 2)


def _trains(cell) -> list[np.ndarray]:
    return [np.asarray(c, dtype=float).ravel() for c in np.asarray(cell, dtype=object).ravel()]


def load_session(session: str, raw_dir: Path = RAW) -> dict:
    d = raw_dir / session
    p = lambda suffix: d / f"{session}{suffix}"

    # --- spikes: the authors' stable-unit set (SStable), as given ---------------
    st = _mat(p("_SStable.mat"), squeeze=False)
    spikes = _trains(st["S_CellFormat"])
    shank = np.asarray(st["shank"]).ravel().astype(int)
    cell_ix = np.asarray(st["cellIx"]).ravel().astype(int)

    # --- cell types (authors' putative E / I) ----------------------------------
    ids = _mat(p("_CellIDs.mat"))["CellIDs"]
    cellids = {f: _idx(getattr(ids, f)) for f in ["EAll", "IAll", "EDefinite", "IDefinite", "ELike", "ILike"]}

    # cross-check sources (counts and train lengths), used by QC
    ss = _mat(p("_SSubtypes.mat"), squeeze=False)
    se = _trains(ss["Se_CellFormat"]) if "Se_CellFormat" in ss else []
    si = _trains(ss["Si_CellFormat"]) if "Si_CellFormat" in ss else []
    mw = _mat(p("_MeanWaveforms.mat"))["MeanWaveforms"]
    mw_shanks = _idx(mw.shanks)

    # --- anatomy: shank -> channels (.xml spike groups) -> region (csv) ---------
    root = ET.parse(p(".xml")).getroot()
    groups = root.findall(".//spikeDetection/channelGroups/group")
    shank_channels = {i + 1: [int(c.text) + 1 for c in g.findall("channels/channel")]  # 1-based
                      for i, g in enumerate(groups)}
    anat = pd.read_csv(p("_ChannelAnatomy.csv"), header=None, names=["channel", "region"])
    anat["region"] = anat["region"].fillna("").astype(str).str.strip()
    ch_region = dict(zip(anat.channel.astype(int), anat.region))

    # --- states (authors' scoring, 1 s resolution) -----------------------------
    ws = _mat(p("_WSRestrictedIntervals.mat"))
    states = {k: _pairs(ws.get(v + "TimePairFormat", [])) for k, v in STATE_VARS.items()}
    episodes = {k: _pairs(ws.get(v + "TimePairFormat", [])) for k, v in EPISODE_VARS.items()}

    # --- events -----------------------------------------------------------------
    events = {}
    f_ud = p("_UPDOWNINtervals.mat")
    if f_ud.exists():
        ud = _mat(f_ud)
        events["UP"] = _pairs(ud.get("UPIntsTimePairFormat", []))
        events["DOWN"] = _pairs(ud.get("DNIntsTimePairFormat", []))
    f_sp = p("_Spindles.mat")
    if f_sp.exists():
        sp = _mat(f_sp)["SpindleData"]
        ns = np.asarray(sp.normspindles, dtype=float)
        events["spindles"] = ns.reshape(-1, 4) if ns.size else np.zeros((0, 4))  # start, peak, stop, power

    # --- metadata -----------------------------------------------------------------
    bmd = _mat(p("_BasicMetaData.mat"))["bmd"]
    meta = dict(
        session=session, rat=rat_of(session),
        recording_file_intervals=_pairs(bmd.RecordingFileIntervals).tolist(),
        goodshanks=_idx(bmd.goodshanks).tolist(),
        sampling_rate_hz=float(root.findtext(".//acquisitionSystem/samplingRate") or "nan"),
    )
    f_gs = p("_GoodSleepInterval.mat")
    if f_gs.exists():
        meta["good_sleep_interval"] = _pairs(_mat(f_gs)["GoodSleepInterval"].timePairFormat).tolist()
    f_cn = p("_ClusteringNotes.csv")
    meta["clustering_notes_rows"] = int(len(pd.read_csv(f_cn))) if f_cn.exists() else None

    return dict(session=session, spikes=spikes, shank=shank, cell_ix=cell_ix, cellids=cellids,
                ssubtypes=dict(Se=se, Si=si), mw_shanks=mw_shanks,
                shank_channels=shank_channels, ch_region=ch_region,
                states=states, episodes=episodes, events=events, meta=meta)


def authors_table() -> pd.DataFrame:
    """The authors' per-session spreadsheet (WatsonSleepHomestasis2016Table.xlsx)."""
    x = pd.read_excel(ROOT / "data" / "manifests" / "WatsonSleepHomestasis2016Table.xlsx",
                      sheet_name=0, header=None)
    x.columns = [str(c) for c in x.iloc[0]]
    x = x.iloc[2:]
    x = x[x["Session Name"].astype(str).str.contains("_")]
    return x.set_index("Session Name")
