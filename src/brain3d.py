"""3D model of the recorded neurons in the BrainGlobe rat atlas (whs_sd_rat_39um).

Region mapping (dataset label -> atlas structures), chosen as the closest named areas:
  mPFC -> PrL + IL   (prelimbic and infralimbic areas)
  ACC  -> Cg1 + Cg2  (cingulate areas 1 and 2)
  M2   -> M2         (secondary motor area)
  OFC  -> VO + LO + MO (ventral, lateral and medial orbital areas)
Units are placed inside their region's volume, grouped by shank: shanks are spread along
the anterior-posterior axis of the region, and each unit is jittered within 150 um of its
shank's anchor. Within-region positions are schematic; the dataset gives shank and region,
not coordinates.

Showcase: the Dino_061914 ACC + mPFC pair (same rat, same day, shared clock confirmed in
QC), across the WAKE -> SLEEP switch, with a timeline of the authors' state and the
decoder's prediction. Outputs: results/brain3d/brain3d.html (rotatable) and brain3d.mp4.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import ROOT, BINS_PER_S
from .reference import TYPE_COLORS, STATE_COLORS

OUT = ROOT / "results" / "brain3d"
ATLAS = "whs_sd_rat_39um"
REGION_MAP = {"mPFC": ["PrL", "IL"], "ACC": ["Cg1", "Cg2"], "M2": ["M2"], "OFC": ["VO", "LO", "MO"]}
JITTER_UM = 150


def sample_region_points(n_per_region: int = 4000, seed: int = 0) -> Path:
    """Voxel centres (um, atlas coordinates) inside each mapped region, right hemisphere, cached."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "region_points.npz"
    if out.exists():
        return out
    from brainglobe_atlasapi import BrainGlobeAtlas
    a = BrainGlobeAtlas(ATLAS)
    ann = a.annotation                                   # (ap, si, rl) voxel ids
    res = np.array(a.resolution)
    mid_rl = ann.shape[2] / 2
    rng = np.random.default_rng(seed)
    pts, meshes = {}, {}
    for region, acronyms in REGION_MAP.items():
        ids = [a.structures[ac]["id"] for ac in acronyms]
        vox = np.argwhere(np.isin(ann, ids))
        vox = vox[vox[:, 2] > mid_rl]                      # right hemisphere only
        take = vox[rng.choice(len(vox), size=min(n_per_region, len(vox)), replace=False)]
        pts[region] = take * res
        for ac in acronyms:
            m = a.mesh_from_structure(ac)
            v = np.asarray(m.points); f = np.asarray(m.cells[0].data)
            keep = v[:, 2] > mid_rl * res[2]
            meshes[ac] = (v, f, keep)
    np.savez_compressed(out, **{f"pts_{k}": v for k, v in pts.items()},
                        **{f"mesh_{k}_v": v for k, (v, f, kp) in meshes.items()},
                        **{f"mesh_{k}_f": f for k, (v, f, kp) in meshes.items()})
    return out


def place_units(session: str, region_pts: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Schematic coordinates for one session's units: shanks along AP, units jittered."""
    u = pd.read_csv(ROOT / "data" / "processed" / session / "units.csv")
    rng = np.random.default_rng(seed + hash(session) % 1000)
    shanks = sorted(u.shank.unique())
    order = np.argsort(region_pts[:, 0])                   # along AP
    q = np.linspace(0.2, 0.8, len(shanks))
    anchors = {sh: region_pts[order[int(qq * (len(order) - 1))]] for sh, qq in zip(shanks, q)}
    xyz = np.array([anchors[sh] + rng.uniform(-JITTER_UM, JITTER_UM, 3) for sh in u.shank])
    u[["ap", "si", "rl"]] = xyz
    return u


def spike_frames(session: str, t0: float, t1: float, step_s: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(frames, units) spike counts per frame of `step_s` between t0 and t1."""
    bins = np.load(ROOT / "data" / "processed" / session / "bins10ms.npy", mmap_mode="r")
    b0, b1 = int(t0 * BINS_PER_S), int(t1 * BINS_PER_S)
    per = int(round(step_s * BINS_PER_S))
    n = (b1 - b0) // per
    x = np.asarray(bins[:, b0:b0 + n * per]).reshape(bins.shape[0], n, per).sum(2).T
    return np.arange(n) * step_s + t0, x


def state_strip(session: str, t0: float, t1: float, pred_exp: str = "decoder") -> dict:
    """Authors' label and decoder prediction per second in [t0, t1)."""
    lab = np.load(ROOT / "data" / "processed" / session / "labels_1s.npz")["labels"]
    secs = np.arange(int(t0), int(t1))
    out = {"seconds": secs, "true": lab[secs]}
    meta = json.loads((ROOT / "data" / "processed" / session / "meta.json").read_text())
    pf = ROOT / "results" / pred_exp / meta["rat"] / f"preds_{session}.npz"
    if pf.exists():
        p = np.load(pf); idx = {s: i for i, s in enumerate(p["seconds"])}
        pred = np.array([p["probs"][idx[s]].argmax() + 1 if s in idx else 0 for s in secs])   # 1=WAKE 2=nonREM 3=REM
        out["pred"] = pred
    return out
