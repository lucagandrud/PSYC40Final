"""Phase 1a inventory: describe every variable/column in every file of a session.

Usage: python -I scripts/inventory.py data/raw/<session> [data/raw/<session> ...]
Appends/replaces rows for those sessions in data/manifests/inventory.csv and
prints a compact per-file summary.
"""
from __future__ import annotations

import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests" / "inventory.csv"
warnings.filterwarnings("ignore")


def mat_version(path: Path) -> str:
    head = path.open("rb").read(128)
    if head.startswith(b"MATLAB 7.3") or b"\x89HDF" in path.open("rb").read(1024):
        return "v7.3 (HDF5)"
    if head.startswith(b"MATLAB 5.0"):
        return "v5"
    return "unknown"


def sample(x) -> str:
    try:
        if isinstance(x, np.ndarray):
            if x.dtype == object:
                return f"cell[0]: {describe_type(x.flat[0])}" if x.size else "empty cell"
            flat = x.ravel()
            return np.array2string(flat[:4], precision=3, separator=",")[:80]
        if isinstance(x, (str, bytes)):
            return repr(x)[:80]
        if np.isscalar(x):
            return repr(x)[:80]
    except Exception as e:  # pragma: no cover
        return f"<{e}>"
    return ""


def describe_type(x) -> str:
    if isinstance(x, np.ndarray):
        return f"ndarray{x.shape} {x.dtype}"
    return type(x).__name__


def walk(name: str, x, rows: list, depth: int = 0):
    """Record one variable; recurse into structs (and object-array of structs)."""
    t = type(x).__name__
    if isinstance(x, sio.matlab.mat_struct):
        fields = list(x._fieldnames)
        rows.append(dict(variable=name, python_type="struct", shape="",
                         dtype="", sample="fields: " + ", ".join(fields)[:200]))
        if depth < 3:
            for f in fields:
                walk(f"{name}.{f}", getattr(x, f), rows, depth + 1)
        return
    if isinstance(x, sio.matlab.MatlabOpaque) or "Opaque" in t or "MatlabObject" in t:
        rows.append(dict(variable=name, python_type="MATLAB object (unreadable)",
                         shape="", dtype="", sample=""))
        return
    if isinstance(x, np.ndarray):
        kind = "cell array" if x.dtype == object else "ndarray"
        rows.append(dict(variable=name, python_type=kind, shape=str(x.shape),
                         dtype=str(x.dtype), sample=sample(x)))
        if x.dtype == object and x.size and depth < 3:
            first = x.flat[0]
            if isinstance(first, sio.matlab.mat_struct):
                walk(f"{name}[0]", first, rows, depth + 1)
            elif isinstance(first, np.ndarray):
                lens = [np.size(e) for e in x.flat]
                rows.append(dict(variable=f"{name}[*]", python_type="cell elements",
                                 shape=f"n={x.size}, len min/med/max={min(lens)}/{int(np.median(lens))}/{max(lens)}",
                                 dtype=str(first.dtype), sample=sample(first)))
        return
    if isinstance(x, dict):  # mat73 output
        rows.append(dict(variable=name, python_type="struct(dict)", shape="", dtype="",
                         sample="fields: " + ", ".join(x.keys())[:200]))
        if depth < 3:
            for k, v in x.items():
                walk(f"{name}.{k}", v, rows, depth + 1)
        return
    if isinstance(x, list):
        rows.append(dict(variable=name, python_type="list", shape=f"len={len(x)}", dtype="",
                         sample=describe_type(x[0]) if x else ""))
        return
    rows.append(dict(variable=name, python_type=t, shape="", dtype="", sample=sample(x)))


def inventory_mat(path: Path) -> tuple[str, list]:
    ver = mat_version(path)
    rows: list = []
    if ver == "v5":
        try:
            d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            return ver, [dict(variable="<load error>", python_type=str(e)[:150], shape="", dtype="", sample="")]
        for k, v in d.items():
            if not k.startswith("__"):
                walk(k, v, rows)
    else:
        try:
            import mat73
            d = mat73.loadmat(str(path))
            for k, v in d.items():
                walk(k, v, rows)
        except Exception as e:
            import h5py
            with h5py.File(path, "r") as f:
                def visit(n, o):
                    if isinstance(o, h5py.Dataset):
                        rows.append(dict(variable=n, python_type="h5 dataset", shape=str(o.shape),
                                         dtype=str(o.dtype), sample=""))
                f.visititems(visit)
            rows.insert(0, dict(variable="<mat73 failed>", python_type=str(e)[:150], shape="", dtype="", sample=""))
    return ver, rows


def inventory_csv(path: Path) -> list:
    df = pd.read_csv(path, header=None)
    rows = [dict(variable="<table>", python_type="csv", shape=str(df.shape), dtype="",
                 sample=" | ".join(map(str, df.iloc[0].tolist()))[:150])]
    for i, r in df.head(6).iterrows():
        rows.append(dict(variable=f"row{i}", python_type="csv row", shape="", dtype="",
                         sample=" | ".join(map(str, r.tolist()))[:150]))
    return rows


def inventory_xml(path: Path) -> list:
    root = ET.parse(path).getroot()
    rows = []
    nch = root.findtext(".//acquisitionSystem/nChannels")
    sr = root.findtext(".//acquisitionSystem/samplingRate")
    rows.append(dict(variable="acquisitionSystem", python_type="xml", shape="",
                     dtype="", sample=f"nChannels={nch}, samplingRate={sr}"))
    groups = root.findall(".//anatomicalDescription/channelGroups/group")
    rows.append(dict(variable="anatomicalDescription.channelGroups", python_type="xml",
                     shape=f"n_groups={len(groups)}", dtype="",
                     sample="; ".join(",".join(c.text for c in g.findall("channel")) for g in groups[:3])[:150]))
    sgroups = root.findall(".//spikeDetection/channelGroups/group")
    rows.append(dict(variable="spikeDetection.channelGroups", python_type="xml",
                     shape=f"n_groups={len(sgroups)}", dtype="",
                     sample="; ".join(",".join(c.text for c in g.findall("channels/channel")) for g in sgroups[:3])[:150]))
    return rows


def run(session_dir: Path) -> pd.DataFrame:
    session = session_dir.name
    out = []
    for p in sorted(session_dir.iterdir()):
        if p.name.startswith("."):
            continue
        suffix = p.name[len(session):] if p.name.startswith(session) else p.name
        if p.suffix == ".mat":
            ver, rows = inventory_mat(p)
        elif p.suffix == ".csv":
            ver, rows = "csv", inventory_csv(p)
        elif p.suffix == ".xml":
            ver, rows = "xml", inventory_xml(p)
        else:
            ver, rows = p.suffix, [dict(variable="<not inspected>", python_type="", shape="", dtype="", sample="")]
        for r in rows:
            out.append(dict(session=session, file=suffix, format=ver, bytes=p.stat().st_size, **r))
    return pd.DataFrame(out)


if __name__ == "__main__":
    frames = [run(Path(a).resolve()) for a in sys.argv[1:]]
    new = pd.concat(frames, ignore_index=True)
    if OUT.exists():
        old = pd.read_csv(OUT)
        old = old[~old.session.isin(new.session.unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(OUT, index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 500)
    for s in frames:
        print(s[["file", "format", "variable", "python_type", "shape", "dtype", "sample"]].to_string(index=False))
    print(f"\nwrote {OUT} ({len(new)} rows)")
