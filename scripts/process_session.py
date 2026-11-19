"""Phase 1b/1c/1d for one session: load -> preprocess -> QC checks -> figure -> save.

Usage: python -I scripts/process_session.py <session>
Writes data/processed/<session>/, figures/qc/<session>.png, and replaces this session's
rows in reports/qc_checks.csv (one row per check) and reports/qc_table.csv (one row per session).
Checks never drop units or time; exclusions come only from the authors' notes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.load import load_session, authors_table
from src.preprocess import preprocess, save, LABEL_CODES, MODEL_STATES
from src.reference import TABLE_S1, PAPER_MEDIAN_HZ, STATE_COLORS, TYPE_COLORS, rat_of

REPORTS = ROOT / "reports"
FIGS = ROOT / "figures" / "qc"
STATE_ORDER = ["WAKE", "nonREM", "REM", "MA", "WI"]
STATE_LABEL = {"WAKE": "WAKE", "nonREM": "nonREM", "REM": "REM", "MA": "Microarousal", "WI": "Wake interruption"}


def check(rows, area, name, status, detail):
    rows.append(dict(area=area, check=name, status=status, detail=detail))


def run_checks(S: str, s: dict, pp: dict) -> tuple[list[dict], dict]:
    rows, info = [], {}
    units = pp["units"]
    n = len(s["spikes"])
    E, I = s["cellids"]["EAll"], s["cellids"]["IAll"]
    dur = pp["duration_s"]
    lab, excl, usable = pp["labels"], pp["excluded"], pp["usable"]

    # ---- files ---------------------------------------------------------------
    mf = ROOT / "data" / "manifests" / "missing_files.csv"
    miss = pd.read_csv(mf) if mf.exists() else pd.DataFrame(columns=["session"])
    miss = miss[miss.session == S]
    check(rows, "Files", "files match authors' file list",
          "PASS" if miss.empty else "FAIL",
          "all files present, sizes match" if miss.empty else f"{len(miss)} missing/wrong size: " + "; ".join(miss.file))

    # ---- units ---------------------------------------------------------------
    se, si = s["ssubtypes"]["Se"], s["ssubtypes"]["Si"]
    lens = np.array([t.size for t in s["spikes"]])
    problems = []
    if len(E) + len(I) != n: problems.append(f"E+I={len(E)+len(I)} but SStable has {n}")
    if set(E) & set(I): problems.append(f"{len(set(E)&set(I))} units both E and I")
    if any((x < 1 or x > n) for x in np.r_[E, I]): problems.append("CellIDs index out of range")
    if len(se) != len(E) or len(si) != len(I): problems.append(f"SSubtypes Se/Si={len(se)}/{len(si)} vs CellIDs {len(E)}/{len(I)}")
    elif sorted(t.size for t in se) != sorted(lens[E - 1]) or sorted(t.size for t in si) != sorted(lens[I - 1]):
        problems.append("SSubtypes spike counts differ from SStable")
    if len(s["mw_shanks"]) == n and not np.array_equal(s["mw_shanks"], s["shank"]):
        problems.append("MeanWaveforms shanks differ from SStable shanks")
    check(rows, "Units", "E/I labels consistent across authors' files",
          "FAIL" if problems else "PASS",
          "; ".join(problems) if problems else f"{n} stable units = {len(E)} E + {len(I)} I; SSubtypes and MeanWaveforms agree")
    check(rows, "Units", "inhibitory units >= 3",
          "PASS" if len(I) >= 3 else "WARN",
          f"{len(I)} inhibitory units" + ("" if len(I) >= 3 else " (skip this session in the E-vs-I ablation)"))
    info.update(n_units=n, n_E=len(E), n_I=len(I),
                n_E_definite=int(np.isin(E, s["cellids"]["EDefinite"]).sum()),
                n_I_definite=int(np.isin(I, s["cellids"]["IDefinite"]).sum()))

    # ---- spikes --------------------------------------------------------------
    rec_end = pp["rec_file_end_s"] if pp["rec_file_end_s"] is not None else dur
    short = rec_end - dur
    check(rows, "Spikes", "spike data covers the recording files",
          "PASS" if short <= 10 else "WARN",
          (f"last spike {dur:.1f} s, recording files end {rec_end:.1f} s"
           if pp["rec_file_end_s"] is not None else f"recording-file end not given (inf); last spike {dur:.1f} s")
          + ("" if short <= 10 else f" -> {short:.0f} s at the end have no spike data and are not used"))
    bad = dict(unsorted=0, negative=0, beyond_end=0, duplicates=0, empty=0)
    for t in s["spikes"]:
        if t.size == 0: bad["empty"] += 1; continue
        d = np.diff(t)
        bad["unsorted"] += int((d < 0).any()); bad["duplicates"] += int((d == 0).any())
        bad["negative"] += int(t.min() < 0); bad["beyond_end"] += int(t.max() > rec_end + 1e-3)
    nbad = sum(bad.values())
    check(rows, "Spikes", "sorted, non-negative, within recording, no duplicates",
          "PASS" if nbad == 0 else "FAIL",
          f"all {n} units clean (recording end {rec_end:.1f} s)" if nbad == 0 else
          ", ".join(f"{k}: {v} units" for k, v in bad.items() if v))
    r1 = units.isi_lt_1ms_frac
    worst = units.loc[r1.idxmax()] if n else None
    check(rows, "Spikes", "refractory period (authors' sorting criterion: >1 ms)",
          "PASS" if (r1 <= 0.01).all() else "WARN",
          f"worst unit {int(worst.unit_id)}: {100*worst.isi_lt_1ms_frac:.2f}% ISIs <1 ms, "
          f"{100*worst.isi_lt_1p5ms_frac:.2f}% <1.5 ms; units with >1% ISIs <1 ms: {int((r1 > 0.01).sum())}"
          + " (reported only; nothing excluded)")
    silent = []
    for k in MODEL_STATES:
        if ((lab == LABEL_CODES[k]) & ~excl).sum():
            z = units.loc[units[f"rate_{k}_hz"] == 0, "unit_id"].tolist()
            if z: silent.append(f"{k}: units {z}")
    check(rows, "Spikes", "units with zero spikes in an analyzed state",
          "PASS" if not silent else "WARN",
          "none" if not silent else "; ".join(silent) + " (listed only; nothing excluded)")

    # ---- states --------------------------------------------------------------
    gap = pp["longest_gap"]
    check(rows, "Spikes", "longest stretch with no spikes from any unit",
          "PASS" if gap["seconds"] <= 10 else "WARN",
          f"{gap['seconds']:.2f} s starting at {gap['start_s']:.1f} s"
          + ("" if gap["seconds"] <= 10 else " (possible data gap; reported, not excluded)"))
    malformed = []
    for k, iv in s["states"].items():
        if len(iv) and ((iv[:, 1] <= iv[:, 0]).any() or iv.min() < 0):
            malformed.append(k)
    check(rows, "States", "intervals well-formed (start < stop, non-negative)",
          "PASS" if not malformed else "FAIL",
          "all intervals well-formed" if not malformed else "bad intervals in: " + ", ".join(malformed))
    bey = pp["states_beyond"]
    check(rows, "States", "scored states lie within the spike recording",
          "PASS" if not bey else "WARN",
          f"all scored time is before the recording end ({dur:.0f} s)" if not bey else
          ", ".join(f"{k} {v:.0f} s" for k, v in bey.items())
          + f" scored after the spike recording ends at {dur:.0f} s -> not used")
    ov = pp["overlaps"]
    check(rows, "States", "state types do not overlap",
          "PASS" if not ov else "WARN",
          "no second has two state labels" if not ov else
          ", ".join(f"{k} {v} s" for k, v in ov.items()) + " -> marked AMBIGUOUS, excluded from modeling")
    mins = {k: round(float(((lab == LABEL_CODES[k]) & ~excl).sum()) / 60, 1) for k in STATE_ORDER}
    info.update({f"min_{k}": v for k, v in mins.items()})
    info["min_usable_WAKE"] = mins["WAKE"]
    info["min_usable_SLEEP"] = round(mins["nonREM"] + mins["REM"], 1)
    info["pct_seconds_scored"] = round(100 * float((lab != 0).mean()), 1)
    ok = mins["WAKE"] > 0 and (mins["nonREM"] + mins["REM"]) > 0
    check(rows, "States", "usable WAKE and SLEEP time exists",
          "PASS" if ok else "FAIL",
          f"usable WAKE {mins['WAKE']} min, nonREM {mins['nonREM']} min, REM {mins['REM']} min "
          f"({info['pct_seconds_scored']}% of seconds scored by authors)")
    wk, sl = s["states"]["WAKE"], s["episodes"]["SLEEP_episode"]
    w2s = int(sum(np.any(np.abs(sl[:, 0] - b) <= 1) for b in wk[:, 1])) if len(wk) and len(sl) else 0
    s2w = int(sum(np.any(np.abs(wk[:, 0] - b) <= 1) for b in sl[:, 1])) if len(wk) and len(sl) else 0
    info.update(n_WAKE_to_SLEEP=w2s, n_SLEEP_to_WAKE=s2w,
                n_MA=int(len(s["states"]["MA"])), n_WI=int(len(s["states"]["WI"])),
                n_WAKE_bouts=int(len(wk)), n_SLEEP_episodes=int(len(sl)),
                n_UP=int(len(s["events"].get("UP", []))), n_DOWN=int(len(s["events"].get("DOWN", []))),
                n_spindles=int(len(s["events"].get("spindles", []))))
    check(rows, "States", "transitions and events",
          "PASS" if w2s > 0 else "WARN",
          f"WAKE->SLEEP {w2s}, SLEEP->WAKE {s2w}, microarousal onsets {info['n_MA']}, "
          f"wake-interruption onsets {info['n_WI']} (authors' episode boundaries, +/-1 s)")

    # ---- physiology (comparison with the paper; nothing excluded) -------------
    det, warn = [], False
    for typ in ["E", "I"]:
        sub = units[units.type == typ]
        for k in MODEL_STATES:
            v = sub[f"rate_{k}_hz"].dropna()
            if len(v) == 0: continue
            med = float(v.median()); ref = PAPER_MEDIAN_HZ[typ][k]
            info[f"median_{typ}_{k}_hz"] = round(med, 3)
            ratio = med / ref if ref else np.nan
            if typ == "E" or len(sub) >= 3:
                warn |= bool(ratio > 3 or ratio < 1 / 3)
            det.append(f"{typ} {k} {med:.2f} Hz (paper {ref})")
    if len(I) >= 3:
        for k in MODEL_STATES:
            e, i = info.get(f"median_E_{k}_hz"), info.get(f"median_I_{k}_hz")
            if e is not None and i is not None and i <= e:
                warn = True; det.append(f"I median <= E median in {k}")
    check(rows, "Physiology", "median rates vs Watson 2016 (comparison only)",
          "WARN" if warn else "PASS", "; ".join(det) + ("" if len(I) >= 3 else " (I medians: <3 cells, not judged)"))

    # ---- anatomy ---------------------------------------------------------------
    regs = sorted(units.region.unique())
    unm = units[~units.region.isin([r for r in regs if not r.startswith(("UNMAPPED", "AMBIGUOUS"))])]
    check(rows, "Anatomy", "every unit maps to one region via its shank",
          "PASS" if unm.empty else "FAIL",
          f"regions: {', '.join(regs)}" if unm.empty else f"{len(unm)} units unmapped/ambiguous: {unm.unit_id.tolist()}")
    rat = rat_of(S)
    t1 = TABLE_S1.get(rat, {}).get("location", "")
    conflict = [r for r in regs if r not in t1.split(",")]
    check(rows, "Anatomy", "region agrees with paper Table S1",
          "PASS" if not conflict else "WARN",
          f"data: {', '.join(regs)}; Table S1 ({rat}): {t1}" + ("" if not conflict else
          " -> using the per-channel anatomy file (data), conflict noted"))
    info.update(rat=rat, regions=",".join(regs))

    # ---- authors' notes ---------------------------------------------------------
    try:
        at = authors_table()
        note_txt = " | ".join(f"{c}: {at.loc[S, c]}" for c in ["Wake Activity", "Other Notes"]
                              if S in at.index and str(at.loc[S, c]) != "nan")
    except Exception as e:  # pragma: no cover
        note_txt = f"(could not read authors' table: {e})"
    note = pp["note"]
    info["authors_note"] = note_txt
    if note.get("problem"):
        check(rows, "Flags", "authors' notes applied", "FAIL", note["rule"])
    elif note["applied"]:
        cut = note["cut_s"]
        n_spk = int(sum((t >= cut).sum() for t in s["spikes"]))
        n_lab = int((lab[int(np.floor(cut)):] != 0).sum())
        gsi = s["meta"].get("good_sleep_interval")
        gs_end = max(r[1] for r in gsi) if gsi else None
        verdict = (f"cut at {cut:.0f} s ({note['rule']}); after it: {n_spk} spikes, {n_lab} scored seconds; "
                   f"GoodSleepInterval ends {gs_end if gs_end is None else f'{gs_end:.0f} s'}; "
                   + (f"{note['seconds_excluded']} s excluded" if n_spk or n_lab else
                      "no data after the cut (nothing affected)"))
        info["note_verdict"] = verdict
        check(rows, "Flags", "authors' notes applied", "PASS", f"{note['why']} -> {verdict}")
    else:
        check(rows, "Flags", "authors' notes applied", "PASS",
              (note_txt or "no notes") + " (no timestamped exclusion in the notes)")

    info["cleared_for_modeling"] = not any(r["status"] == "FAIL" for r in rows)
    return rows, info


def make_figure(S: str, s: dict, pp: dict, info: dict) -> Path:
    units, lab, excl = pp["units"], pp["labels"], pp["excluded"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig = plt.figure(figsize=(13, 8.2), facecolor="#fcfcfb")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 0.8, 2.6], hspace=0.55, wspace=0.32)

    # (a) hypnogram over the full session
    ax = fig.add_subplot(gs[0, :]); ax.set_facecolor("#fcfcfb")
    hrs = 3600.0
    for i, k in enumerate(STATE_ORDER):
        segs = [(a / hrs, (b - a) / hrs) for a, b in s["states"][k]]
        ax.broken_barh(segs, (len(STATE_ORDER) - 1 - i - 0.38, 0.76), color=STATE_COLORS[k], lw=0)
    if excl.any():
        x0 = np.argmax(excl) / hrs
        ax.axvspan(x0, len(excl) / hrs, color="#8a8984", alpha=0.18, lw=0)
        ax.text(x0, len(STATE_ORDER) - 0.35, " excluded (authors' note)", fontsize=8, color="#52514e", va="bottom")
    amb = np.flatnonzero(lab == LABEL_CODES["AMBIGUOUS"])
    if amb.size:
        ax.plot(amb / hrs, np.full(amb.size, -0.75), "|", color="#0b0b0b", ms=5, mew=0.8)
        ax.text(amb[0] / hrs, -0.75, "  ambiguous", fontsize=7, va="center", color="#52514e")
    ax.set_yticks(range(len(STATE_ORDER)))
    ax.set_yticklabels([STATE_LABEL[k] for k in STATE_ORDER[::-1]])
    xmax = max([len(lab)] + [iv[:, 1].max() for iv in s["states"].values() if len(iv)]) / hrs
    if xmax > len(lab) / hrs + 1e-6:
        ax.axvspan(len(lab) / hrs, xmax, facecolor="none", edgecolor="#8a8984", hatch="///", lw=0)
        ax.text(len(lab) / hrs, len(STATE_ORDER) - 0.35, " no spike data",
                fontsize=8, color="#52514e", va="bottom")
    ax.set_xlim(0, xmax); ax.set_ylim(-1.1, len(STATE_ORDER) - 0.4)
    ax.set_title("Authors' state scoring over the whole session (blank = unscored)", loc="left", fontsize=10)
    ax.tick_params(axis="y", length=0)

    # (b) population rate per cell, 10 s bins
    ax2 = fig.add_subplot(gs[1, :], sharex=ax); ax2.set_facecolor("#fcfcfb")
    c = pp["counts"]; nb = c.shape[1] // 10
    for typ in ["E", "I"]:
        idx = units.index[units.type == typ].to_numpy()
        if idx.size and nb:
            r = c[idx, :nb * 10].reshape(idx.size, nb, 10).sum(2).mean(0) / 10
            ax2.plot((np.arange(nb) * 10 + 5) / hrs, r, color=TYPE_COLORS[typ], lw=1.2,
                     label=f"{'excitatory' if typ == 'E' else 'inhibitory'} (n={idx.size})")
    ax2.set_yscale("log"); ax2.set_ylabel("Hz per cell"); ax2.set_xlabel("time (h)")
    ax2.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax2.set_title("Mean firing rate per cell, 10 s bins", loc="left", fontsize=10)

    # (c) 120 s raster across a WAKE->SLEEP boundary
    sub = gs[2, :2].subgridspec(2, 1, height_ratios=[0.06, 1], hspace=0.04)
    axs, axr = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    wk, sl = s["states"]["WAKE"], s["episodes"]["SLEEP_episode"]
    b = None
    for x in wk[:, 1] if len(wk) else []:
        if len(sl) and np.any(np.abs(sl[:, 0] - x) <= 1) and not excl[min(int(x), len(excl) - 1)]:
            b = float(x); break
    if b is None:
        b = float(wk[0, 1]) if len(wk) else len(lab) / 2
    t0, t1 = b - 60, b + 60
    sec = np.arange(int(max(t0, 0)), int(min(t1, len(lab))))
    inv = {v: k for k, v in LABEL_CODES.items()}
    for k in sec:
        name = inv[int(lab[k])]
        axs.axvspan(k - b, k + 1 - b, color=STATE_COLORS.get(name, "#e6e5e0"), lw=0)
    axs.set_xlim(-60, 60); axs.axis("off")
    axs.set_title("Spikes across the authors' WAKE→SLEEP boundary (state bar on top)", loc="left", fontsize=10)
    order = units.assign(r=units.rate_WAKE_hz.fillna(0)).sort_values(["type", "r"], ascending=[True, False])
    for row_i, (_, u) in enumerate(order.iterrows()):
        t = s["spikes"][int(u.unit_id) - 1]
        t = t[(t >= t0) & (t < t1)] - b
        axr.vlines(t, row_i + 0.1, row_i + 0.9, color=TYPE_COLORS.get(u.type, "#8a8984"), lw=0.5)
    axr.axvline(0, color="#0b0b0b", lw=0.8, ls="--")
    axr.set_xlim(-60, 60); axr.set_ylim(len(order), 0)
    axr.set_xlabel("seconds from scored WAKE→SLEEP boundary"); axr.set_ylabel("unit (E then I, by WAKE rate)")
    nE = int((units.type == "E").sum())
    axr.text(61, nE / 2, "E", color=TYPE_COLORS["E"], va="center", fontsize=9, fontweight="bold")
    if (units.type == "I").any():
        axr.text(61, nE + (len(order) - nE) / 2, "I", color=TYPE_COLORS["I"], va="center", fontsize=9, fontweight="bold")
    axr.set_yticks([])

    # (d) per-state firing-rate distributions (log), E and I
    sub2 = gs[2, 2].subgridspec(2, 1, hspace=0.75)
    bins = np.logspace(-3, 2, 26)
    for j, typ in enumerate(["E", "I"]):
        a = fig.add_subplot(sub2[j]); a.set_facecolor("#fcfcfb")
        uu = units[units.type == typ]
        for k in MODEL_STATES:
            v = uu[f"rate_{k}_hz"].dropna()
            v = v[v > 0]
            if len(v):
                a.hist(v, bins=bins, histtype="step", lw=1.8, color=STATE_COLORS[k], label=k)
            ref = PAPER_MEDIAN_HZ[typ][k]
            a.axvline(ref, color=STATE_COLORS[k], lw=1, ls=":", alpha=0.9)
        a.set_xscale("log"); a.set_ylabel("units")
        a.set_title(f"{'Excitatory' if typ == 'E' else 'Inhibitory'} rates by state (n={len(uu)})\n"
                    f"dotted = paper medians", loc="left", fontsize=9)
        if j == 0: a.legend(frameon=False, fontsize=8)
        if j == 1: a.set_xlabel("firing rate (Hz)")

    status = "cleared for modeling" if info["cleared_for_modeling"] else "BLOCKED (see QC)"
    fig.suptitle(f"{S}   ·   rat {info['rat']}   ·   {info['regions']}   ·   "
                 f"{info['n_E']} E / {info['n_I']} I units   ·   {status}",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color="#0b0b0b")
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"{S}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def upsert(path: Path, df: pd.DataFrame, S: str):
    if path.exists():
        old = pd.read_csv(path)
        old = old[old.session != S]
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values("session").to_csv(path, index=False)


def main(S: str):
    REPORTS.mkdir(exist_ok=True)
    s = load_session(S)
    pp = preprocess(s)
    save(s, pp)
    rows, info = run_checks(S, s, pp)
    make_figure(S, s, pp, info)
    upsert(REPORTS / "qc_checks.csv", pd.DataFrame([dict(session=S, **r) for r in rows]), S)
    statuses = {f"{r['area']}: {r['check']}": r["status"] for r in rows}
    upsert(REPORTS / "qc_table.csv", pd.DataFrame([dict(session=S, **info, **statuses)]), S)
    nf = sum(r["status"] == "FAIL" for r in rows); nw = sum(r["status"] == "WARN" for r in rows)
    print(f"{S}: {len(rows)} checks, {nf} FAIL, {nw} WARN, cleared={info['cleared_for_modeling']}")
    for r in rows:
        if r["status"] != "PASS":
            print(f"   {r['status']} {r['area']}: {r['check']} -> {r['detail']}")


if __name__ == "__main__":
    for S in sys.argv[1:]:
        main(S)
