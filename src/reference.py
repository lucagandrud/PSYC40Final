"""Reference values taken from the authors (Watson et al. 2016 and the CRCNS fcx-1 files).

Everything here is copied from a published source; each entry names where it came from.
"""
from __future__ import annotations

# Supplementary Table S1 (Watson et al. 2016): rat -> anatomical location, number of
# sessions, mean putative-excitatory (pE) and putative-inhibitory (pI) units per session.
TABLE_S1 = {
    "BWRat17":   dict(location="ACC",      n_sessions=2, pE=38.5, pI=4.0),
    "BWRat18":   dict(location="M2",       n_sessions=1, pE=34.0, pI=6.0),
    "BWRat19":   dict(location="M2",       n_sessions=2, pE=44.0, pI=11.5),
    "BWRat20":   dict(location="ACC",      n_sessions=2, pE=38.5, pI=5.5),
    "BWRat21":   dict(location="mPFC",     n_sessions=3, pE=19.7, pI=2.0),
    "Dino":      dict(location="mPFC,ACC", n_sessions=8, pE=41.8, pI=3.9),
    "B22":       dict(location="OFC",      n_sessions=3, pE=34.0, pI=7.3),
    "Bogey":     dict(location="mPFC",     n_sessions=1, pE=33.0, pI=3.0),
    "Splinter":  dict(location="OFC",      n_sessions=2, pE=33.5, pI=5.5),
    "Rizzo":     dict(location="OFC",      n_sessions=2, pE=57.0, pI=2.5),
    "Templeton": dict(location="OFC",      n_sessions=1, pE=10.0, pI=0.0),
}

# Watson et al. 2016, Results: median firing rates (Hz) per state.
PAPER_MEDIAN_HZ = {
    "E": {"WAKE": 0.76, "nonREM": 0.69, "REM": 0.88},
    "I": {"WAKE": 5.59, "nonREM": 4.69, "REM": 4.25},
}

# Time-window exclusions derived ONLY from the authors' notes in
# WatsonSleepHomestasis2016Table.xlsx ("Other Notes" / "Wake Activity" columns).
#   exclude_from_s: drop every second from this time on (note gives a timestamp)
#   injection_between_files: (a, b) -> drop every second after recording file a ends
NOTE_WINDOWS = {
    "BWRat18_020513": dict(exclude_from_s=7650.0,
                           why="authors: 'Big Instabilities at 7650s(?) & 9700s(shocks) in ctx units'"),
    "BWRat21_121113": dict(exclude_from_s=22000.0,
                           why="authors: 'Used only until UNSTABLE ~22000sec'"),
    "BWRat20_101513": dict(injection_between_files=(3, 4), why="authors: 'ketamine btw 03&04'"),
    "BWRat21_121613": dict(injection_between_files=(6, 7), why="authors: 'ketamine btw 06&07'"),
    "BWRat21_121813": dict(injection_between_files=(5, 6), why="authors: 'saline btw 05&06'"),
}

# ChannelAnatomy.csv label -> the name the paper uses for the same region.
# Table S1 lists BWRat18/BWRat19 as "M2"; their anatomy files say "MotorCtx"; the paper's
# methods list recording sites as "premotor cortex/M2". The file label is kept in the unit table.
REGION_ALIASES = {"MotorCtx": "M2"}

# Plot colors (validated categorical palette; status colors reserved for QC status).
STATE_COLORS = {"WAKE": "#2a78d6", "nonREM": "#eb6834", "REM": "#1baf7a",
                "MA": "#eda100", "WI": "#e87ba4"}
TYPE_COLORS = {"E": "#008300", "I": "#6250d6"}
STATUS_COLORS = {"PASS": "#0ca30c", "WARN": "#fab219", "FAIL": "#d03b3b"}


def rat_of(session: str) -> str:
    """Rat ID as used in Table S1. The 2014052x sessions belong to rat 'B22'
    (authors' spreadsheet: JennBuzsaki22)."""
    return "B22" if session.startswith("2014") else session.split("_")[0]
