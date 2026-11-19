"""Build notebooks/01_data.ipynb (the data description section) and execute it.

Usage: python -I scripts/make_data_notebook.py
"""
from pathlib import Path
import subprocess
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""# 01: Data description, preprocessing and quality checks

This notebook is the data section of my PSYC 40 final project. It documents exactly what data I used, how I turned the authors' files into a model-ready format, and the checks I ran to make sure nothing was lost, misread or invented along the way.

**Dataset.** CRCNS fcx-1 (Watson, Levenstein, Greene, Gelinas & Buzsáki, 2016, *Neuron* 90:839; data doi:10.6080/K02N506Q). Rats carried 64-site silicon probes in deep layers of frontal cortex and were recorded in their home cage during the lights-on period, across natural waking and sleep.

**Guiding principle.** I use the authors' values and curation as given: their spike times, their stable-unit set, their excitatory/inhibitory labels and their sleep scoring. I never drop units or time on a metric of my own. Time is excluded only where the authors' own notes say so, or where there is no spike data."""),
code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from IPython.display import Image, display, Markdown
from src.preprocess import load_processed, LABEL_CODES
pd.set_option('display.max_colwidth', 120)
qc = pd.read_csv(ROOT/'reports/qc_table.csv').sort_values(['rat','session'])
checks = pd.read_csv(ROOT/'reports/qc_checks.csv')
print(f"{len(qc)} sessions, {qc.rat.nunique()} rats, {int(qc.n_E.sum())} excitatory + {int(qc.n_I.sum())} inhibitory units")"""),
md("""## 1. What the data is

**Spikes.** The authors recorded voltage at 20 kHz, detected spikes, and sorted them into single neurons ("units"). For each unit I get the list of times it fired, in seconds, precise to 0.05 ms (every time lies on the 1/20,000 s grid). This is event data, not a regularly sampled signal. A session is naturally pictured as a raster: x = time (hours), y = unit, one tick per spike.

**The authors' labels.**
- *Per unit:* putative excitatory (E) or inhibitory (I), from spike waveform shape and, where available, confirmed by short-latency synaptic interactions with other cells ("definite" vs "waveform only"); and the probe shank the unit was recorded on.
- *Per time:* brain state, scored at 1 s resolution from the LFP spectrum (first principal component), hippocampal/cortical theta and an EMG proxy derived from the LFP, then manually reviewed. States: **WAKE** (≥7 min arousal), **nonREM** packets, **REM**, **microarousals** (<40 s arousals between nonREM packets) and **wake interruptions** (40–120 s).
- *Events:* UP/DOWN states and spindles.

**Files I use** (all MATLAB v5, plain-array formats only): `_SStable` (spike times of the authors' stable units, plus shank), `_CellIDs` (E/I labels), `_SSubtypes` and `_MeanWaveforms` (cross-checks), `.xml` + `_ChannelAnatomy.csv` (shank → channels → region), `_WSRestrictedIntervals` (state scoring), `_UPDOWNINtervals`, `_Spindles`, `_BasicMetaData`. I skipped the raw LFP (`.eeg`), which the plan does not use."""),
md("## 2. Sessions I used"),
code("""cols = ['session','rat','regions','n_E','n_I','min_usable_WAKE','min_nonREM','min_REM','n_WAKE_to_SLEEP','n_MA','pct_seconds_scored','cleared_for_modeling']
qc[cols].rename(columns={'n_E':'E units','n_I':'I units','min_usable_WAKE':'WAKE min','min_nonREM':'nonREM min',
                         'min_REM':'REM min','n_WAKE_to_SLEEP':'WAKE→SLEEP','n_MA':'microarousals',
                         'pct_seconds_scored':'% seconds scored','cleared_for_modeling':'cleared'})"""),
code("display(Image(filename=str(ROOT/'figures/qc/_overview.png')))"),
md("""## 3. Preprocessing: what I did to make it model-ready

1. **Load as given.** Spike trains come from the authors' stable-unit set. Three loading traps are handled explicitly: (a) `scipy` collapses a one-element cell array into its contents (one session's `SiDef` loads as a single 196,195-spike train instead of a list of one train), so cell arrays are read unsqueezed; (b) a single interval loads as shape `(2,)` and is reshaped to `(1, 2)`; (c) one-element index lists load as scalars.
2. **Region by shank (a lookup, not an inference).** No file gives a per-unit region. Each unit's shank → that shank's channels (`.xml`) → each channel's region (`_ChannelAnatomy.csv`). Every shank with units lies entirely in one region, so the lookup is unambiguous. The anatomy files label BWRat18/19 as "MotorCtx"; the paper calls this region M2, and I use M2 (the file label is kept in the unit table).
3. **1 s labels (the answer key).** The authors' state intervals written out one entry per second. Seconds no interval covers stay *unscored* (neurons fire normally then; the authors simply did not assign a state outside their wake-to-sleep episodes). Seconds covered by two state types are marked *ambiguous* (no precedence rule) and are never used.
4. **End of data = last spike.** In several sessions the authors' scoring, and sometimes the recording-file metadata, extend past the last spike. Those seconds have no spike data and are never used.
5. **Exclusions from the authors' notes only.** BWRat18_020513: from 7650 s ("Big instabilities at 7650 s"); BWRat21_121113: from 22000 s ("Used only until UNSTABLE ~22000 sec"); injection sessions: everything after the recording file before the injection ends.
6. **Per-state firing rates** = spike count ÷ seconds the authors scored as that state (arithmetic only).

Output per session in `data/processed/<session>/`: `spikes.npz`, `units.csv`, `labels_1s.npz` (labels, excluded, usable), `states.npz`, `events.npz`, `meta.json`."""),
code("""s = load_processed('BWRat20_101013')
print('units:', len(s['spikes']), '| seconds:', len(s['labels']), '| usable seconds:', int(s['usable'].sum()))
print('label codes:', LABEL_CODES)
s['units'][['unit_id','type','type_source','shank','region','n_spikes','rate_WAKE_hz','rate_nonREM_hz','rate_REM_hz']].head(8)"""),
md("""## 4. Quality checks

Every check reports PASS / WARN / FAIL per session. A FAIL blocks a session from modeling; a WARN is reported and explained but changes nothing unless the authors' notes say so."""),
code("display(Image(filename=str(ROOT/'figures/qc/_checks_heatmap.png')))"),
code("""flags = checks[checks.status != 'PASS'][['session','status','area','check','detail']].sort_values('session')
flags.style.hide(axis='index')"""),
md("""**Unit counts against the paper.** Supplementary Table S1 gives each rat's mean E and I units per session. For every rat whose sessions are all here, my counts must match exactly."""),
code("""from src.reference import TABLE_S1
rows = []
for rat, g in qc.groupby('rat'):
    r = TABLE_S1[rat]
    rows.append(dict(rat=rat, sessions=len(g), sessions_in_paper=r['n_sessions'],
                     E_mine=round(g.n_E.mean(),1), E_paper=r['pE'], I_mine=round(g.n_I.mean(),1), I_paper=r['pI'],
                     verdict=('MATCH' if abs(g.n_E.mean()-r['pE'])<0.06 and abs(g.n_I.mean()-r['pI'])<0.06 else 'MISMATCH')
                             if len(g)==r['n_sessions'] else 'partial'))
pd.DataFrame(rows)"""),
md("""**Firing rates against the paper.** Watson 2016 reports median rates of excitatory cells of 0.76 / 0.69 / 0.88 Hz (WAKE / nonREM / REM) and of inhibitory cells of 5.59 / 4.69 / 4.25 Hz. Pooled over my sessions:"""),
code("""units = pd.concat([load_processed(x)['units'] for x in qc.session])
med = units.groupby('type')[['rate_WAKE_hz','rate_nonREM_hz','rate_REM_hz']].median().round(2)
med['n_units'] = units.groupby('type').size()
med"""),
md("## 5. Example sessions (one per Phase 1 session)"),
code("""for x in ['BWRat20_101013', '20140526_277um']:
    display(Image(filename=str(ROOT/f'figures/qc/{x}.png')))"""),
md("""## 6. What the data can and cannot support

- **Labels and inputs share signal.** The authors scored states from LFP recorded on the same probes as the spikes, and the main scoring signal (LFP PC1) tracks the same UP/DOWN alternation that appears as spike synchrony. High decoding accuracy is expected partly for this reason.
- **WAKE means active wake.** WAKE labels cover only arousals of ≥7 min, which the authors chose because they go with active behavior; microarousals and wake interruptions are excluded from the WAKE/SLEEP labels.
- **Lights-on recordings, deep layers, frontal cortex only.**
- **Region and rat are confounded.** Each rat contributes one region (Dino contributes ACC and mPFC as separate recordings). One conflict: Table S1 lists rat B22 as OFC while its anatomy files and the authors' spreadsheet say mPFC; I follow the anatomy files.
- **Same-rat sessions mostly record different neurons** (probes advanced up to 300 µm/day), which is one more reason to split training and testing by rat.

The full interactive report with every session's panel is `reports/qc_report.html`."""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = ROOT / "notebooks" / "01_data.ipynb"
nbf.write(nb, out)
r = subprocess.run([str(ROOT / ".venv/bin/jupyter"), "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=600", str(out)], capture_output=True, text=True, cwd=ROOT / "notebooks")
print(r.stdout[-500:], r.stderr[-1500:])
sys.exit(r.returncode)
