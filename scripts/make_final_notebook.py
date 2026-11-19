"""Build and execute notebooks/PSYC40Final.ipynb from the saved results, then export it to
reports/PSYC40Final.html and reports/PSYC40Final.pdf (headless Chrome).

The notebook reads only committed files (results/, figures/, reports/qc_table.csv), so it runs
on any laptop after cloning, without the raw data. Rerun this script after new results land.
"""
import json
import subprocess
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
NB = ROOT / "notebooks" / "PSYC40Final.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

# ------------------------------------------------------------------ numbers pulled from results
exp = pd.read_csv(RES / "summary_experiments.csv").set_index("experiment")
folds = pd.read_csv(RES / "summary_folds.csv")
qc = pd.read_csv(ROOT / "reports" / "qc_table.csv")
n_sessions, n_E, n_I = len(qc), int(qc.n_E.sum()), int(qc.n_I.sum())
hours = {k: qc[f"min_{k}"].sum() / 60 for k in ("WAKE", "nonREM", "REM")}
rats_per_region = qc.groupby("regions").rat.nunique().to_dict()


def stat(e, col="bal_acc_mean", fmt="{:.2f}"):
    try:
        return fmt.format(exp.loc[e, col])
    except Exception:
        return "–"


def n_rats(e):
    try:
        return int(exp.loc[e, "n_rats"])
    except Exception:
        return 0


dec, base = stat("decoder"), stat("baseline")
dec_sd, base_sd = stat("decoder", "bal_acc_sd"), stat("baseline", "bal_acc_sd")
dec_k, base_k = stat("decoder", "kappa_mean"), stat("baseline", "kappa_mean")
dec2, base2 = stat("decoder", "bal_acc_2class_mean"), stat("baseline", "bal_acc_2class_mean")
dec2_sd, base2_sd = stat("decoder", "bal_acc_2class_sd"), stat("baseline", "bal_acc_2class_sd")
dec2_k, base2_k = stat("decoder", "kappa_2class_mean"), stat("baseline", "kappa_2class_mean")
rec = {k: stat("decoder", f"recall_{k}") for k in ("WAKE", "nonREM", "REM")}
act, act_b = stat("decoder", "nonrem_vs_activated_mean"), stat("baseline", "nonrem_vs_activated_mean")
act_sd, act_b_sd = stat("decoder", "nonrem_vs_activated_sd"), stat("baseline", "nonrem_vs_activated_sd")


def wins(col):
    d = folds[folds.experiment == "decoder"].set_index("rat")[col]
    b = folds[folds.experiment == "baseline"].set_index("rat")[col]
    return f"{int((d > b.loc[d.index]).sum())} of {len(d)}"


ABL_NAMES = {"decoder_short": "all cells (reference)", "E_only": "excitatory cells only", "I_only": "inhibitory cells only",
             "E_matched": "excitatory cells, as many as inhibitory", "no_population": "without the population convolutions"}


def ablation_table():
    """Mean 3-class balanced accuracy over the ablation rats, for the reference and each ablation."""
    rats = set(folds[folds.experiment == "decoder_short"].rat)
    out, vals = [], {}
    for e, name in ABL_NAMES.items():
        d = folds[(folds.experiment == e) & folds.rat.isin(rats)]
        if d.empty:
            continue
        vals[e] = (d.balanced_accuracy.mean(), d.balanced_accuracy.std(), len(d))
        out.append(f"| {name} | {vals[e][0]:.2f} ± {vals[e][1]:.2f} | {vals[e][2]} |")
    return "| model | WAKE / nonREM / REM balanced accuracy (mean ± sd) | rats |\n|---|---|---|\n" + "\n".join(out), vals


abl_table, v = ablation_table()
n_collapsed = len(list(RES.glob("*/*_collapsed_seed*")))
n_abl_runs = sum(1 for e in ABL_NAMES for _ in RES.glob(f"{e}/*/metrics.json") if "_collapsed" not in str(_))
d_pop = v["no_population"][0] - v["decoder_short"][0]
pop_verdict = ("within the rat-to-rat spread, so five rats cannot tell whether the population dynamics over the second add information"
               if abs(d_pop) < max(v["no_population"][1], v["decoder_short"][1]) else
               ("so the population dynamics over the second carry part of the information" if d_pop < 0 else "so the population layers did not help"))
weak_rats = folds[(folds.experiment == "decoder") & (folds.recall_WAKE < 0.1)].rat.tolist()

# ------------------------------------------------------------------ notebook text
md("""
# Decoding sleep and wake from frontal-cortical spikes

Code and results are in [github.com/lucagandrud/PSYC40Final](https://github.com/lucagandrud/PSYC40Final). The results shown here are computed from files committed in that repo; the raw data (CRCNS fcx-1) are not needed to run this notebook.
""")

md(f"""
## 1. Data

**Source.** CRCNS fcx-1 (Watson, Levenstein, Greene, Gelinas & Buzsáki 2016, *Neuron* 90:839; doi:10.6080/K02N506Q):

* Silicon-probe recordings from the frontal cortex of rats across natural sleep and wake, during the lights-on period.
* Spike times of sorted single units (the authors' stable-unit set), each labeled by the authors as excitatory (E) or inhibitory (I), with the probe shank it was recorded on.
* The authors' sleep-wake state intervals: WAKE, nonREM and REM (scored from the LFP and an LFP-derived EMG proxy), plus microarousal and wake-interruption intervals.

**What I used.** {n_sessions} sessions from 10 rats (up to 3 per rat), from four frontal regions: medial prefrontal cortex (mPFC, {rats_per_region['mPFC']} rats), anterior cingulate cortex (ACC, {rats_per_region['ACC']}), secondary motor cortex (M2, {rats_per_region['M2']}) and orbitofrontal cortex (OFC, {rats_per_region['OFC']}). One rat, Dino, has an ACC and an mPFC probe recorded at the same time. In total {n_E} excitatory and {n_I} inhibitory units, with {hours['WAKE']:.1f} h of WAKE, {hours['nonREM']:.1f} h of nonREM and {hours['REM']:.1f} h of REM.

**Format.** Per session: spike times per unit (seconds), a unit table (type, shank, brain region) and the sleep-wake state intervals (start and stop in seconds). Below, one session: the authors' sleep-wake state timeline, mean firing rate over the session, 2 min of spiking across a WAKE→sleep boundary, and firing rates per state.
""")
code("""
from IPython.display import Image, HTML, display
display(Image("../figures/qc/BWRat20_101013.png", width=1000))
""")
md("""
**Preprocessing** (code in `src/`):

1. **Loaded as given.** Spike times, unit types and sleep-wake state intervals are the authors' values, unchanged.
2. **One label per second.** Every second of the recording gets a label, WAKE, nonREM or REM, if it lies inside one of the authors' sleep-wake state intervals. The authors left 47% of the recording unscored; those seconds, their microarousal and wake-interruption seconds, and the 0.7% of seconds where two intervals overlap are part of the input but never training targets.
3. **10 ms bins.** Each unit's spikes are counted in 10 ms bins over the whole recording.
4. **Relative counts.** Each unit's counts are divided by that unit's own mean count per bin over the recording (no labels involved). Firing rates differ several-fold between rats; with raw counts the network did not learn at all, with relative counts it did.
""")
code("""
import pandas as pd
qc = pd.read_csv("../reports/qc_table.csv")
cols = {"session": "session", "rat": "rat name", "regions": "brain region", "n_E": "excitatory units", "n_I": "inhibitory units",
        "min_WAKE": "WAKE (min)", "min_nonREM": "nonREM (min)", "min_REM": "REM (min)", "pct_seconds_scored": "% scored"}
t = qc[list(cols)].rename(columns=cols).round(0)
t[list(cols.values())[5:]] = t[list(cols.values())[5:]].astype(int)
display(HTML(f"<p style='break-after: avoid'><b>Sessions used</b>: metadata for the {len(t)} sessions the model was trained and tested on.</p>" + t.set_index("session").to_html()))
display(HTML("<p><i>Each session is one recording of a rat's frontal-cortical activity across sleep and wake. "
             "% scored is the share of 1 s time points the authors labeled WAKE, nonREM or REM. "
             f"All {len(t)} sessions were cleared by my QC, which checked that every file loads and matches the authors' file list, "
             "spike times are sorted and lie within the recording, sleep-wake state intervals are well-formed, "
             "and unit counts and brain regions match the paper.</i></p>"))
""")

md("""
## 2. Objective

To decode a rat's sleep-wake state (WAKE, nonREM or REM) from its frontal-cortical spiking: train the model on rats' recordings, then test it on a rat it has never seen. The trained model can then be used to ask which cortical neurons and activity patterns carry information about the sleep-wake state.

Sleep and wake are switched by hypothalamic and brainstem circuits (Saper, Scammell & Lu 2005, *Nature* 437:1257). Frontal cortex sits downstream of that switch, so what the decoder reads is the cortical signature of the state, and the ablations ask which parts of that signature carry the most information.
""")

md("""
## 3. Model

A 1D convolutional neural network that reads the neuronal spiking time series directly. The same small filters slide along every neuron's spike train, an attention layer compares neurons with each other at every 10 ms step, a learned pooling step merges them into one population signal, and three wider convolutions read how that signal evolves over the second before the readout. Every weight belongs to a filter or an attention layer that is shared by all neurons, so a network trained on nine rats runs on a tenth with a different number of cells.

I chose this over a feature-based classifier because the aim was to work with the activity itself rather than with a handful of numbers I would have picked, and over an LSTM or transformer because a recording of ten rats is too little data for models with many more weights. The logistic-regression baseline below shows what firing rates alone can do.
""")
code("""
display(Image("../figures/architecture.png", width=950))
""")
md(f"""
**Training and testing.** Leave-one-rat-out, ten folds: one test rat, one validation rat (for early stopping) and eight training rats, with a rat's sessions always kept together. Each epoch draws the same number of labeled seconds from every training rat and every state, so no rat and no state dominates. The loss is cross-entropy against the authors' label of each second. The network has 16 filters, 2 pooling queries and 6.5k weights.

A run that never left chance on its validation rat was restarted with a new random seed ({n_collapsed} of {n_abl_runs + n_collapsed} shortened ablation runs; none of the ten main folds). The failed runs are kept in `results/` under `_collapsed`.

**Baseline.** Logistic regression on the same 2.26 s window, with the spike trains first collapsed to two traces: spikes per excitatory cell and per inhibitory cell in each bin. In practice it can only read overall excitatory and inhibitory firing rates.
""")

md("""
## 4. Metric

* **Balanced accuracy** on the held-out rat: the mean over states of the fraction of each state's seconds that were called correctly. The large nonREM class (55% of labeled seconds) cannot inflate it. Chance is 0.33 for three states and 0.50 for two.
* **Cohen's kappa**: agreement with the authors' labels beyond chance, the standard figure in sleep scoring.
* Both are reported for the three states (WAKE / nonREM / REM) and for WAKE vs SLEEP (the same predictions with nonREM and REM merged into SLEEP), with a confusion matrix and results per rat.
* Every number below comes from rats the model never trained on.
""")

md(f"""
## 5. Results

### 5.1 Decoding held-out rats

| model | WAKE / nonREM / REM: balanced accuracy (mean ± sd over rats) | kappa | WAKE vs SLEEP: balanced accuracy | kappa | rats |
|---|---|---|---|---|---|
| CNN on the spike trains | **{dec} ± {dec_sd}** | {dec_k} | **{dec2} ± {dec2_sd}** | {dec2_k} | {n_rats("decoder")} |
| logistic regression on excitatory / inhibitory rates | {base} ± {base_sd} | {base_k} | {base2} ± {base2_sd} | {base2_k} | {n_rats("baseline")} |

*WAKE / nonREM / REM is the 3-class decoder: the model names the state of each second as one of the three. WAKE vs SLEEP uses the same predictions with nonREM and REM counted as SLEEP.*

The CNN beats the baseline on the three-state score in {wins("balanced_accuracy")} rats. Separating nonREM from the two activated states (WAKE and REM merged): CNN {act} ± {act_sd}, baseline {act_b} ± {act_b_sd}, better in {wins("bal_acc_nonrem_vs_activated")} rats. Per-state recall of the CNN, averaged over rats: WAKE {rec["WAKE"]}, nonREM {rec["nonREM"]}, REM {rec["REM"]}.
""")
code("""
display(Image("../figures/results_decoder.png", width=900))
display(Image("../figures/results_confusion.png", width=300))
folds = pd.read_csv("../results/summary_folds.csv")
d = folds[folds.experiment.isin(["decoder", "baseline"])]
show = d.pivot(index=["rat", "region"], columns="experiment", values=["balanced_accuracy", "bal_acc_2class"]).round(2)
show.columns = [f"{'WAKE / nonREM / REM' if a == 'balanced_accuracy' else 'WAKE vs SLEEP'}: {'CNN' if b == 'decoder' else 'logistic regression'}" for a, b in show.columns]
display(HTML("<p style='break-after: avoid'><b>Balanced accuracy per held-out rat</b></p>" + show.to_html()))
""")

md(f"""
### 5.2 Ablations: which cells carry the state information?

Each ablation retrains the same network from scratch with something removed, on five held-out rats (one per brain region); the reference row is the full network trained the same way.

* **Excitatory cells only / inhibitory cells only**: the model sees only the cells of one type.
* **Excitatory cells, as many as inhibitory**: random subsets of excitatory cells, as many as the session has inhibitory cells (50 draws). This is the control for the excitatory vs inhibitory comparison, because inhibitory cells are far fewer ({n_I} vs {n_E}).
* **Without the population convolutions**: the pooled population signal goes straight to the readout, so the model cannot read how the population evolves over the second.

{abl_table}
""")
code("""
display(Image("../figures/results_ablations.png", width=850))
display(HTML("<p><i>All ablation runs and the reference used the same shortened training budget. "
             "Sessions with fewer than 3 inhibitory cells are skipped in the cell-type rows.</i></p>"))
""")

md("""
### 5.3 Firing around state boundaries

The figure shows excitatory rate, inhibitory rate and population synchrony (the paper's measure: the fraction of excitatory cells firing within a 50 ms window) from 60 s before to 60 s after four kinds of boundary in the authors' scoring, as a percent change from each session's own baseline (−60 to −30 s); median and interquartile range across sessions. The boundaries:

* **Microarousal onset**: the start of a brief wake-like period (up to 40 s) inside nonREM sleep, as scored by the authors.
* **Wake-interruption onset**: the start of a longer wake-like period (40–120 s) during sleep.
* **WAKE → SLEEP** and **SLEEP → WAKE**: the authors' full state switches; only 25 and 3 events, so these two columns are descriptive.

Only microarousals and wake interruptions that directly follow scored sleep are included (227 and 288 events); the rest lie in unscored stretches, where no baseline state is defined. Changes are small (within ±15% of baseline) and no cell type reliably leads the scored boundary at 1 s resolution.
""")
code("""
display(Image("../figures/transitions.png", width=1000))
""")

sc = json.loads((RES / "brain3d" / "showcase.json").read_text())
lag = sc["decoder_sleep_lag_s"]
md(f"""
### 5.4 3D model

A 3D model of the {sc["n_units"]} neurons recorded in session {sc["session"]}, with one probe in the {sc["region_name"]} ({sc["region"]}) of rat {sc["rat"]}, across one of the authors' WAKE→SLEEP switches. Units are placed inside the {sc["region"]} volume of the BrainGlobe `whs_sd_rat_39um` rat atlas (cingulate areas 1 and 2); positions within the region are schematic. Excitatory neurons are green, inhibitory neurons purple, and each marker swells with the spikes it fires. The strip below shows the authors' label and the decoder's prediction for each second; {sc["rat"]} was the held-out rat for these predictions and is one of the two best-decoded rats, and this is the switch where the decoder agrees best with the authors. In the window the decoder is right on {100 * sc["wake_correct"]:.0f}% of the waking seconds and {100 * sc["sleep_correct"]:.0f}% of the sleeping seconds, switches to nonREM {"%.0f s after" % lag if lag else "at"} the authors' boundary, and calls the authors' microarousal that follows WAKE.

* Rotatable version: `results/brain3d/brain3d.html` (open in a browser; play button below the strip).
* Video: `results/brain3d/brain3d.mp4`.
""")
code("""
display(Image("../figures/brain3d_still.png", width=800))
""")

md(f"""
## 6. Interpretation, limitations and improvements

**What the results show**

* Trained on other rats, the network scores a held-out rat at {dec} balanced accuracy over WAKE / nonREM / REM (chance 0.33), against {base} for logistic regression on firing rates, and it wins in {wins("balanced_accuracy")} rats.
* What transfers across rats is nonREM. Separating nonREM from the two activated states (WAKE and REM) reaches {act}, above the baseline in {wins("bal_acc_nonrem_vs_activated")} rats. nonREM has a signature visible in spikes alone: the alternation of population silence and activity (the paper's OFF periods).
* Telling WAKE from REM is the weak point (WAKE recall {rec["WAKE"]}, REM recall {rec["REM"]}). Both are activated cortex with similar firing; the authors themselves separated them with an EMG proxy, which the decoder never sees. The error depends on the rat: in {len(weak_rats)} rats ({", ".join(weak_rats)}) almost no waking second is called WAKE, most are called REM.
* Inhibitory cells alone decode as well as all cells ({v["I_only"][0]:.2f} vs {v["decoder_short"][0]:.2f}), while the same number of excitatory cells reaches only {v["E_matched"][0]:.2f}. A few inhibitory cells carry as much state information as the whole population, which matches the paper's finding that inhibitory firing tracks state most strongly.
* Removing the population convolutions changes the score by {d_pop:+.2f} ({v["no_population"][0]:.2f} vs {v["decoder_short"][0]:.2f}), {pop_verdict}.
* Around the authors' boundaries, firing changes are small and no cell type leads the boundary, consistent with a cortex that follows the subcortical switch rather than driving it.

**Limitations**

* *Information, not causation.* The ablations show which cells carry state information. Cortex is downstream of the hypothalamic-brainstem switch, so nothing here says what causes the state.
* *WAKE is the authors' scored WAKE*: arousals of 7 min or longer, during the lights-on period. Shorter arousals and the unscored 47% of the recording were never targets, so the decoder is untested on them.
* *Ten rats.* Each held-out rat is one data point, and brain region is confounded with rat (2–4 rats per region), so region differences cannot be separated from rat differences.
* *Schematic 3D positions.* The dataset gives region and shank, not coordinates.
* *One run per fold.* One random seed per fold, one network width, and shortened training for the ablations, to fit a laptop CPU.

**Improvements**

* Test the decoder against an independent measure of wakefulness, such as video or a separately recorded EMG, instead of a wake score derived from the LFP on the same probes as the spikes.
* Improve the network: {dec} balanced accuracy leaves clear room. The width (16 filters) and the number of pooling queries (2) were never tuned and each fold was trained once, so a search over width, queries and seeds, with a longer training budget, is the first thing to try.
* Compare the learned per-neuron filters with the paper's OFF-period and spindle definitions, to see what the network actually picked up.

**References**

* Watson BO, Levenstein D, Greene JP, Gelinas JN, Buzsáki G (2016). Network homeostasis and state dynamics of neocortical sleep. *Neuron* 90:839. doi:10.1016/j.neuron.2016.03.036
* Dataset: CRCNS fcx-1, doi:10.6080/K02N506Q
* Saper CB, Scammell TE, Lu J (2005). Hypothalamic regulation of sleep and circadian rhythms. *Nature* 437:1257. doi:10.1038/nature04284

**AI use.** I used an AI assistant (Claude) to write the code in this repository and to edit the text of this notebook. The project idea, the model ideation, the background reading, the direction of troubleshooting and every key decision (what the input is, which cell groups to use, how to evaluate, what to keep and what to cut) were mine.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
NB.parent.mkdir(exist_ok=True); nbf.write(nb, NB)
subprocess.run([str(ROOT / ".venv/bin/jupyter"), "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NB)], check=True)
subprocess.run([str(ROOT / ".venv/bin/jupyter"), "nbconvert", "--to", "html", "--no-input", str(NB), "--output-dir", str(ROOT / "reports")], check=True)
html = ROOT / "reports" / "PSYC40Final.html"                      # keep headings with what follows them when printing
html.write_text(html.read_text().replace("</head>", "<style>@media print { h1, h2, h3, h4 { break-after: avoid-page; } img, table { break-inside: avoid; } }</style></head>", 1))
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if Path(chrome).exists():
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={ROOT / 'reports' / 'PSYC40Final.pdf'}", f"file://{ROOT / 'reports' / 'PSYC40Final.html'}"],
                   check=True, capture_output=True)
print("wrote", NB, "reports/PSYC40Final.html", "reports/PSYC40Final.pdf")
