# PSYC40Final: reading sleep and wake from cortical spikes

Dartmouth PSYC 40 final project. I decode the authors' sleep-wake state (WAKE / nonREM / REM) from rat frontal-cortex spiking (CRCNS fcx-1, Watson et al. 2016), test on rats the model never saw, ask which neurons and which features of their firing carry the state (ablations with controls), describe firing around state boundaries, and show the recorded neurons spiking in a rotatable 3D brain model.

**Main deliverable:** [`notebooks/PSYC40Final.ipynb`](notebooks/PSYC40Final.ipynb) (also exported as [`reports/PSYC40Final.pdf`](reports/PSYC40Final.pdf)). It runs from the committed `results/` and `figures/` folders, so it needs no raw data.

**3D model:** [`results/brain3d/brain3d.html`](results/brain3d/brain3d.html) (rotatable, press play) and [`results/brain3d/brain3d.mp4`](results/brain3d/brain3d.mp4).

## What is in here

| folder | contents |
|---|---|
| `src/` | `load.py` and `preprocess.py` (the authors' data, as given, into 1 s labels and unit tables); `dataset.py` (10 ms spike counts per unit, one example = one labeled second); `model.py` (the decoder); `train.py` (leave-one-rat-out training and testing); `baseline.py` (logistic regression on excitatory / inhibitory rate traces); `ablate_dropping.py`; `transitions.py`; `brain3d.py` |
| `scripts/` | data download and QC pipeline; `run_jobs.py` (training scheduler); `summarize_results.py`; `plot_transitions.py`; `build_brain3d.py`; `make_final_notebook.py` |
| `results/` | per-fold metrics and predictions for every experiment, `summary_*.csv`, transitions, 3D model |
| `figures/` | all figures used in the notebook; `figures/qc/` has one QC panel per session |
| `reports/` | `qc_report.html` (visual QC of every session), `qc_table.csv`, the exported notebook |
| `notebooks/` | `01_data.ipynb` (data description and QC), `PSYC40Final.ipynb` (the deliverable) |

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_final_notebook.py        # rebuilds the notebook, HTML and PDF from results/
```

To retrain, you need a free CRCNS account. Put your login in `~/.crcns_credentials` (outside the repo):

```bash
export CRCNS_USERNAME='your_username'
export CRCNS_PASSWORD='your_password'
```

then `chmod 600 ~/.crcns_credentials` and:

```bash
bash scripts/run_unattended.sh                # downloads the sessions in scripts/download_queue.txt and runs QC on each
python -m src.dataset                         # 10 ms spike-count bins per session
python -m src.baseline                        # logistic-regression baseline, 10 folds
python scripts/run_jobs.py results/jobs/queue_A.txt 4   # the CNN: main folds and ablations
python -m src.ablate_dropping                 # neuron-dropping curve from the trained models
python -m src.transitions && python scripts/plot_transitions.py
python scripts/build_brain3d.py
python scripts/summarize_results.py && python scripts/make_final_notebook.py
```

Everything runs on a laptop CPU; one training fold takes about 40 minutes with 2 threads.

## Ground rules I followed

- The authors' spike times, stable-unit set, excitatory / inhibitory labels and sleep-wake state scoring are used unchanged. The only preprocessing that changes values is dividing each unit's spike counts by its own session mean (label-free), which the model needs to learn and to transfer across rats.
- Nothing is excluded on a criterion of my own; exclusions could only come from the authors' notes or a failed integrity check, and none were needed.
- Unscored seconds are never used as targets and never filled in.
- Train/test splits are by rat, never by second or session.

## References

- Watson BO, Levenstein D, Greene JP, Gelinas JN, Buzsáki G (2016). Network homeostasis and state dynamics of neocortical sleep. *Neuron* 90:839. doi:10.1016/j.neuron.2016.03.036
- Dataset: CRCNS fcx-1, doi:10.6080/K02N506Q
- Saper CB, Scammell TE, Lu J (2005). Hypothalamic regulation of sleep and circadian rhythms. *Nature* 437:1257. doi:10.1038/nature04284
