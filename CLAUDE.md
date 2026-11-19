# PSYC40Final

Dartmouth PSYC 40 (Computational Neuroscience) final project, using the CRCNS fcx-1 dataset: rat frontal cortex spiking across wake and sleep (Watson et al. 2016, Neuron).

## Ground rules for the agent
1. **Inspect, never assume.** Before using any file, list its actual contents (variable names, shapes, types, columns) and build code on what's there. If this file and the data disagree, the data wins; tell the user.
2. **Work in phases. Stop at every gate**, report in a few lines, and wait for the user's go-ahead.
3. **No model code until the user approves an architecture in Phase 2.**
4. **Keep messages concise, readable and direct, ending with the next step.** Explain concepts plainly when asked.
5. **Never print, log or commit credentials.**
6. **Commit and push at the end of every phase.**

## Project objective (context for every decision)
1. Decode WAKE vs SLEEP from cortical spiking, and test on held-out rats.
2. Find which neuron groups carry state information (ablations, with controls).
3. Describe how each group's firing changes around transitions between sleep and wake-like states.
4. Show it all in a rotatable 3D model of the recorded neurons spiking over time, colored by cell type and separated by brain region.

**Framing:** sleep and wake are switched by hypothalamic and brainstem circuits (Saper, Scammell & Lu 2005, Nature 437:1257). Frontal cortex is downstream of that switch, so cortical changes are read as consequences of it. Ablations show *information*, not causation; transition timing shows which groups *lead* the scored boundary, not causation.

**Course deliverable:** one main notebook covering:
- data description and visualization, including preprocessing
- objective
- model and why it was chosen
- metric
- interpretation, limitations and improvements

Submitted as a public GitHub repo.

**Compute:** everything planned runs on a laptop CPU in minutes. Do not set up GPUs or Colab.

---

## Phase 0: Setup
State of the machine when you start:
- You are in the empty `PSYC40Final` folder, which contains only this file.
- The user has run `gh auth login`.
- The user's CRCNS login is saved in `~/.crcns_credentials` as `export CRCNS_USERNAME=...` and `export CRCNS_PASSWORD=...` lines. `source` it inside scripts; never read it back out to the user.

Steps:
1. **Repo.**
   - `git init`
   - folders: `data/raw`, `data/processed`, `data/manifests`, `src`, `scripts`, `notebooks`, `figures`, `reports`
   - `.gitignore` covering `data/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.tar.gz`
2. **Environment.**
   - Create a Python virtual environment in `.venv`.
   - Install numpy, scipy, pandas, scikit-learn, matplotlib, plotly, kaleido, openpyxl, mat73, h5py, jupyter, ipykernel, brainglobe-atlasapi.
   - Write `requirements.txt`.
   - Create `.vscode/settings.json` pointing the interpreter at `.venv`.
   - Check whether `ffmpeg` is installed; if not, tell the user to run `brew install ffmpeg`.
3. **GitHub.** Write a README stub, commit, and run `gh repo create PSYC40Final --public --source=. --push`.

## Phase 0.5: Data download (runs in the background; Phase 1 starts as soon as the first session lands)
CRCNS serves files through a login-form POST: `curl -L --fail -o <out> --data-urlencode "username=$CRCNS_USERNAME" --data-urlencode "password=$CRCNS_PASSWORD" --data-urlencode "fn=fcx-1/<path>" --data-urlencode "submit=Login" "https://portal.nersc.gov/project/crcns/download/fcx-1/<path>"`. A failed login returns an HTML page, not the file, so check every download's type.

1. **Fetch the small reference files first** into `data/manifests/`:
   - `checksums.md5`
   - `filelist.txt`
   - `data/0_data_file_list.txt` (the list of every file inside every tarball)
   - `docs/crcns_fcx-1_data_description.pdf`
   - `docs/WatsonSleepHomestasis2016Table.xlsx` (per-session summary and notes)
2. **Build the session list from `filelist.txt`.** Do not hardcode it. Parse the xlsx and print its columns and session notes, so session flags come from the authors' table rather than from this file.
3. **Write `scripts/download_data.sh`** (idempotent and resumable). For each session tarball:
   - Check free disk space first; require 1.5× the tarball size.
   - Download it.
   - Verify its MD5 against `checksums.md5`; on a mismatch, retry once, then flag it.
   - Extract, skipping `*.eeg` (raw LFP, about 85% of the volume and unused by the plan) and `*ClusterQualityMeasures*`.
   - Delete the tarball, and log to `data/download.log`.
   - Skip sessions already extracted and verified.
4. **Order:** `BWRat20_101013` first, then `20140526_277um` (smallest clean sessions), then every remaining session, smallest first. Download all of them, including sessions the authors' notes flag; flags are applied at analysis time, not by skipping the download.
5. **Check the extracted files.** After each extraction, compare the session's files against `0_data_file_list.txt` (minus the excluded files) and record any that are missing.

Total is about 90 GB to download, but only a few GB stay on disk. Report progress when asked.

**Gate 0: report the repo URL, download status, and any MD5 or missing-file problems.**

## Phase 1: Inventory, loading, validation
Run on the first session the moment it lands, then on each new one.

### 1a. Inventory (before writing any loader)
For every file type in a session, record into `data/manifests/inventory.csv`:
- every variable or column name, its type, shape or length, and a short sample
- whether the file is MATLAB v5 or v7.3 (HDF5)

Use `scipy.io.whosmat`/`loadmat`, falling back to `mat73`/`h5py`. Read the data description PDF and match each documented variable to what actually exists. Prefer plain array formats (cell arrays, n×2 time-pair arrays) over MATLAB object formats, which Python can't read. Report any documented variable that's missing or named differently.

**Gate 1a: show the user the inventory summary and the proposed loader mapping before writing the loaders.**

### 1b. Loaders and processed format
Write `src/load.py` and `src/preprocess.py`. For each session, write to `data/processed/<session>/`:
- **spike times:** per unit, in seconds
- **unit table:**
  - unit id
  - putative type (excitatory/inhibitory), with the label source (definite vs waveform-only)
  - shank, channel and region, plus how each was derived
  - firing rate in each state
  - stability metric
- **state intervals:** per state type, as given
- **1 s label vector:** WAKE / nonREM / REM / microarousal / wake-interruption / unlabeled. Document the precedence rule used where intervals overlap.
- **event times:** UP/DOWN states and spindles, if present
- **session metadata:** duration, recording file intervals, the good-sleep interval, rat id, region(s), and flags from the authors' table

### 1c. Validation and anomaly checks
Each check reports PASS / WARN / FAIL per session into `reports/qc_table.csv`.

| Area | Checks |
|---|---|
| Units | Count matches the authors' table and the documented stable-unit counts. Excitatory + inhibitory counts are consistent with the stable-unit file. Fewer than 3 inhibitory units is a WARN. |
| Spikes | Times are sorted, non-negative and within the session. No duplicates. Refractory violations (inter-spike interval under 1.5 ms) are reported as a fraction per unit. Units with zero spikes in any analyzed state are listed. |
| Stability | Each unit's firing rate in the first vs last third of the session; a change of more than 3× is a WARN. Any unexplained session-wide rate jumps are found and reported. |
| States | Intervals are within range and well-formed (start < stop). Overlaps between state types are reported. Minutes per state, number of WAKE→SLEEP and SLEEP→WAKE transitions, and number of microarousal and wake-interruption onsets. |
| Physiology | Excitatory and inhibitory rate distributions per state vs Watson 2016 medians (excitatory about 0.7–0.9 Hz, inhibitory about 4.7–5.6 Hz; heavy right tail for excitatory). Medians more than 3× off are a WARN. Inhibitory median rate should exceed excitatory. |
| Anatomy | Every unit maps to shank, channel and region, or the gaps are listed. Sessions recording two regions are identified. |
| Simultaneity | For session pairs that look simultaneous (e.g. the Dino ACC/mPFC pairs from the same date), verify from the metadata (duration, recording file intervals, matching state intervals) whether they share a clock. Report confirmed / not confirmed. Do not assume. |
| Flags | Authors' table notes (instability, shocks/fear conditioning, maze training, injections, unstable after a timestamp) are applied as session or time-window exclusion flags, listed explicitly. |

### 1d. Visual proof
- **`reports/qc_report.html`** (open it for the user):
  - a per-session panel with:
    - a hypnogram (state timeline) over the full session
    - a 120 s raster across a WAKE→SLEEP transition, colored by cell type
    - excitatory and inhibitory firing-rate histograms per state on a log scale
    - unit stability scatter (first vs last third)
    - the session's QC row
  - a cross-session summary at the top:
    - sessions × checks heatmap (PASS/WARN/FAIL)
    - total units by type and region
    - total minutes per state
    - transition and event counts
    - which sessions are cleared for modeling
- Save the same figures as PNGs in `figures/qc/`.
- Write `notebooks/01_data.ipynb` documenting the data, preprocessing and QC. This becomes the data description section of the deliverable.

**Gate 1: open the QC report, then summarize in at most 8 lines:**
- what's clean
- what's flagged and why
- what's excluded
- total usable rats, sessions and units
- whether the data is ready for modeling

## Phase 2: Model discussion (no code)
Present a short comparison table: what each option is, what it's good at, what it assumes, and its fit to this data.
- Logistic regression on population features
- Small multilayer perceptron
- 1D convolutional network over a time window
- Recurrent network (GRU/LSTM)
- Hidden Markov model (states plus transition probabilities; the classic fit for sleep scoring)
- Transformer: explain why it's likely overkill at this data size.

Also present the design choices:
- **Input representation:** session-independent population features vs per-neuron counts (neuron counts differ per session).
- **Bin size and window length.**
- **Label scheme:** WAKE vs SLEEP, or 3 classes.
- **Evaluation:** hold out whole rats, never individual bins; balanced accuracy and Cohen's kappa.

Recommend one option with reasons; the user decides.

**Gate 2: wait for the user's choice.**

## Planned analyses (build only after Gate 2, in this order, stopping after each)
1. **Decoder** as approved, evaluated with held-out rats across all cleared sessions.
2. **Ablations, each with its control:**
   - **Excitatory vs inhibitory:** subsample excitatory units to the same count as inhibitory, 50 repeats; skip sessions with fewer than 3 inhibitory units.
   - **Neuron-dropping curve:** k = 1, 2, 5, 10, 20, all; 20 repeats each; mean with a spread band.
   - **Fast vs slow firing:** excitatory top vs bottom third by WAKE rate, also compared at matched total spike counts by thinning the fast group's spikes.
   - **Rate-only surrogate:** Poisson spikes at each unit's per-state mean rate. If real data beats the surrogate, timing and coordination carry information beyond rates.
3. **Transitions:**
   - Align firing to microarousal and wake-interruption onsets (many events) and to the full WAKE↔SLEEP transitions (few events; descriptive only), ±60 s at 1 s bins.
   - Plot per group (excitatory fast, excitatory slow, inhibitory) plus synchrony, and report which group leads the boundary.
   - Interpret against the subcortical switch.
4. **3D model:**
   - **Atlas:** BrainGlobe `whs_sd_rat_39um`. List its structure names and map the dataset's region labels to the closest structures; report the mapping.
   - **Placement:** units sit inside their region meshes, grouped by shank; within-region positions are schematic, and the caption says so.
   - **Coloring:** excitatory-fast / excitatory-slow / inhibitory; units pulse on their spikes.
   - **Showcase:** a session pair confirmed simultaneous in Phase 1 (two regions, same rat, same clock), across a WAKE→SLEEP transition, with a timeline of true and predicted state.
   - **Outputs:** a rotatable Plotly HTML and an mp4.

## Final notebook (`notebooks/PSYC40Final.ipynb`)
1. Data description and visualization, including preprocessing and QC.
2. Objective, framed against the subcortical sleep-wake switch.
3. Model: the choice, the alternatives considered, and why.
4. Metrics.
5. Results and interpretation.
6. Limitations and improvements, including:
   - labels were scored from LFP and an LFP-derived EMG proxy on the same probes as the spikes, so labels and inputs share signal
   - information, not causation
   - cortex is downstream of the switch
   - schematic 3D positions
   - deep-layer frontal cortex only
7. 3D model.

**Cite:**
- Watson BO, Levenstein D, Greene JP, Gelinas JN, Buzsáki G (2016) Neuron 90:839, doi:10.1016/j.neuron.2016.03.036
- dataset: doi:10.6080/K02N506Q
- Saper CB, Scammell TE, Lu J (2005) Nature 437:1257, doi:10.1038/nature04284

---

## Updates from the user (2026-11-18, deadline midnight)
- **Download scope:** 10 rats, up to 3 sessions each; Templeton excluded (authors' notes: "all unstable", "all E & Hippoc too much strange HF noise"). Round-robin queue in `data/manifests/download_queue.txt`: round 1 = one session per rat (BWRat20_101013 and 20140526_277um first), round 2 = second session, round 3 = third. Dino's ACC/mPFC same-day pair counts as 2 of its 3. Within a rat, prefer more units and no instability notes.
- **Unattended:** `scripts/run_unattended.sh` runs downloads + automatic QC under `caffeinate -i` + `nohup`. Progress: `bash scripts/status.sh`.
- **Automatic QC:** every session that finishes extracting is inventoried and validated automatically, appended to `reports/qc_table.csv`, and `reports/qc_report.html` is regenerated. Only interrupt the user if a session fails or a check would block modeling.
- **Dino regions:** Dino ACC and Dino mPFC sessions are distinct recordings with distinct region labels. Never pool or generalize across them without saying so; region effects are confounded with rat.
- **Time budget (5 h):** 0:00–0:45 Phase 1 on 2 sessions; 0:45–1:15 model discussion; 1:15–2:30 decoder + evaluation; 2:30–3:30 ablations + transitions; 3:30–4:15 3D model; 4:15–5:00 final rerun on all cleared sessions, notebook, push. If behind, cut the rate-only surrogate first, then transitions. Never cut the decoder, the controlled ablations, or the 3D model.
- **BWRat17** (fear conditioning / shocks): QC and the authors' notes decide usability; report the verdict.
- **Repo writing:** first person singular ("I"), never "we". Describe only the data actually used; don't mention sessions not downloaded or used.

## Guiding principle: the authors' data and curation, as given
- Use the authors' values and curation unchanged: their spike times, their stable-unit set (`SStable`, already filtered for instability with their cluster-quality measures), their E/I labels, and their state scoring.
- Never exclude units or time on a metric I made up. Exclusions come only from the authors' own notes (injections, "unstable after X s") or from a failed integrity check (file won't load, counts don't match, times out of range), and every exclusion is listed.
- Never state a guess as a fact. If something is an inference, label it as one or check it in the data first.
- No synthetic data in the data pipeline. Unscored seconds are excluded, never filled in.

## Paper definitions and notes (Watson 2016 supplementary methods)
- **OFF periods:** 75–1250 ms of population silence. The authors' `OFFIntsTimePairFormat` contains intervals of 1–1211 ms, so it does not follow this definition; compute OFF periods from the spikes with the paper's definition instead.
- **Population synchrony:** fraction of excitatory cells firing at least one spike in a 50 ms window, stepped every 25 ms.
- **Limitations to state:** WAKE labels cover only arousals of 7 min or longer, which go with active behavior, so the decoder separates *active* wake from sleep. All recordings are from the lights-on period. Probes were advanced up to 300 µm per day, so sessions from the same rat mostly but not entirely record different neurons; this supports splitting train/test by rat.

## Injection sessions (BWRat20_101513, BWRat21_121613, BWRat21_121813)
For each: use `RecordingFileIntervals` to find the boundary between the recording files named in the authors' note (the injection time); check whether any spikes or state labels fall after it; compare with `GoodSleepInterval`; if post-injection data exists, exclude it automatically and add a QC flag. Report one line per session.
