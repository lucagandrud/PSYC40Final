# PSYC40Final: reading sleep and wake from cortical spikes

Decoding WAKE vs SLEEP from rat frontal-cortex spiking (CRCNS fcx-1, Watson et al. 2016), testing on a held-out rat, finding which neuron groups carry state information, and describing how they change around sleep/wake transitions. Includes a rotatable 3D model of the recorded neurons.

**Status:** work in progress. Main notebook: `notebooks/PSYC40Final.ipynb` (coming).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

CRCNS fcx-1 (doi:10.6080/K02N506Q) requires a free CRCNS account. Put your login in `~/.crcns_credentials` (outside the repo):

```bash
export CRCNS_USERNAME='your_username'
export CRCNS_PASSWORD='your_password'
```

then `chmod 600 ~/.crcns_credentials` and run `bash scripts/download_data.sh`. It downloads two sessions (`BWRat20_101013`, `20140526_277um`, about 5 GB total) into `data/raw/`.

## References

- Watson BO, Levenstein D, Greene JP, Gelinas JN, Buzsáki G (2016). Network homeostasis and state dynamics of neocortical sleep. *Neuron* 90:839. doi:10.1016/j.neuron.2016.03.036
- Dataset: doi:10.6080/K02N506Q
- Saper CB, Scammell TE, Lu J (2005). Hypothalamic regulation of sleep and circadian rhythms. *Nature* 437:1257. doi:10.1038/nature04284
