"""Neuron-dropping curve: the trained decoder, tested on random subsets of k neurons.

Uses each fold's trained model (results/decoder/<rat>/model.pt) on its own test rat, with
k = 1, 2, 5, 10, 20 random units (20 draws each; each draw predicts an interleaved 1/20 of
the labeled seconds, so every second is scored once and all 20 draws count) and all units.
No retraining: the question is how many neurons the trained readout needs.
Writes results/dropping/summary.csv.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch

from .dataset import load_all, rats_of, CLASS_NAMES
from .model import SpikeNet
from .evaluate import scores, pooled_rat_scores
from .train import RESULTS, predict

KS = [1, 2, 5, 10, 20]
DRAWS = 20


def main(exp: str = "decoder"):
    torch.set_num_threads(2)
    data = load_all()
    for sd in data.values():
        sd.input_mode = "relative"
    rows_out = []
    for rat, sessions in sorted(rats_of(data).items()):
        mfile = RESULTS / exp / rat / "model.pt"
        if not mfile.exists():
            print(f"{rat}: no model yet, skipping"); continue
        meta = json.loads((RESULTS / exp / rat / "metrics.json").read_text())
        model = SpikeNet(filters=meta["filters"], queries=meta["queries"], norm=meta.get("norm", False))
        model.load_state_dict(torch.load(mfile)); model.eval()
        rng = np.random.default_rng(0)
        for k in KS + ["all"]:
            res = {}
            for s in sessions:
                sd = data[s]; secs = sd.labeled_seconds
                if k != "all" and k > sd.n_units:
                    continue
                p = np.zeros((secs.size, len(CLASS_NAMES)))
                if k == "all":
                    p = predict(model, sd, secs, np.arange(sd.n_units))
                else:
                    for d in range(DRAWS):
                        sl = slice(d, None, DRAWS)
                        rows = np.sort(rng.choice(sd.n_units, size=k, replace=False))
                        p[sl] = predict(model, sd, secs[sl], rows)
                res[s] = dict(y_true=sd.y[secs], y_pred=p.argmax(1))
            if res:
                r = pooled_rat_scores(res)
                rows_out.append(dict(rat=rat, k=k, balanced_accuracy=r["balanced_accuracy"], kappa=r["kappa"]))
                print(f"{rat:9s} k={k!s:4s} bal.acc {r['balanced_accuracy']:.3f} kappa {r['kappa']:.3f}", flush=True)
    df = pd.DataFrame(rows_out)
    (RESULTS / "dropping").mkdir(exist_ok=True)
    df.to_csv(RESULTS / "dropping" / "summary.csv", index=False)
    print(df.groupby("k")[["balanced_accuracy", "kappa"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
