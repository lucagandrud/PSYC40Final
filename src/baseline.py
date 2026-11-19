"""Baseline: logistic regression on the two group traces.

Same input window as the CNN (226 bins of 10 ms), but the raster is collapsed to two
traces before the model sees it: spikes per E cell and spikes per I cell in each bin
(452 numbers per second). Bin position within the window is arbitrary, so this model
can in practice only read overall E and I firing. If the CNN beats it on held-out rats,
firing patterns and relationships between neurons carry state information beyond rate.

Same folds, same balanced draw (each training rat and each state equal) and the same
scoring as the CNN. Features are standardized with the training rats' mean and sd.
"""
from __future__ import annotations

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .dataset import SessionData, load_all, rats_of, CLASS_NAMES, WINDOW, BINS_PER_S, HALF_REACH
from .evaluate import scores, pooled_rat_scores
from .train import folds, RESULTS

N_PER_CLASS_PER_RAT = 2000   # seconds drawn per state per training rat (with replacement if fewer)


def traces(sd: SessionData) -> np.ndarray:
    """(2, T) spikes per E cell and per I cell in each 10 ms bin."""
    out = np.zeros((2, sd.bins.shape[1]), dtype=np.float32)
    for k in (0, 1):
        rows = np.where(sd.types == k)[0]
        if rows.size:
            out[k] = np.asarray(sd.bins[rows]).sum(0) / rows.size
    return out


def features(tr: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    idx = (seconds * BINS_PER_S - HALF_REACH)[:, None] + np.arange(WINDOW)[None, :]
    return tr[:, idx].transpose(1, 0, 2).reshape(seconds.size, -1)    # (B, 452)


def main():
    data = load_all()
    tr = {s: traces(sd) for s, sd in data.items()}
    by_rat = rats_of(data)
    out_root = RESULTS / "baseline"
    summary = []
    for fold in folds(data):
        rng = np.random.default_rng(0)
        X, Y = [], []
        for rat in fold["train"]:                     # the validation rat is left unused, as for the CNN
            for c in range(len(CLASS_NAMES)):
                pool = [(s, sec) for s in by_rat[rat] for sec in data[s].labeled_seconds[data[s].y[data[s].labeled_seconds] == c]]
                take = rng.choice(len(pool), size=N_PER_CLASS_PER_RAT, replace=len(pool) < N_PER_CLASS_PER_RAT)
                for s in by_rat[rat]:
                    secs = np.array([pool[j][1] for j in take if pool[j][0] == s])
                    if secs.size:
                        X.append(features(tr[s], secs)); Y.append(np.full(secs.size, c))
        X, Y = np.concatenate(X), np.concatenate(Y)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), Y)
        res = {}
        d = out_root / fold["test"]; d.mkdir(parents=True, exist_ok=True)
        for s in by_rat[fold["test"]]:
            sd = data[s]
            p = clf.predict_proba(sc.transform(features(tr[s], sd.all_seconds)))
            y_all = sd.y[sd.all_seconds]; lab = y_all >= 0
            res[s] = dict(y_true=y_all[lab], y_pred=p[lab].argmax(1))
            np.savez_compressed(d / f"preds_{s}.npz", seconds=sd.all_seconds, probs=p.astype(np.float32))
        m = dict(test_rat=fold["test"], train_rats=fold["train"], n_train=int(Y.size),
                 rat=pooled_rat_scores(res), sessions={s: scores(r["y_true"], r["y_pred"]) for s, r in res.items()})
        (d / "metrics.json").write_text(json.dumps(m, indent=1))
        r = m["rat"]
        print(f"{fold['test']:9s} baseline bal.acc {r['balanced_accuracy']:.3f} kappa {r['kappa']:.3f}  recall {r['recall']}", flush=True)
        summary.append(m)
    ba = [m["rat"]["balanced_accuracy"] for m in summary]; ka = [m["rat"]["kappa"] for m in summary]
    print(f"mean over rats: bal.acc {np.mean(ba):.3f} ± {np.std(ba):.3f}, kappa {np.mean(ka):.3f} ± {np.std(ka):.3f}")


if __name__ == "__main__":
    main()
