"""Train and test the decoder, leaving one rat out.

Folds: the 10 rats in alphabetical order. In fold i, rat i is the test rat, rat i+1 is
the validation rat (early stopping and setting choice), and the other 8 train. All of a
rat's sessions stay on the same side of the split. The test rat's labels are used only
to score it.

Early stopping: validation cross-entropy on the validation rat, checked every epoch, no stop
before MIN_EPOCHS; the state with the lowest validation loss is kept.

Training examples: each epoch draws the same number of labeled seconds from every
training rat and from every state within a rat (WAKE / nonREM / REM), so no rat and no
state dominates. The loss is plain cross-entropy on the authors' label of each second.
Unlabeled seconds are never targets; they only appear as the 0.63 s of surrounding spikes.

Usage:
  python -m src.train --exp decoder --filters 16 --queries 2 --test BWRat20
  python -m src.train --exp E_only --variant E_only --filters 16 --queries 2 --test all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import SessionData, load_all, rats_of, CLASS_NAMES, ROOT
from .model import SpikeNet
from .evaluate import scores, pooled_rat_scores

RESULTS = ROOT / "results"
MIN_I_FOR_EI = 3          # sessions with fewer inhibitory cells are skipped in E/I comparisons
N_PER_CLASS_PER_RAT = 256  # examples drawn per state per training rat per epoch
BATCH = 64
MAX_EPOCHS = 12
MIN_EPOCHS = 6           # no early stop before this many epochs
PATIENCE = 3
VAL_PER_CLASS = 500        # validation subsample per state (fixed per run)


# ----------------------------------------------------------------------------- variants
def session_allowed(sd: SessionData, variant: str) -> bool:
    if variant in ("E_only", "I_only", "E_matched"):
        return int((sd.types == 1).sum()) >= MIN_I_FOR_EI
    return True


def pick_rows(sd: SessionData, variant: str, rng: np.random.Generator) -> np.ndarray:
    """Which neuron rows a variant feeds the model for one batch."""
    e, i = np.where(sd.types == 0)[0], np.where(sd.types == 1)[0]
    if variant == "E_only":
        return e
    if variant == "I_only":
        return i
    if variant == "E_matched":                    # as many E cells as the session has I cells
        return np.sort(rng.choice(e, size=i.size, replace=False))
    return np.arange(sd.n_units)


def make_model(args) -> SpikeNet:
    return SpikeNet(filters=args.filters, queries=args.queries,
                    relational=args.variant != "no_relational",
                    population=args.variant != "no_population",
                    norm=args.norm)


# ----------------------------------------------------------------------------- folds
def folds(data: dict[str, SessionData]) -> list[dict]:
    rats = sorted(rats_of(data))
    out = []
    for k, test in enumerate(rats):
        val = rats[(k + 1) % len(rats)]
        train = [r for r in rats if r not in (test, val)]
        out.append(dict(test=test, val=val, train=train))
    return out


# ----------------------------------------------------------------------------- sampling
def epoch_batches(data, train_rats, variant, rng):
    """Balanced draw: N_PER_CLASS_PER_RAT seconds per state per rat, batched within session."""
    by_rat = rats_of(data)
    draws: dict[str, list[tuple[int, int]]] = {}
    for rat in train_rats:
        sess = [s for s in by_rat[rat] if session_allowed(data[s], variant)]
        for c in range(len(CLASS_NAMES)):
            pool = [(s, sec) for s in sess for sec in data[s].labeled_seconds[data[s].y[data[s].labeled_seconds] == c]]
            if not pool:
                continue
            take = rng.choice(len(pool), size=N_PER_CLASS_PER_RAT, replace=len(pool) < N_PER_CLASS_PER_RAT)
            for j in take:
                s, sec = pool[j]
                draws.setdefault(s, []).append((sec, c))
    batches = []
    for s, items in draws.items():
        items = [items[j] for j in rng.permutation(len(items))]
        for b in range(0, len(items), BATCH):
            chunk = items[b:b + BATCH]
            batches.append((s, np.array([x[0] for x in chunk]), np.array([x[1] for x in chunk])))
    return [batches[j] for j in rng.permutation(len(batches))]


def val_set(data, val_rat, variant, rng):
    by_rat = rats_of(data)
    out = []
    for s in by_rat[val_rat]:
        sd = data[s]
        if not session_allowed(sd, variant):
            continue
        for c in range(len(CLASS_NAMES)):
            secs = sd.labeled_seconds[sd.y[sd.labeled_seconds] == c]
            if secs.size:
                out.append((s, np.sort(rng.choice(secs, size=min(VAL_PER_CLASS, secs.size), replace=False))))
    return out


# ----------------------------------------------------------------------------- run
@torch.no_grad()
def predict(model, sd: SessionData, seconds: np.ndarray, rows: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    types = torch.from_numpy(sd.types[rows]).long()
    probs = []
    for b in range(0, seconds.size, batch):
        x = torch.from_numpy(sd.window(seconds[b:b + batch], rows))
        probs.append(torch.softmax(model(x, types), dim=-1).numpy())
    return np.concatenate(probs)


E_MATCHED_SUBSETS = 50


def evaluate_sessions(model, data, sessions, variant, rng, predict_all=False):
    """Per-session class probabilities (every second, or labeled seconds only); scores on
    labeled seconds. E_matched: 50 random E subsets of the session's I count, each predicting
    an interleaved 1/50 of the seconds, so every second is scored once and all 50 draws count."""
    out = {}
    for s in sessions:
        sd = data[s]
        if not session_allowed(sd, variant):
            continue
        secs = sd.all_seconds if predict_all else sd.labeled_seconds
        p = np.zeros((secs.size, len(CLASS_NAMES)))
        if variant == "E_matched":
            for k in range(E_MATCHED_SUBSETS):
                sl = slice(k, None, E_MATCHED_SUBSETS)
                p[sl] = predict(model, sd, secs[sl], pick_rows(sd, variant, rng))
        else:
            p = predict(model, sd, secs, pick_rows(sd, variant, rng))
        y_all = sd.y[secs]
        lab = y_all >= 0
        out[s] = dict(seconds=secs, probs=p.astype(np.float32), y_true=y_all[lab], y_pred=p[lab].argmax(1))
    return out


def run_fold(args, data, fold, out_dir: Path):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)
    model = make_model(args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    vset = val_set(data, fold["val"], args.variant, rng)
    best, best_loss, best_state, bad, history = -1.0, np.inf, None, 0, []
    t_start = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train(); t0 = time.time(); losses = []
        for s, secs, y in epoch_batches(data, fold["train"], args.variant, rng):
            sd = data[s]; rows = pick_rows(sd, args.variant, rng)
            x = torch.from_numpy(sd.window(secs, rows)); types = torch.from_numpy(sd.types[rows]).long()
            loss = F.cross_entropy(model(x, types), torch.from_numpy(y))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        # validation on the validation rat: cross-entropy (the stopping criterion; balanced
        # accuracy on one rat is too noisy to stop on) and balanced accuracy (reported)
        yt, yp, pp = [], [], []
        for s, secs in vset:
            sd = data[s]
            p = predict(model, sd, secs, pick_rows(sd, args.variant, rng))
            yt.append(sd.y[secs]); yp.append(p.argmax(1)); pp.append(p)
        yt_, pp_ = np.concatenate(yt), np.concatenate(pp)
        v = scores(yt_, np.concatenate(yp))["balanced_accuracy"]
        vloss = float(-np.log(np.clip(pp_[np.arange(yt_.size), yt_], 1e-9, 1)).mean())
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=vloss, val_bal_acc=v, seconds=time.time() - t0))
        print(f"  epoch {epoch:2d} loss {np.mean(losses):.3f} val loss {vloss:.3f} bal.acc {v:.3f} ({time.time()-t0:.0f} s)", flush=True)
        if vloss < best_loss:
            best_loss, best, bad = vloss, v, 0; best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE and epoch + 1 >= MIN_EPOCHS:
                break
    model.load_state_dict(best_state)
    # test: every second of every session of the test rat
    test_sessions = rats_of(data)[fold["test"]]
    res = evaluate_sessions(model, data, test_sessions, args.variant, rng, predict_all=args.predict_all)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    per_session = {s: scores(r["y_true"], r["y_pred"]) for s, r in res.items()}
    for s, r in res.items():
        np.savez_compressed(out_dir / f"preds_{s}.npz", seconds=r["seconds"], probs=r["probs"])
    metrics = dict(test_rat=fold["test"], val_rat=fold["val"], train_rats=fold["train"],
                   variant=args.variant, filters=args.filters, queries=args.queries, seed=args.seed,
                   input=args.input, norm=args.norm, lr=args.lr,
                   n_params=model.n_params(), best_val_bal_acc=best, epochs=len(history),
                   max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS, patience=PATIENCE, epoch_size=N_PER_CLASS_PER_RAT, best_val_loss=best_loss,
                   train_minutes=(time.time() - t_start) / 60, history=history,
                   rat=pooled_rat_scores(res) if res else None, sessions=per_session)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    r = metrics["rat"]
    print(f"{fold['test']:9s} test bal.acc {r['balanced_accuracy']:.3f} kappa {r['kappa']:.3f} "
          f"(val {best:.3f}, {len(history)} epochs, {metrics['train_minutes']:.1f} min)", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="results subfolder name")
    ap.add_argument("--variant", default="full",
                    choices=["full", "no_relational", "no_population", "E_only", "I_only", "E_matched"])
    ap.add_argument("--filters", type=int, default=16)
    ap.add_argument("--queries", type=int, default=2)
    ap.add_argument("--test", default="all", help="test rat name, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--min-epochs", type=int, default=None)
    ap.add_argument("--epoch-size", type=int, default=None,
                    help="labeled seconds drawn per state per training rat per epoch (default 256)")
    ap.add_argument("--input", default="relative", choices=["counts", "relative"],
                    help="counts divided by each unit's own session mean (label-free), or raw spike counts")
    ap.add_argument("--norm", action="store_true", help="normalization layers inside the network (off by default)")
    ap.add_argument("--predict-all", action="store_true",
                    help="predict every second of the test sessions (default: labeled seconds only)")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    global MAX_EPOCHS, MIN_EPOCHS, PATIENCE, N_PER_CLASS_PER_RAT
    if args.max_epochs:
        MAX_EPOCHS = args.max_epochs
    if args.min_epochs:
        MIN_EPOCHS = args.min_epochs
    if args.patience:
        PATIENCE = args.patience
    if args.epoch_size:
        N_PER_CLASS_PER_RAT = args.epoch_size
    data = load_all()
    for sd in data.values():
        sd.input_mode = args.input
    for fold in folds(data):
        if args.test != "all" and fold["test"] != args.test:
            continue
        out_dir = RESULTS / args.exp / fold["test"]
        if (out_dir / "metrics.json").exists():
            print(f"{fold['test']}: done already, skipping"); continue
        print(f"fold test={fold['test']} val={fold['val']} variant={args.variant} C={args.filters} Q={args.queries}", flush=True)
        run_fold(args, data, fold, out_dir)


if __name__ == "__main__":
    main()
