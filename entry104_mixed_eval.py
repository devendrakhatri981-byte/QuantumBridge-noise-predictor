"""
QuantumBridge — Entry 104: evaluation on the MIXED bell/GHZ3 template.
Three tests (same GNN architecture/JIT training as entry100; own-stats
normalization for held-out chips, 5 seeds each):
  A. fold-in  : 5-fold-style single 80/20 split, per-chip normalization,
                model trained on mixed data from all 3 chips.
  B. leave-one-chip-out zero-shot on mixed data (train on 2 chips).
  C. TEMPLATE TRANSFER: train ONLY on bell-only multi-bell circuits
                (entry090 dataset, 3 chips), test on ALL mixed circuits
                (no mixed circuit ever seen) -- does the model extrapolate
                to GHZ components it never trained on?
Baseline everywhere: product of independent per-component v4.1 predictions.
usage: python3 entry104_mixed_eval.py A|B|C
"""
import json, sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.metrics import mean_absolute_error

import entry094_train_multibell as base   # reuses model fns (loads its dataset at import)

MAX_N, MAX_E = base.MAX_N, base.MAX_E
CHIPS = ("kyiv", "sherbrooke", "brisbane")
import os
SEEDS = range(int(os.environ.get("NSEEDS", "5")))


def load(path):
    raw = json.load(open(path))
    pads = [base.pad_graph(g) for g in raw]
    st = lambda i: np.stack([p[i] for p in pads])
    d = {"nf": st(0), "nm": st(1), "src": st(2), "dst": st(3), "ef": st(4), "em": st(5), "gl": st(6),
         "y": np.array([g["aer_ground_truth"] for g in raw], np.float32),
         "b": np.array([g["product_of_marginals_v41"] for g in raw], np.float32),
         "chip": np.array([g["chip"] for g in raw])}
    return d


def stats(d, mask):
    nm = d["nm"][mask].astype(bool); em = d["em"][mask].astype(bool)
    return (d["nf"][mask][nm].mean(0), d["nf"][mask][nm].std(0) + 1e-6,
            d["ef"][mask][em].mean(0), d["ef"][mask][em].std(0) + 1e-6)


def normalize(d, idx_by_stats):
    nfn = np.zeros_like(d["nf"]); efn = np.zeros_like(d["ef"])
    for mask, (a, b, c, e) in idx_by_stats:
        nfn[mask] = (d["nf"][mask] - a) / b; efn[mask] = (d["ef"][mask] - c) / e
    return nfn, efn


@partial(jax.jit, static_argnums=(3,))
def step(params, opt_state, batch, opt):
    g = jax.grad(lambda p: base.loss_fn(p, batch)[0])(params)
    u, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, u), opt_state


def train(d, nfn, efn, glm, gls, idx, seed, epochs=150):
    opt = optax.adam(3e-3); params = base.init_params(jax.random.PRNGKey(seed))
    st = opt.init(params); rng = np.random.RandomState(seed)
    gln = (d["gl"] - glm) / gls
    for _ in range(epochs):
        perm = rng.permutation(len(idx))
        for s in range(0, len(idx), 32):
            i = idx[perm[s:s + 32]]
            params, st = step(params, st, (nfn[i], d["nm"][i], d["src"][i], d["dst"][i], efn[i],
                                           d["em"][i], gln[i], d["y"][i]), opt)
    return params, gln


def predict(params, d, nfn, efn, gln, idx):
    _, p = base.loss_fn(params, (nfn[idx], d["nm"][idx], d["src"][idx], d["dst"][idx], efn[idx],
                                 d["em"][idx], gln[idx], d["y"][idx]))
    return np.array(p)


def pct(d, idx, pred):
    m = mean_absolute_error(d["y"][idx], pred) * 100
    bm = mean_absolute_error(d["y"][idx], d["b"][idx]) * 100
    return (1 - m / bm) * 100, m, bm


if __name__ == "__main__":
    mode = sys.argv[1]
    mx = load("quantumbridge_data/entry103_mixed_graph_dataset.json")
    out = {}
    if mode == "A":
        res = []
        for seed in SEEDS:
            rs = np.random.RandomState(seed); perm = rs.permutation(len(mx["y"]))
            te, tr = perm[:len(perm) // 5], perm[len(perm) // 5:]
            sts = [(mx["chip"] == c, stats(mx, mx["chip"] == c)) for c in CHIPS]
            nfn, efn = normalize(mx, sts)
            glm, gls = mx["gl"].mean(0), mx["gl"].std(0) + 1e-6
            p, gln = train(mx, nfn, efn, glm, gls, tr, seed)
            res.append(pct(mx, te, predict(p, mx, nfn, efn, gln, te)))
        out["fold_in"] = [r[0] for r in res]
        print("A fold-in vs baseline: mean %+.1f%% (per-seed %s) | MAE %.2f vs base %.2f" % (
            np.mean(out["fold_in"]), [round(x, 1) for x in out["fold_in"]],
            np.mean([r[1] for r in res]), np.mean([r[2] for r in res])))
    elif mode == "B":
        for hold in (CHIPS if len(sys.argv) < 3 else (sys.argv[2],)):
            tr = np.where(mx["chip"] != hold)[0]; te = np.where(mx["chip"] == hold)[0]
            sts = [(mx["chip"] == c, stats(mx, mx["chip"] == c)) for c in CHIPS]   # own stats incl. held-out
            nfn, efn = normalize(mx, sts)
            glm, gls = mx["gl"][tr].mean(0), mx["gl"][tr].std(0) + 1e-6
            vals = []
            for seed in SEEDS:
                p, gln = train(mx, nfn, efn, glm, gls, tr, seed)
                vals.append(pct(mx, te, predict(p, mx, nfn, efn, gln, te))[0])
            out[hold] = vals
            print("B holdout %-11s mean %+.1f%% pos %d/5 %s" % (hold, np.mean(vals), sum(v > 0 for v in vals),
                                                             [round(v, 1) for v in vals]), flush=True)
    else:  # C: template transfer
        bell = load("quantumbridge_data/entry090_multibell_graph_dataset.json")
        sel = np.isin(bell["chip"], CHIPS)
        sb = [(bell["chip"] == c, stats(bell, bell["chip"] == c)) for c in CHIPS]
        nfb, efb = normalize(bell, sb)
        glm, gls = bell["gl"].mean(0), bell["gl"].std(0) + 1e-6
        tr = np.where(sel)[0]
        sm = [(mx["chip"] == c, stats(mx, mx["chip"] == c)) for c in CHIPS]
        nfm, efm = normalize(mx, sm)
        glm_m = (mx["gl"] - glm) / gls
        te = np.arange(len(mx["y"])); vals = []
        for seed in SEEDS:
            p, _ = train(bell, nfb, efb, glm, gls, tr, seed)
            vals.append(pct(mx, te, predict(p, mx, nfm, efm, glm_m, te)))
        out["transfer"] = [v[0] for v in vals]
        for k, nm in (("ghz", "has GHZ"), ("all", "all")):
            pass
        print("C transfer (bell-only model -> mixed): mean %+.1f%% per-seed %s | MAE %.2f vs base %.2f" % (
            np.mean(out["transfer"]), [round(x, 1) for x in out["transfer"]],
            np.mean([v[1] for v in vals]), vals[0][2]))
    json.dump(out, open(f"quantumbridge_data/entry104_{mode}{sys.argv[2] if len(sys.argv) > 2 else ''}_results.json", "w"), indent=2)
