"""
QuantumBridge — Entry 105: deployable multi-bell model (5 chips, seed ensemble).
Trains K seeds on ALL entry090 multi-bell graphs (kyiv/sherbrooke/brisbane/osaka/quebec),
per-chip normalization from each chip's own circuits (so for these trained chips the
'own-stats' caveat does not apply at inference). Saves ensemble params + stats for the
API's /predict/multi. Ensemble spread = uncertainty (no MC-dropout in this architecture).
usage: python3 entry105_deploy_multibell.py [K]
"""
import json, sys
import numpy as np
import jax, optax
import entry094_train_multibell as base
import entry104_mixed_eval as ev

OUT = "quantumbridge_data/entry105_multibell_deploy.json"
CHIPS5 = ("kyiv", "sherbrooke", "brisbane", "osaka", "quebec")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5

d = ev.load("quantumbridge_data/entry090_multibell_graph_dataset.json")
sts = {c: ev.stats(d, d["chip"] == c) for c in CHIPS5}
nfn, efn = ev.normalize(d, [(d["chip"] == c, sts[c]) for c in CHIPS5])
glm, gls = d["gl"].mean(0), d["gl"].std(0) + 1e-6
idx = np.arange(len(d["y"]))
members = []
for s in range(K):
    p, _ = ev.train(d, nfn, efn, glm, gls, idx, s)
    members.append({k: {"W": np.array(v["W"]).tolist(), "b": np.array(v["b"]).tolist()} for k, v in p.items()})
    print("trained seed", s, flush=True)
json.dump({"members": members, "glob_mu": glm.tolist(), "glob_sd": gls.tolist(),
           "node_stats": {c: [sts[c][0].tolist(), sts[c][1].tolist()] for c in CHIPS5},
           "edge_stats": {c: [sts[c][2].tolist(), sts[c][3].tolist()] for c in CHIPS5},
           "n_train": int(len(idx)), "max_n": base.MAX_N, "max_e": base.MAX_E}, open(OUT, "w"))
print("saved", OUT, "train n", len(idx))
