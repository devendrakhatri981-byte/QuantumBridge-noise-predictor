"""
QuantumBridge — Entry 046a: train ONE final GNN on all 2,317 circuits.

Every prior training run (Entries 041-045) used 5-fold cross-validation
for honest accuracy reporting -- each fold's model only ever saw 80% of
the data. For the public demo, we want a single model trained on ALL
available data (no held-out split) to get the best possible deployed
predictor. Same architecture/capacity as Entry 045 (MAX_N=48, MAX_E=64,
6 node feats, 4 edge feats, H=16, 2 message-passing rounds, 180 epochs).

Saves the trained parameters (plus the feature normalization stats,
which are needed at inference time) to JSON so entry046_score_all_pairs.py
can load a fixed model instead of retraining. Entry 045's 5-fold CV
numbers (CV MAE 1.21, R2 0.971) remain the honest accuracy estimate for
this architecture -- this script's own self-eval MAE is optimistic
since the model saw 100% of what it's being scored on.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.metrics import mean_absolute_error, r2_score

import entry045_train_gnn as m

OUT_PATH = "quantumbridge_data/entry046_final_gnn_params.json"


def params_to_json(params):
    return {name: {"W": layer["W"].tolist(), "b": layer["b"].tolist()}
            for name, layer in params.items()}


def params_from_json(d):
    return {name: {"W": jnp.array(layer["W"]), "b": jnp.array(layer["b"])}
            for name, layer in d.items()}


CKPT_PATH = "quantumbridge_data/entry046_train_checkpoint.json"


def train_final(seed=0, epochs=180, batch_size=64, lr=3e-3, chunk_epochs=25):
    """Chunked/resumable: trains chunk_epochs more epochs than the saved
    checkpoint each call (this sandbox caps commands at ~170s, and 180
    epochs over 2,317 circuits doesn't finish in one call)."""
    import os
    opt = optax.adam(lr)

    if os.path.exists(CKPT_PATH):
        ck = json.load(open(CKPT_PATH))
        params = params_from_json(ck["params"])
        opt_state = opt.init(params)  # optax adam state is cheap to rebuild;
                                        # momentum reset each chunk is a minor
                                        # approximation, acceptable here
        done_epochs = ck["epoch"]
        print(f"resuming from epoch {done_epochs}")
    else:
        key = jax.random.PRNGKey(seed)
        params = m.init_params(key)
        opt_state = opt.init(params)
        done_epochs = 0

    n = len(m.raw)
    train_idx = np.arange(n)
    rng = np.random.RandomState(seed + done_epochs)
    target = min(epochs, done_epochs + chunk_epochs)
    for epoch in range(done_epochs, target):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (m.node_feat_all[idx], m.node_mask_all[idx], m.src_all[idx],
                    m.dst_all[idx], m.edge_feat_all[idx], m.edge_mask_all[idx],
                    m.glob_all_n[idx], m.y_all[idx])
            (l, grads) = jax.value_and_grad(lambda p: m.loss_fn(p, batch)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    json.dump({"params": params_to_json(params), "epoch": target}, open(CKPT_PATH, "w"))
    print(f"checkpoint saved at epoch {target}/{epochs}")
    return params, target >= epochs


if __name__ == "__main__":
    params, finished = train_final()
    if not finished:
        print("not done yet -- rerun to continue training")
        import sys
        sys.exit(0)

    full_batch = (m.node_feat_all, m.node_mask_all, m.src_all, m.dst_all,
                 m.edge_feat_all, m.edge_mask_all, m.glob_all_n, m.y_all)
    _, pred = m.loss_fn(params, full_batch)
    pred = np.array(pred)
    mae = mean_absolute_error(m.y_all, pred)
    r2 = r2_score(m.y_all, pred)
    print(f"final model (trained on everything, self-eval only): MAE={mae*100:.2f} R2={r2:.3f}")
    print("(optimistic self-eval -- Entry 045's 5-fold CV numbers are the honest estimate)")

    out = {
        "params": params_to_json(params),
        "norm": {
            "node_mu": m.node_mu.tolist(), "node_sd": m.node_sd.tolist(),
            "edge_mu": m.edge_mu.tolist(), "edge_sd": m.edge_sd.tolist(),
            "glob_mu": m.glob_mu.tolist(), "glob_sd": m.glob_sd.tolist(),
        },
        "arch": {"MAX_N": m.MAX_N, "MAX_E": m.MAX_E, "H": m.H,
                 "N_NODE_FEATS": m.N_NODE_FEATS, "N_EDGE_FEATS": m.N_EDGE_FEATS},
        "self_eval_mae": float(mae), "self_eval_r2": float(r2),
        "n_train": len(m.raw),
    }
    json.dump(out, open(OUT_PATH, "w"))
    print(f"saved final model -> {OUT_PATH}")
