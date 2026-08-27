"""
QuantumBridge — Entry 057c: train the deployment MC-Dropout model.

The two models trained earlier in Entry 057 each deliberately excluded one
chip's data (to test cross-chip uncertainty calibration) -- neither is fit
to actually deploy, since a Sherbrooke user would get predictions from a
model that never saw Sherbrooke, and vice versa. This trains one more
dropout-enabled model on the FULL 3,217-circuit dataset (both chips, all
data), matching Entry 046's "train one final full-dataset GNN" practice.
Exports params + per-chip normalization stats to plain JSON for scoring.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

import entry056_train_gnn as base
import entry057_mc_dropout as mcd

CKPT_PATH = "quantumbridge_data/entry057_deploy_ckpt.json"
EXPORT_PATH = "quantumbridge_data/entry057_deploy_params.json"
EPOCHS = 100


def train_full(epochs=EPOCHS, batch_size=64, lr=3e-3, chunk_epochs=25, seed=99):
    import os
    train_idx = np.arange(len(base.raw))  # ALL data
    opt = optax.adam(lr)
    if os.path.exists(CKPT_PATH):
        ck = json.load(open(CKPT_PATH))
        params = {k: {"W": jnp.array(v["W"]), "b": jnp.array(v["b"])} for k, v in ck["params"].items()}
        opt_state = opt.init(params)
        done_epochs = ck["epoch"]
    else:
        params = base.init_params(jax.random.PRNGKey(seed))
        opt_state = opt.init(params)
        done_epochs = 0

    rng = np.random.RandomState(seed + done_epochs)
    target = min(epochs, done_epochs + chunk_epochs)
    n_train = len(train_idx)
    step_key = jax.random.PRNGKey(seed + 10000 + done_epochs)
    for epoch in range(done_epochs, target):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (base.node_feat_norm[idx], base.node_mask_all[idx], base.src_all[idx], base.dst_all[idx],
                    base.edge_feat_norm[idx], base.edge_mask_all[idx], base.glob_all_n[idx], base.y_all[idx],
                    base.weights_all[idx])
            step_key, sub = jax.random.split(step_key)
            (l, grads) = jax.value_and_grad(lambda p: mcd.weighted_loss_fn(p, batch, sub)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    json.dump({"params": params_json, "epoch": target}, open(CKPT_PATH, "w"))
    if target < epochs:
        print(f"checkpoint at epoch {target}/{epochs} -- rerun to continue")
        return None
    return params


if __name__ == "__main__":
    params = train_full()
    if params is None:
        import sys
        sys.exit(0)

    export = {
        "params": {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()},
        "node_stats": {chip: [base.node_stats[chip][0].tolist(), base.node_stats[chip][1].tolist()]
                      for chip in ("kyiv", "sherbrooke")},
        "edge_stats": {chip: [base.edge_stats[chip][0].tolist(), base.edge_stats[chip][1].tolist()]
                      for chip in ("kyiv", "sherbrooke")},
        "glob_mu": base.glob_mu.tolist(), "glob_sd": base.glob_sd.tolist(),
        "max_n": base.MAX_N, "max_e": base.MAX_E, "dropout_rate": mcd.DROPOUT_RATE,
    }
    json.dump(export, open(EXPORT_PATH, "w"))
    print(f"exported deployment model -> {EXPORT_PATH}")
