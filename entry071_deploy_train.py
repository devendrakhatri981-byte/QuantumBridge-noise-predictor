"""
QuantumBridge — Entry 071: unified three-chip MC-Dropout deployment model.

Retires the two-lineage split that's existed since Brisbane was added:
Kyiv/Sherbrooke got calibrated MC-Dropout uncertainty (Entry 057, trained
before Brisbane existed, 2-chip data only), while Brisbane got a separate
point-estimate-only combined model (Entry 065/069, no dropout, so no valid
uncertainty). This trains ONE dropout-enabled model on the full current
5,690-circuit, 3-chip dataset (entry068_graph_dataset.json) -- same
architecture, same per-chip normalization (Entry 048), same floor-collapse
up-weighting (Entry 051) -- so all three chips get a genuine, calibrated
uncertainty estimate from a single deployed model instead of two different
ones with different guarantees.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

import entry068_train_gnn as base
import entry071_mc_dropout as mcd

CKPT_PATH = "quantumbridge_data/entry071_deploy_ckpt.json"
EXPORT_PATH = "quantumbridge_data/entry071_deploy_params.json"
EPOCHS = 100


def train_full(epochs=EPOCHS, batch_size=64, lr=3e-3, chunk_epochs=20, seed=99):
    import os
    train_idx = np.arange(len(base.raw))  # ALL data, all 3 chips
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
    tmp = CKPT_PATH + ".tmp"
    json.dump({"params": params_json, "epoch": target}, open(tmp, "w"))
    import os as _os
    _os.replace(tmp, CKPT_PATH)
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
                      for chip in ("kyiv", "sherbrooke", "brisbane")},
        "edge_stats": {chip: [base.edge_stats[chip][0].tolist(), base.edge_stats[chip][1].tolist()]
                      for chip in ("kyiv", "sherbrooke", "brisbane")},
        "glob_mu": base.glob_mu.tolist(), "glob_sd": base.glob_sd.tolist(),
        "max_n": base.MAX_N, "max_e": base.MAX_E, "dropout_rate": mcd.DROPOUT_RATE,
    }
    tmp = EXPORT_PATH + ".tmp"
    json.dump(export, open(tmp, "w"))
    import os
    os.replace(tmp, EXPORT_PATH)
    print(f"exported deployment model -> {EXPORT_PATH}")
