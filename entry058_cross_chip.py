"""
QuantumBridge — Entry 058b: cross-chip generalization + floor-collapse check
at the new 80/104-capacity, 2,329-circuit scale.

Combines all three fixes at once: per-chip normalization + no chip-identity
leakage (Entry 048), floor-collapse 5x loss up-weighting (Entry 051), and
the larger MAX_N/MAX_E=80/104 capacity fit to the new 65-node/99-edge max
circuits (Entry 052/053). Re-runs the exact Kyiv<->Sherbrooke cold-transfer
test from Entries 047/048/050/051 to confirm all three hold up together.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.metrics import mean_absolute_error, r2_score

import entry058_train_gnn as base

FC_WEIGHT = 5.0


def is_floor_collapse(i):
    g = base.raw[i]
    return abs(g["aer_ground_truth"] - 0.5) < 0.02 and g["bfs_hop_distance"] <= 10


def weighted_loss_fn(params, batch):
    (a, b, c, d, e, f, g, y, w) = batch
    pred = base.forward_batch(params, a, b, c, d, e, f, g)
    return jnp.mean(w * jnp.abs(pred - y)), pred


def run(name, train_idx, test_idx, epochs=100, chunk_epochs=25, batch_size=64, lr=3e-3, seed=0):
    ckpt_path = f"quantumbridge_data/entry058_cc_{name}_ckpt.json"
    opt = optax.adam(lr)

    weights_all = np.array([FC_WEIGHT if is_floor_collapse(i) else 1.0 for i in range(len(base.raw))],
                           dtype=np.float32)

    if os.path.exists(ckpt_path):
        ck = json.load(open(ckpt_path))
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
    for epoch in range(done_epochs, target):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (base.node_feat_norm[idx], base.node_mask_all[idx], base.src_all[idx], base.dst_all[idx],
                    base.edge_feat_norm[idx], base.edge_mask_all[idx], base.glob_all_n[idx], base.y_all[idx],
                    weights_all[idx])
            (l, grads) = jax.value_and_grad(lambda p: weighted_loss_fn(p, batch)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    json.dump({"params": params_json, "epoch": target}, open(ckpt_path, "w"))
    if target < epochs:
        print(f"[{name}] checkpoint at epoch {target}/{epochs} -- rerun to continue")
        return None

    test_batch = (base.node_feat_norm[test_idx], base.node_mask_all[test_idx], base.src_all[test_idx],
                 base.dst_all[test_idx], base.edge_feat_norm[test_idx], base.edge_mask_all[test_idx],
                 base.glob_all_n[test_idx], base.y_all[test_idx], weights_all[test_idx])
    _, test_pred = weighted_loss_fn(params, test_batch)
    test_pred = np.array(test_pred)
    mae = mean_absolute_error(base.y_all[test_idx], test_pred)
    r2 = r2_score(base.y_all[test_idx], test_pred)
    is_fc = np.array([is_floor_collapse(i) for i in test_idx])
    mae_fc = mean_absolute_error(base.y_all[test_idx][is_fc], test_pred[is_fc]) if is_fc.sum() > 0 else None
    result = {"name": name, "mae": float(mae), "r2": float(r2),
             "mae_fc": float(mae_fc) if mae_fc is not None else None,
             "n_test": len(test_idx), "n_fc": int(is_fc.sum())}
    print(f"[{name}] DONE: MAE={mae*100:.2f} R2={r2:.3f} "
          f"fc_MAE={(mae_fc*100 if mae_fc is not None else -1):.2f} (n_fc={int(is_fc.sum())})")
    try:
        os.remove(ckpt_path)
    except PermissionError:
        pass
    return result


if __name__ == "__main__":
    import sys
    direction = sys.argv[1] if len(sys.argv) > 1 else "kyiv_to_sherbrooke"
    idx_all = np.arange(len(base.raw))
    if direction == "kyiv_to_sherbrooke":
        train_idx = idx_all[base.chip_all == "kyiv"]; test_idx = idx_all[base.chip_all == "sherbrooke"]
    else:
        train_idx = idx_all[base.chip_all == "sherbrooke"]; test_idx = idx_all[base.chip_all == "kyiv"]

    RESULTS_PATH = "quantumbridge_data/entry058_cross_chip_results.json"
    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
    r = run(direction, train_idx, test_idx)
    if r:
        results[direction] = r
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
