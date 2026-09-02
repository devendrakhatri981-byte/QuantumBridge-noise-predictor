"""
QuantumBridge — Entry 065: the combined three-chip "generalist" model.

This is the experiment the user asked about directly: instead of testing
strict leave-one-chip-out transfer (Entries 060-064, train on some chips,
evaluate cold on a chip never seen), pool ALL circuits from Kyiv,
Sherbrooke, and Brisbane into one training set, with a random stratified
80/20 split so every chip contributes to both train and test. This tests
a different, complementary question: does giving the model a taste of
every chip's noise behavior make it a better generalist, even though it
can no longer tell us anything about a truly novel fourth chip?

Reuses the exact same 80/20 split (KFold seed=42, fold 0) that produced
entry063_gnn_results.json's same-chip baseline (MAE=1.17, R2=0.973) --
this script just keeps the trained weights (not thrown away) and adds a
per-chip breakdown of the held-out test set, which the aggregate number
alone doesn't show.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

import entry063_train_gnn as base

CKPT_PATH = "quantumbridge_data/entry065_combined_ckpt.json"
DEPLOY_PATH = "quantumbridge_data/entry065_combined_params.json"
RESULTS_PATH = "quantumbridge_data/entry065_combined_results.json"
EPOCHS = 100


def train(train_idx, test_idx, seed=0, epochs=EPOCHS, batch_size=64, lr=3e-3, chunk_epochs=40):
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
    for epoch in range(done_epochs, target):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (base.node_feat_norm[idx], base.node_mask_all[idx], base.src_all[idx], base.dst_all[idx],
                    base.edge_feat_norm[idx], base.edge_mask_all[idx], base.glob_all_n[idx], base.y_all[idx],
                    base.weights_all[idx])
            (l, grads) = jax.value_and_grad(lambda p: base.weighted_loss_fn(p, batch)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    tmp = CKPT_PATH + ".tmp"
    json.dump({"params": params_json, "epoch": target}, open(tmp, "w"))
    os.replace(tmp, CKPT_PATH)
    if target < epochs:
        print(f"checkpoint at epoch {target}/{epochs} -- rerun to continue")
        return None

    test_batch = (base.node_feat_norm[test_idx], base.node_mask_all[test_idx], base.src_all[test_idx],
                 base.dst_all[test_idx], base.edge_feat_norm[test_idx], base.edge_mask_all[test_idx],
                 base.glob_all_n[test_idx], base.y_all[test_idx], base.weights_all[test_idx])
    _, test_pred = base.weighted_loss_fn(params, test_batch)
    test_pred = np.array(test_pred)

    # keep the trained weights permanently -- this is the deployable
    # combined model, not a throwaway CV fold
    tmp = DEPLOY_PATH + ".tmp"
    json.dump(params_json, open(tmp, "w"))
    os.replace(tmp, DEPLOY_PATH)
    return test_pred


if __name__ == "__main__":
    idx_all = np.arange(len(base.raw))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = list(kf.split(idx_all))[0]  # same split as entry063's baseline

    test_pred = train(train_idx, test_idx)
    if test_pred is None:
        raise SystemExit(0)

    y_test = base.y_all[test_idx]
    chip_test = base.chip_all[test_idx]
    is_fc_test = np.array([base.is_floor_collapse(base.raw[i]) for i in test_idx])

    overall_mae = mean_absolute_error(y_test, test_pred)
    overall_r2 = r2_score(y_test, test_pred)
    print(f"\n=== Combined 3-chip model, held-out test (n={len(test_idx)}) ===")
    print(f"Overall: MAE={overall_mae*100:.2f} R2={overall_r2:.3f}")

    per_chip = {}
    for chip in ("kyiv", "sherbrooke", "brisbane"):
        m = chip_test == chip
        if m.sum() == 0:
            continue
        mae = mean_absolute_error(y_test[m], test_pred[m])
        r2 = r2_score(y_test[m], test_pred[m]) if m.sum() > 1 else None
        fc_m = m & is_fc_test
        mae_fc = mean_absolute_error(y_test[fc_m], test_pred[fc_m]) if fc_m.sum() > 0 else None
        per_chip[chip] = {"n": int(m.sum()), "mae": float(mae),
                          "r2": float(r2) if r2 is not None else None,
                          "n_fc": int(fc_m.sum()),
                          "mae_fc": float(mae_fc) if mae_fc is not None else None}
        print(f"  {chip:>10}: n={m.sum():>4} MAE={mae*100:.2f} "
              f"R2={(r2 if r2 is not None else float('nan')):.3f} "
              f"fc_MAE={(mae_fc*100 if mae_fc is not None else -1):.2f} (n_fc={fc_m.sum()})")

    result = {"overall_mae": float(overall_mae), "overall_r2": float(overall_r2),
             "n_test": len(test_idx), "per_chip": per_chip}
    tmp = RESULTS_PATH + ".tmp"
    json.dump(result, open(tmp, "w"), indent=2)
    os.replace(tmp, RESULTS_PATH)
    print(f"\nsaved -> {RESULTS_PATH}")
    print(f"deployable weights -> {DEPLOY_PATH}")
