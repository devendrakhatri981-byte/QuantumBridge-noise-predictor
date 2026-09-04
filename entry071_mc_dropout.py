"""
QuantumBridge — Entry 057: uncertainty quantification via MC-Dropout.

Cheaper alternative to a 5-model ensemble (one training run instead of
five): dropout (rate=0.15) is applied after node embedding and after each
message-passing round, active during training (standard regularization)
AND kept active at inference time. At test time, T=20 stochastic forward
passes are run per circuit; their mean is the point prediction and their
std is the uncertainty estimate (Gal & Ghahramani, 2016 MC-Dropout).

Reuses Entry 056's data pipeline, architecture, per-chip normalization
(Entry 048) and floor-collapse up-weighting (Entry 051) unchanged --
only dropout is added, so any change in behavior is attributable to that.

Validates uncertainty two ways:
  1. Calibration: does higher predicted std correlate with higher actual
     error on the same-chip test set?
  2. Distribution-shift sensitivity: does mean predicted std widen when
     evaluated on the *unseen* chip (cross-chip cold transfer) vs the
     trained chip -- i.e. does the model know when it's out of its depth?
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import optax

import entry068_train_gnn as base

DROPOUT_RATE = 0.15
T_SAMPLES = 20


def dropout(x, key, mask=None):
    keep = jax.random.bernoulli(key, 1.0 - DROPOUT_RATE, x.shape)
    out = jnp.where(keep, x / (1.0 - DROPOUT_RATE), 0.0)
    if mask is not None:
        out = out * mask[:, None]
    return out


def forward_single_stochastic(params, node_feat, node_mask, src, dst, edge_feat, edge_mask, glob, key):
    k1, k2, k3 = jax.random.split(key, 3)
    h = jax.nn.relu(base.apply_lin(params["node_embed"], node_feat)) * node_mask[:, None]
    h = dropout(h, k1, node_mask)
    h = base.mp_round(params["msg"], params["update"], h, node_mask, src, dst, edge_feat, edge_mask)
    h = dropout(h, k2, node_mask)
    h = base.mp_round(params["msg2"], params["update2"], h, node_mask, src, dst, edge_feat, edge_mask)
    h = dropout(h, k3, node_mask)
    pooled = (h * node_mask[:, None]).sum(axis=0) / (node_mask.sum() + 1e-6)
    x = jnp.concatenate([pooled, glob], axis=-1)
    x = jax.nn.relu(base.apply_lin(params["readout1"], x))
    out = base.apply_lin(params["readout2"], x)[0]
    return jax.nn.sigmoid(out)


forward_batch_stochastic = jax.vmap(forward_single_stochastic, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))


def weighted_loss_fn(params, batch, key):
    (a, b, c, d, e, f, g, y, w) = batch
    keys = jax.random.split(key, a.shape[0])
    pred = forward_batch_stochastic(params, a, b, c, d, e, f, g, keys)
    return jnp.mean(w * jnp.abs(pred - y)), pred


def predict_with_uncertainty(params, idx, master_key, t=T_SAMPLES):
    """Return (mean_pred, std_pred) over t stochastic forward passes."""
    a = base.node_feat_norm[idx]; b = base.node_mask_all[idx]
    c = base.src_all[idx]; d = base.dst_all[idx]
    e = base.edge_feat_norm[idx]; f = base.edge_mask_all[idx]; g = base.glob_all_n[idx]
    samples = []
    keys = jax.random.split(master_key, t)
    for tk in keys:
        subkeys = jax.random.split(tk, a.shape[0])
        pred = forward_batch_stochastic(params, a, b, c, d, e, f, g, subkeys)
        samples.append(np.array(pred))
    samples = np.stack(samples, axis=0)  # (t, n)
    return samples.mean(axis=0), samples.std(axis=0)


CKPT_PATH = "quantumbridge_data/entry071_mcdropout_ckpt.json"
RESULTS_PATH = "quantumbridge_data/entry071_mcdropout_results.json"
EPOCHS = 100


def train(train_idx, epochs=EPOCHS, batch_size=64, lr=3e-3, chunk_epochs=25, seed=0, ckpt_path=CKPT_PATH):
    import os
    opt = optax.adam(lr)
    if os.path.exists(ckpt_path):
        ck = json.load(open(ckpt_path))
        params = {k: {"W": jnp.array(v["W"]), "b": jnp.array(v["b"])} for k, v in ck["params"].items()}
        opt_state = opt.init(params)
        done_epochs = ck["epoch"]
        rng_seed = ck["rng_seed"]
    else:
        params = base.init_params(jax.random.PRNGKey(seed))
        opt_state = opt.init(params)
        done_epochs = 0
        rng_seed = seed

    rng = np.random.RandomState(rng_seed + done_epochs)
    target = min(epochs, done_epochs + chunk_epochs)
    n_train = len(train_idx)
    step_key = jax.random.PRNGKey(rng_seed + 10000 + done_epochs)
    for epoch in range(done_epochs, target):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (base.node_feat_norm[idx], base.node_mask_all[idx], base.src_all[idx], base.dst_all[idx],
                    base.edge_feat_norm[idx], base.edge_mask_all[idx], base.glob_all_n[idx], base.y_all[idx],
                    base.weights_all[idx])
            step_key, sub = jax.random.split(step_key)
            (l, grads) = jax.value_and_grad(lambda p: weighted_loss_fn(p, batch, sub)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    json.dump({"params": params_json, "epoch": target, "rng_seed": rng_seed}, open(ckpt_path, "w"))
    if target < epochs:
        print(f"checkpoint at epoch {target}/{epochs} -- rerun to continue")
        return None
    return params


if __name__ == "__main__":
    import os, sys
    idx_all = np.arange(len(base.raw))
    is_fc = np.array([base.is_floor_collapse(g) for g in base.raw])

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = list(kf.split(idx_all))[0]

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    direction = sys.argv[2] if len(sys.argv) > 2 else "kyiv_to_sherbrooke"
    train_chip, cold_chip = ("kyiv", "sherbrooke") if direction == "kyiv_to_sherbrooke" else ("sherbrooke", "kyiv")
    suffix = "" if direction == "kyiv_to_sherbrooke" else "_rev"
    CC_CKPT = f"quantumbridge_data/entry071_mcdropout_cc{suffix}_ckpt.json"
    CC_RESULTS = f"quantumbridge_data/entry071_mcdropout_cc{suffix}_results.json"

    if mode == "train":
        params = train(train_idx)
        if params is None:
            sys.exit(0)
        print("training complete, run `python3 entry057_mc_dropout.py eval` next")

    elif mode == "train_cc":
        train_chip_idx = idx_all[base.chip_all == train_chip]
        rng = np.random.RandomState(7)
        perm = rng.permutation(len(train_chip_idx))
        split = int(len(train_chip_idx) * 0.8)
        cc_train_idx = train_chip_idx[perm[:split]]
        params = train(cc_train_idx, ckpt_path=CC_CKPT, seed=7)
        if params is None:
            sys.exit(0)
        print(f"cross-chip training complete ({direction}), run "
              f"`python3 entry057_mc_dropout.py eval_cc {direction}` next")

    elif mode == "eval_cc":
        ck = json.load(open(CC_CKPT))
        params = {k: {"W": jnp.array(v["W"]), "b": jnp.array(v["b"])} for k, v in ck["params"].items()}
        assert ck["epoch"] >= EPOCHS, f"training not finished ({ck['epoch']}/{EPOCHS})"

        train_chip_idx = idx_all[base.chip_all == train_chip]
        rng = np.random.RandomState(7)
        perm = rng.permutation(len(train_chip_idx))
        split = int(len(train_chip_idx) * 0.8)
        cc_heldout_warm = train_chip_idx[perm[split:]]
        cold_idx = idx_all[base.chip_all == cold_chip]

        mean_warm, std_warm = predict_with_uncertainty(params, cc_heldout_warm, jax.random.PRNGKey(11))
        y_warm = base.y_all[cc_heldout_warm]
        err_warm = np.abs(mean_warm - y_warm)
        mae_warm = mean_absolute_error(y_warm, mean_warm)
        corr_warm = float(np.corrcoef(std_warm, err_warm)[0, 1])
        print(f"[warm: held-out {train_chip}] MAE={mae_warm*100:.2f} mean std={std_warm.mean()*100:.2f} pts, "
              f"corr(std,|err|)={corr_warm:.3f}")

        mean_cold, std_cold = predict_with_uncertainty(params, cold_idx, jax.random.PRNGKey(12))
        y_cold = base.y_all[cold_idx]
        err_cold = np.abs(mean_cold - y_cold)
        mae_cold = mean_absolute_error(y_cold, mean_cold)
        corr_cold = float(np.corrcoef(std_cold, err_cold)[0, 1])
        ratio = std_cold.mean() / std_warm.mean()
        print(f"[cold: ALL {cold_chip}, truly unseen] MAE={mae_cold*100:.2f} "
              f"mean std={std_cold.mean()*100:.2f} pts, corr(std,|err|)={corr_cold:.3f}")
        print(f"[widening check] cold_std/warm_std ratio = {ratio:.2f}x")
        result = {
            "warm": {"chip": f"{train_chip}_heldout", "mae": float(mae_warm), "mean_std": float(std_warm.mean()),
                    "corr_std_error": corr_warm},
            "cold": {"chip": f"{cold_chip}_all_unseen", "mae": float(mae_cold), "mean_std": float(std_cold.mean()),
                    "corr_std_error": corr_cold},
            "widening_ratio": float(ratio)}
        json.dump(result, open(CC_RESULTS, "w"), indent=2)
        print(f"saved -> {CC_RESULTS}")

    elif mode == "eval_old":
        ck = json.load(open(CKPT_PATH))
        params = {k: {"W": jnp.array(v["W"]), "b": jnp.array(v["b"])} for k, v in ck["params"].items()}
        assert ck["epoch"] >= EPOCHS, f"training not finished ({ck['epoch']}/{EPOCHS})"

        # 1. same-chip test set: calibration check
        mean_pred, std_pred = predict_with_uncertainty(params, test_idx, jax.random.PRNGKey(1))
        y_test = base.y_all[test_idx]
        err = np.abs(mean_pred - y_test)
        mae = mean_absolute_error(y_test, mean_pred)
        r2 = r2_score(y_test, mean_pred)
        corr = float(np.corrcoef(std_pred, err)[0, 1])
        mask_fc = is_fc[test_idx]
        mae_fc = mean_absolute_error(y_test[mask_fc], mean_pred[mask_fc]) if mask_fc.sum() > 0 else None
        print(f"[same-chip test] MAE={mae*100:.2f} R2={r2:.3f} fc_MAE={(mae_fc*100 if mae_fc else -1):.2f}")
        print(f"[same-chip test] mean predicted std={std_pred.mean()*100:.2f} pts, "
              f"corr(std, |error|)={corr:.3f}")

        # 2. cross-chip cold transfer: does uncertainty widen on the unseen chip?
        chip_train = base.chip_all[train_idx[0]]
        cold_chip = "sherbrooke" if chip_train == "kyiv" else "kyiv"
        cold_idx = idx_all[base.chip_all == cold_chip]
        mean_cold, std_cold = predict_with_uncertainty(params, cold_idx, jax.random.PRNGKey(2))
        y_cold = base.y_all[cold_idx]
        err_cold = np.abs(mean_cold - y_cold)
        mae_cold = mean_absolute_error(y_cold, mean_cold)
        corr_cold = float(np.corrcoef(std_cold, err_cold)[0, 1])
        print(f"[cold chip={cold_chip}] MAE={mae_cold*100:.2f} "
              f"mean predicted std={std_cold.mean()*100:.2f} pts, corr(std, |error|)={corr_cold:.3f}")
        print(f"[widening check] cold_std/warm_std ratio = {std_cold.mean()/std_pred.mean():.2f}x")

        result = {
            "train_chip": chip_train, "cold_chip": cold_chip,
            "same_chip": {"mae": float(mae), "r2": float(r2), "mae_fc": float(mae_fc) if mae_fc else None,
                         "mean_std": float(std_pred.mean()), "corr_std_error": corr},
            "cold_chip_result": {"mae": float(mae_cold), "mean_std": float(std_cold.mean()),
                                 "corr_std_error": corr_cold},
            "widening_ratio": float(std_cold.mean() / std_pred.mean()),
        }
        json.dump(result, open(RESULTS_PATH, "w"), indent=2)
        print(f"saved -> {RESULTS_PATH}")
