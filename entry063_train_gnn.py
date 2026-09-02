"""
QuantumBridge — Entry 058: GNN capacity bump #2, 48/64 -> 80/104.

Same architecture and feature set as Entry 045/048 (6 node feats, 4 edge
feats, 2 message-passing rounds, H=16, per-chip normalization + no
chip-identity input from Entry 048's fix, floor-collapse 5x up-weighting
from Entry 051) -- only MAX_N/MAX_E changed, from 48/64 (Entry 045's max:
42 nodes/56 edges) to 80/104 (fits Entry 058's max: 65 nodes/99 edges,
with headroom). Trained on the combined 2,329-circuit dataset.

This is a same-chip 5-fold CV run (not cross-chip) to establish the new
baseline at this larger capacity before repeating the cross-chip checks.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

MAX_N, MAX_E = 90, 130
IN_PATH = "quantumbridge_data/entry063_graph_dataset.json"
N_NODE_FEATS, N_EDGE_FEATS = 6, 4
H = 16
FC_WEIGHT = 5.0

raw = json.load(open(IN_PATH))


def is_floor_collapse(g):
    return abs(g["aer_ground_truth"] - 0.5) < 0.02 and g["bfs_hop_distance"] <= 10


def pad_graph(g):
    n = g["n_nodes"]
    node_feat = np.zeros((MAX_N, N_NODE_FEATS), dtype=np.float32)
    for i, f in enumerate(g["nodes"]):
        node_feat[i] = f
    node_mask = np.zeros(MAX_N, dtype=np.float32); node_mask[:n] = 1.0

    e = len(g["edges"])
    src = np.zeros(MAX_E, dtype=np.int32); dst = np.zeros(MAX_E, dtype=np.int32)
    edge_feat = np.zeros((MAX_E, N_EDGE_FEATS), dtype=np.float32)
    for i, edge in enumerate(g["edges"]):
        u, v, err, dur, pos, is_final = edge
        src[i] = u; dst[i] = v
        edge_feat[i] = [err, dur, pos, is_final]
    edge_mask = np.zeros(MAX_E, dtype=np.float32); edge_mask[:e] = 1.0

    glob = np.array([
        1.0 if g["kind"] == "bell" else 0.0,
        1.0 if g["kind"] == "ghz" else 0.0,
        float(g["bfs_hop_distance"]), float(g["v4_1_prediction"]),
    ], dtype=np.float32)
    return node_feat, node_mask, src, dst, edge_feat, edge_mask, glob


all_padded = [pad_graph(g) for g in raw]
node_feat_all = np.stack([p[0] for p in all_padded])
node_mask_all = np.stack([p[1] for p in all_padded])
src_all = np.stack([p[2] for p in all_padded])
dst_all = np.stack([p[3] for p in all_padded])
edge_feat_all = np.stack([p[4] for p in all_padded])
edge_mask_all = np.stack([p[5] for p in all_padded])
glob_all = np.stack([p[6] for p in all_padded])
y_all = np.array([g["aer_ground_truth"] for g in raw], dtype=np.float32)
v41_all = np.array([g["v4_1_prediction"] for g in raw], dtype=np.float32)
chip_all = np.array([g["chip"] for g in raw])
weights_all = np.array([FC_WEIGHT if is_floor_collapse(g) else 1.0 for g in raw], dtype=np.float32)
N_GLOB = glob_all.shape[1]

# per-chip normalization (Entry 048 fix)
node_stats, edge_stats = {}, {}
for chip in ("kyiv", "sherbrooke", "brisbane"):
    m = chip_all == chip
    nm = node_mask_all[m].astype(bool)
    em_ = edge_mask_all[m].astype(bool)
    node_stats[chip] = (node_feat_all[m][nm].mean(axis=0), node_feat_all[m][nm].std(axis=0) + 1e-6)
    edge_stats[chip] = (edge_feat_all[m][em_].mean(axis=0), edge_feat_all[m][em_].std(axis=0) + 1e-6)
glob_mu, glob_sd = glob_all.mean(axis=0), glob_all.std(axis=0) + 1e-6

node_feat_norm = np.zeros_like(node_feat_all)
edge_feat_norm = np.zeros_like(edge_feat_all)
for chip in ("kyiv", "sherbrooke", "brisbane"):
    m = chip_all == chip
    mu, sd = node_stats[chip]; node_feat_norm[m] = (node_feat_all[m] - mu) / sd
    mu, sd = edge_stats[chip]; edge_feat_norm[m] = (edge_feat_all[m] - mu) / sd
glob_all_n = (glob_all - glob_mu) / glob_sd


def init_params(key):
    ks = jax.random.split(key, 12)
    def lin(k, i, o):
        return {"W": jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i), "b": jnp.zeros(o)}
    return {
        "node_embed": lin(ks[0], N_NODE_FEATS, H),
        "msg": lin(ks[1], 2 * H + N_EDGE_FEATS, H),
        "update": lin(ks[2], 2 * H, H),
        "msg2": lin(ks[3], 2 * H + N_EDGE_FEATS, H),
        "update2": lin(ks[4], 2 * H, H),
        "readout1": lin(ks[5], H + N_GLOB, H),
        "readout2": lin(ks[6], H, 1),
    }


def apply_lin(p, x): return x @ p["W"] + p["b"]


def mp_round(pm, pu, h, node_mask, src, dst, edge_feat, edge_mask):
    h_src, h_dst = h[src], h[dst]
    msg_in = jnp.concatenate([h_src, h_dst, edge_feat], axis=-1)
    msg = jax.nn.relu(apply_lin(pm, msg_in)) * edge_mask[:, None]
    agg = jnp.zeros((MAX_N, H)).at[dst].add(msg)
    deg = jnp.zeros((MAX_N,)).at[dst].add(edge_mask) + 1e-6
    agg = agg / deg[:, None]
    upd_in = jnp.concatenate([h, agg], axis=-1)
    h_new = jax.nn.relu(apply_lin(pu, upd_in))
    return h_new * node_mask[:, None]


def forward_single(params, node_feat, node_mask, src, dst, edge_feat, edge_mask, glob):
    h = jax.nn.relu(apply_lin(params["node_embed"], node_feat)) * node_mask[:, None]
    h = mp_round(params["msg"], params["update"], h, node_mask, src, dst, edge_feat, edge_mask)
    h = mp_round(params["msg2"], params["update2"], h, node_mask, src, dst, edge_feat, edge_mask)
    pooled = (h * node_mask[:, None]).sum(axis=0) / (node_mask.sum() + 1e-6)
    x = jnp.concatenate([pooled, glob], axis=-1)
    x = jax.nn.relu(apply_lin(params["readout1"], x))
    out = apply_lin(params["readout2"], x)[0]
    return jax.nn.sigmoid(out)


forward_batch = jax.vmap(forward_single, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))


def weighted_loss_fn(params, batch):
    (a, b, c, d, e, f, g, y, w) = batch
    pred = forward_batch(params, a, b, c, d, e, f, g)
    return jnp.mean(w * jnp.abs(pred - y)), pred


FOLD_CKPT = "quantumbridge_data/entry063_gnn_folds.json"
RESULTS_PATH = "quantumbridge_data/entry063_gnn_results.json"
EPOCHS = 100


def train_fold(train_idx, test_idx, seed=0, epochs=EPOCHS, batch_size=64, lr=3e-3, chunk_epochs=25):
    import os
    ckpt_path = f"quantumbridge_data/entry063_fold_tmp_ckpt.json"
    opt = optax.adam(lr)
    if os.path.exists(ckpt_path):
        ck = json.load(open(ckpt_path))
        params = {k: {"W": jnp.array(v["W"]), "b": jnp.array(v["b"])} for k, v in ck["params"].items()}
        opt_state = opt.init(params)
        done_epochs = ck["epoch"]
    else:
        params = init_params(jax.random.PRNGKey(seed))
        opt_state = opt.init(params)
        done_epochs = 0

    rng = np.random.RandomState(seed + done_epochs)
    target = min(epochs, done_epochs + chunk_epochs)
    n_train = len(train_idx)
    for epoch in range(done_epochs, target):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (node_feat_norm[idx], node_mask_all[idx], src_all[idx], dst_all[idx],
                    edge_feat_norm[idx], edge_mask_all[idx], glob_all_n[idx], y_all[idx], weights_all[idx])
            (l, grads) = jax.value_and_grad(lambda p: weighted_loss_fn(p, batch)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    json.dump({"params": params_json, "epoch": target}, open(ckpt_path, "w"))
    if target < epochs:
        return None, None, target

    test_batch = (node_feat_norm[test_idx], node_mask_all[test_idx], src_all[test_idx], dst_all[test_idx],
                 edge_feat_norm[test_idx], edge_mask_all[test_idx], glob_all_n[test_idx], y_all[test_idx],
                 weights_all[test_idx])
    _, test_pred = weighted_loss_fn(params, test_batch)
    try:
        os.remove(ckpt_path)
    except PermissionError:
        pass
    return float(0), np.array(test_pred), target


if __name__ == "__main__":
    import os
    n = len(raw)
    idx_all = np.arange(n)
    is_fc = np.array([is_floor_collapse(g) for g in raw])
    print(f"n={n}, floor-collapse n={is_fc.sum()}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(idx_all))[:1]  # single 80/20 split (matches Entry 047/048/051 precedent)

    done = json.load(open(FOLD_CKPT)) if os.path.exists(FOLD_CKPT) else {}
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        if str(fold_i) in done:
            continue
        _, test_pred, target = train_fold(train_idx, test_idx, seed=fold_i, chunk_epochs=40)
        if test_pred is None:
            print(f"fold {fold_i}: checkpoint at epoch {target}/{EPOCHS} -- rerun to continue")
            break
        mae = mean_absolute_error(y_all[test_idx], test_pred)
        r2 = r2_score(y_all[test_idx], test_pred)
        mask = is_fc[test_idx]
        mae_fc = mean_absolute_error(y_all[test_idx][mask], test_pred[mask]) if mask.sum() > 0 else None
        done[str(fold_i)] = {"mae": mae, "r2": r2, "mae_fc": mae_fc, "n_fc": int(mask.sum())}
        json.dump(done, open(FOLD_CKPT, "w"))
        print(f"fold {fold_i}: MAE={mae*100:.2f} R2={r2:.3f} fc_MAE={(mae_fc*100 if mae_fc else -1):.2f} "
              f"(n_test={len(test_idx)}, n_fc={mask.sum()})")

    if len(done) == 1:
        fold_mae = [done[str(i)]["mae"] for i in range(1)]
        fold_r2 = [done[str(i)]["r2"] for i in range(1)]
        fold_mae_fc = [done[str(i)]["mae_fc"] for i in range(1) if done[str(i)]["mae_fc"] is not None]
        v41_mae = mean_absolute_error(y_all, v41_all)
        v41_fc_mae = mean_absolute_error(y_all[is_fc], v41_all[is_fc])
        print(f"\nGNN CV MAE={np.mean(fold_mae)*100:.2f} R2={np.mean(fold_r2):.3f} "
              f"floor-collapse MAE={np.mean(fold_mae_fc)*100:.2f}")
        print(f"v4.1: MAE={v41_mae*100:.2f} floor-collapse MAE={v41_fc_mae*100:.2f}")
        result = {"n": n, "n_floor_collapse": int(is_fc.sum()),
                 "max_n_nodes": int(max(g["n_nodes"] for g in raw)),
                 "max_n_edges": int(max(len(g["edges"]) for g in raw)),
                 "gnn_cv_mae": float(np.mean(fold_mae)), "gnn_cv_r2": float(np.mean(fold_r2)),
                 "gnn_cv_mae_floor_collapse": float(np.mean(fold_mae_fc)),
                 "v41_mae": float(v41_mae), "v41_mae_floor_collapse": float(v41_fc_mae)}
        json.dump(result, open(RESULTS_PATH, "w"), indent=2)
        print(f"saved -> {RESULTS_PATH}")
    else:
        print(f"{len(done)}/5 folds done")
