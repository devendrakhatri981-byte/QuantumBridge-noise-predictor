"""
QuantumBridge — Entry 044b: GNN retrained on the feature-engineered graphs.

Same architecture as Entry 041/042 (2 message-passing rounds, H=16,
JAX+Optax), same 2,288 circuits as Entry 043 -- only the per-node/per-edge
feature set changed (Entry 044a): 6 node features (T1, T2, readout,
T1/T2 ratio, is_control, is_target) instead of 3, and 6 edge features
(gate error, duration, route position, is_final_gate, plus the original
src/dst indices) instead of 2. Comparing against Entry 043's result
(same circuits, old features) isolates the effect of the new features
from the effect of more data -- Entry 043 already showed data volume
alone was ambiguous at this scale, so any improvement here is more
likely attributable to the features actually being useful signal.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

MAX_N, MAX_E = 40, 40
IN_PATH = "quantumbridge_data/entry044_graph_dataset.json"
N_NODE_FEATS = 6
N_EDGE_FEATS = 4  # gate error, duration, route position, is_final

raw = json.load(open(IN_PATH))


def pad_graph(g):
    n = g["n_nodes"]
    node_feat = np.zeros((MAX_N, N_NODE_FEATS), dtype=np.float32)
    for i, f in enumerate(g["nodes"]):
        node_feat[i] = f
    node_mask = np.zeros(MAX_N, dtype=np.float32)
    node_mask[:n] = 1.0

    e = len(g["edges"])
    src = np.zeros(MAX_E, dtype=np.int32)
    dst = np.zeros(MAX_E, dtype=np.int32)
    edge_feat = np.zeros((MAX_E, N_EDGE_FEATS), dtype=np.float32)
    for i, edge in enumerate(g["edges"]):
        u, v, err, dur, pos, is_final = edge
        src[i] = u; dst[i] = v
        edge_feat[i] = [err, dur, pos, is_final]
    edge_mask = np.zeros(MAX_E, dtype=np.float32)
    edge_mask[:e] = 1.0

    glob = np.array([
        1.0 if g["chip"] == "kyiv" else 0.0,
        1.0 if g["chip"] == "sherbrooke" else 0.0,
        1.0 if g["kind"] == "bell" else 0.0,
        1.0 if g["kind"] == "ghz" else 0.0,
        float(g["bfs_hop_distance"]),
        float(g["v4_1_prediction"]),
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

node_mu = node_feat_all[node_mask_all.astype(bool)].mean(axis=0)
node_sd = node_feat_all[node_mask_all.astype(bool)].std(axis=0) + 1e-6
edge_mu = edge_feat_all[edge_mask_all.astype(bool)].mean(axis=0)
edge_sd = edge_feat_all[edge_mask_all.astype(bool)].std(axis=0) + 1e-6
glob_mu = glob_all.mean(axis=0); glob_sd = glob_all.std(axis=0) + 1e-6

node_feat_all = (node_feat_all - node_mu) / node_sd
edge_feat_all = (edge_feat_all - edge_mu) / edge_sd
glob_all_n = (glob_all - glob_mu) / glob_sd

H = 16


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
        "readout1": lin(ks[5], H + 6, H),
        "readout2": lin(ks[6], H, 1),
    }


def apply_lin(p, x):
    return x @ p["W"] + p["b"]


def mp_round(params_msg, params_upd, h, node_mask, src, dst, edge_feat, edge_mask):
    h_src = h[src]
    h_dst = h[dst]
    msg_in = jnp.concatenate([h_src, h_dst, edge_feat], axis=-1)
    msg = jax.nn.relu(apply_lin(params_msg, msg_in)) * edge_mask[:, None]

    agg = jnp.zeros((MAX_N, H)).at[dst].add(msg)
    deg = jnp.zeros((MAX_N,)).at[dst].add(edge_mask) + 1e-6
    agg = agg / deg[:, None]

    upd_in = jnp.concatenate([h, agg], axis=-1)
    h_new = jax.nn.relu(apply_lin(params_upd, upd_in))
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


def loss_fn(params, batch):
    (nf, nm, s, d, ef, em, gl, y) = batch
    pred = forward_batch(params, nf, nm, s, d, ef, em, gl)
    return jnp.mean(jnp.abs(pred - y)), pred


def train_fold(train_idx, test_idx, seed=0, epochs=180, batch_size=64, lr=3e-3):
    key = jax.random.PRNGKey(seed)
    params = init_params(key)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    n_train = len(train_idx)
    rng = np.random.RandomState(seed)
    for epoch in range(epochs):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (node_feat_all[idx], node_mask_all[idx], src_all[idx], dst_all[idx],
                    edge_feat_all[idx], edge_mask_all[idx], glob_all_n[idx], y_all[idx])
            (l, grads) = jax.value_and_grad(lambda p: loss_fn(p, batch)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    test_batch = (node_feat_all[test_idx], node_mask_all[test_idx], src_all[test_idx],
                 dst_all[test_idx], edge_feat_all[test_idx], edge_mask_all[test_idx],
                 glob_all_n[test_idx], y_all[test_idx])
    test_loss, test_pred = loss_fn(params, test_batch)
    return float(test_loss), np.array(test_pred)


FOLD_CKPT = "quantumbridge_data/entry044_gnn_folds.json"
RESULTS_PATH = "quantumbridge_data/entry044_gnn_results.json"
EPOCHS = 180

if __name__ == "__main__":
    import os
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    n = len(raw)
    idx_all = np.arange(n)
    is_fc = np.array([abs(g["aer_ground_truth"] - 0.5) < 0.02 and g["bfs_hop_distance"] <= 10
                      for g in raw])
    print(f"n={n}, floor-collapse n={is_fc.sum()}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(idx_all))

    done = json.load(open(FOLD_CKPT)) if os.path.exists(FOLD_CKPT) else {}
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        if str(fold_i) in done:
            print(f"fold {fold_i}: already done, skipping")
            continue
        test_loss, test_pred = train_fold(train_idx, test_idx, seed=fold_i, epochs=EPOCHS)
        mae = mean_absolute_error(y_all[test_idx], test_pred)
        r2 = r2_score(y_all[test_idx], test_pred)
        mask = is_fc[test_idx]
        mae_fc = (mean_absolute_error(y_all[test_idx][mask], test_pred[mask])
                  if mask.sum() > 0 else None)
        done[str(fold_i)] = {"test_idx": test_idx.tolist(), "pred": test_pred.tolist(),
                             "mae": mae, "r2": r2, "mae_fc": mae_fc, "n_fc": int(mask.sum())}
        json.dump(done, open(FOLD_CKPT, "w"))
        print(f"fold {fold_i}: MAE={mae*100:.2f} R2={r2:.3f} "
              f"(n_test={len(test_idx)}, n_fc={mask.sum()}) -- saved checkpoint")
        break

    if len(done) == 5:
        fold_mae = [done[str(i)]["mae"] for i in range(5)]
        fold_r2 = [done[str(i)]["r2"] for i in range(5)]
        fold_mae_fc = [done[str(i)]["mae_fc"] for i in range(5) if done[str(i)]["mae_fc"] is not None]

        print(f"\nGNN CV MAE={np.mean(fold_mae)*100:.2f} R2={np.mean(fold_r2):.3f} "
              f"floor-collapse MAE={np.mean(fold_mae_fc)*100:.2f}")

        v41_mae = mean_absolute_error(y_all, v41_all)
        v41_fc_mae = mean_absolute_error(y_all[is_fc], v41_all[is_fc])
        print(f"v4.1: MAE={v41_mae*100:.2f} floor-collapse MAE={v41_fc_mae*100:.2f}")

        result = {
            "n": n, "n_floor_collapse": int(is_fc.sum()),
            "gnn_cv_mae": float(np.mean(fold_mae)), "gnn_cv_r2": float(np.mean(fold_r2)),
            "gnn_cv_mae_floor_collapse": float(np.mean(fold_mae_fc)),
            "v41_mae": float(v41_mae), "v41_mae_floor_collapse": float(v41_fc_mae),
        }
        json.dump(result, open(RESULTS_PATH, "w"), indent=2)
        print(f"\nSaved to {RESULTS_PATH}")
    else:
        print(f"\n{len(done)}/5 folds done so far -- rerun to continue")
