"""
QuantumBridge — Entry 041b: the first GNN.

ARCHITECTURE
------------
Each circuit's real physical route (Entry 041a) is a small graph: nodes
are the physical qubits actually touched by the transpiled circuit (2-37
of them), node features are that qubit's own T1/T2/readout, edges are the
real 2-qubit gates the router issued (SWAPs and the final entangling
gate), edge features are that edge's gate error and duration. Two rounds
of message passing let information from qubits several hops away
propagate into a node's embedding -- exactly the kind of route-shape
reasoning the tabular model (Entry 037) could only approximate through
hand-picked scalar summaries like "worst T1 on the BFS path."

Graphs are padded to a fixed 40 nodes / 40 edges (the largest real graph
has 37 of each) with boolean masks, so the whole batch can be vmapped --
this is a small enough dataset (1,376 graphs) that a plain padded/masked
implementation in JAX is simpler and fast enough without a dedicated
graph library.

v4.1's own prediction is included as an auxiliary input to the final
readout MLP, same as Entry 037's tabular baseline, so the comparison is
fair: both models get the same "cheat" input, and the question is whether
the graph structure adds anything beyond it.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

MAX_N, MAX_E = 40, 40
IN_PATH = "quantumbridge_data/entry041_graph_dataset.json"
FOLD_CKPT_OVERRIDE = None
RESULTS_PATH_OVERRIDE = None

raw = json.load(open(IN_PATH))


def pad_graph(g):
    n = g["n_nodes"]
    node_feat = np.zeros((MAX_N, 3), dtype=np.float32)
    for i, f in enumerate(g["nodes"]):
        node_feat[i] = f
    node_mask = np.zeros(MAX_N, dtype=np.float32)
    node_mask[:n] = 1.0

    e = len(g["edges"])
    src = np.zeros(MAX_E, dtype=np.int32)
    dst = np.zeros(MAX_E, dtype=np.int32)
    edge_feat = np.zeros((MAX_E, 2), dtype=np.float32)
    for i, (u, v, err, dur) in enumerate(g["edges"]):
        src[i] = u; dst[i] = v
        edge_feat[i] = [err, dur]
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

# normalize node/edge features (zero mean, unit var) using full-dataset stats
node_mu = node_feat_all[node_mask_all.astype(bool)].mean(axis=0)
node_sd = node_feat_all[node_mask_all.astype(bool)].std(axis=0) + 1e-6
edge_mu = edge_feat_all[edge_mask_all.astype(bool)].mean(axis=0)
edge_sd = edge_feat_all[edge_mask_all.astype(bool)].std(axis=0) + 1e-6
glob_mu = glob_all.mean(axis=0); glob_sd = glob_all.std(axis=0) + 1e-6

node_feat_all = (node_feat_all - node_mu) / node_sd
edge_feat_all = (edge_feat_all - edge_mu) / edge_sd
glob_all_n = (glob_all - glob_mu) / glob_sd

H = 16  # hidden dim


def init_params(key):
    ks = jax.random.split(key, 12)
    def lin(k, i, o):
        return {"W": jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i), "b": jnp.zeros(o)}
    return {
        "node_embed": lin(ks[0], 3, H),
        "msg": lin(ks[1], 2 * H + 2, H),
        "update": lin(ks[2], 2 * H, H),
        "msg2": lin(ks[3], 2 * H + 2, H),
        "update2": lin(ks[4], 2 * H, H),
        "readout1": lin(ks[5], H + 6, H),
        "readout2": lin(ks[6], H, 1),
    }


def apply_lin(p, x):
    return x @ p["W"] + p["b"]


def mp_round(params_msg, params_upd, h, node_mask, src, dst, edge_feat, edge_mask):
    h_src = h[src]            # [E, H]
    h_dst = h[dst]
    msg_in = jnp.concatenate([h_src, h_dst, edge_feat], axis=-1)
    msg = jax.nn.relu(apply_lin(params_msg, msg_in)) * edge_mask[:, None]  # [E, H]

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


def train_fold(train_idx, test_idx, seed=0, epochs=250, batch_size=64, lr=3e-3):
    key = jax.random.PRNGKey(seed)
    params = init_params(key)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    grad_fn = jax.jit(jax.value_and_grad(lambda p, b: loss_fn(p, b)[0]))

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


FOLD_CKPT = "quantumbridge_data/entry041_gnn_folds.json"
EPOCHS = 180


def load_fold_ckpt():
    import os
    if os.path.exists(FOLD_CKPT):
        return json.load(open(FOLD_CKPT))
    return {}


if __name__ == "__main__":
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    n = len(raw)
    idx_all = np.arange(n)
    is_fc = np.array([abs(g["aer_ground_truth"] - 0.5) < 0.02 and g["bfs_hop_distance"] <= 10
                      for g in raw])
    print(f"n={n}, floor-collapse n={is_fc.sum()}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(idx_all))

    done = load_fold_ckpt()
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
        break  # one fold per invocation, to fit the tool timeout budget

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
        json.dump(result, open("quantumbridge_data/entry041_gnn_results.json", "w"), indent=2)
        print("\nSaved to quantumbridge_data/entry041_gnn_results.json")
    else:
        print(f"\n{len(done)}/5 folds done so far -- rerun to continue")
