"""
QuantumBridge — Entry 091: first GNN trained on the multi-bell-pair
template (Entry 089/090). Same architecture family as every prior
template (node embed -> 2 message-passing rounds -> masked mean-pool ->
readout MLP -> sigmoid), sized for this template's 7 node / 5 edge
features and much smaller graphs (max 33 nodes / 32 edges vs 90/130 for
single-pair circuits, since the hop cap here is 8 not 26).

Single chip (Kyiv only) and a small dataset (310 circuits) -- this is a
first-pass check of whether the template is learnable at all, not a
production model. A single 80/20 CV split, honest comparison against
Entry 089's product-of-marginals baseline (the "no crosstalk" naive
prediction), following the same "prove it's learnable before scaling"
order used for every prior template (Entry 037 was the same kind of
first pass for single bell-pairs).
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

MAX_N, MAX_E = 40, 40
IN_PATH = "quantumbridge_data/entry090_multibell_graph_dataset.json"
N_NODE_FEATS, N_EDGE_FEATS = 7, 5
H = 16
EPOCHS = 150

raw = json.load(open(IN_PATH))


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
        u, v, err, dur, pos, is_final, pair_id = edge
        src[i] = u; dst[i] = v
        edge_feat[i] = [err, dur, pos, is_final, pair_id]
    edge_mask = np.zeros(MAX_E, dtype=np.float32); edge_mask[:e] = 1.0

    glob = np.array([float(g["n_pairs"]), float(g["product_of_marginals_v41"])], dtype=np.float32)
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
baseline_all = np.array([g["product_of_marginals_v41"] for g in raw], dtype=np.float32)
N_GLOB = glob_all.shape[1]

node_mu, node_sd = node_feat_all[node_mask_all.astype(bool)].mean(axis=0), \
                   node_feat_all[node_mask_all.astype(bool)].std(axis=0) + 1e-6
edge_mu, edge_sd = edge_feat_all[edge_mask_all.astype(bool)].mean(axis=0), \
                   edge_feat_all[edge_mask_all.astype(bool)].std(axis=0) + 1e-6
glob_mu, glob_sd = glob_all.mean(axis=0), glob_all.std(axis=0) + 1e-6

node_feat_norm = (node_feat_all - node_mu) / node_sd
edge_feat_norm = (edge_feat_all - edge_mu) / edge_sd
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


def loss_fn(params, batch):
    (a, b, c, d, e, f, g, y) = batch
    pred = forward_batch(params, a, b, c, d, e, f, g)
    return jnp.mean(jnp.abs(pred - y)), pred


if __name__ == "__main__":
    n = len(raw)
    idx_all = np.arange(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = list(kf.split(idx_all))[0]

    opt = optax.adam(3e-3)
    params = init_params(jax.random.PRNGKey(0))
    opt_state = opt.init(params)
    rng = np.random.RandomState(0)
    batch_size = 32

    for epoch in range(EPOCHS):
        perm = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (node_feat_norm[idx], node_mask_all[idx], src_all[idx], dst_all[idx],
                    edge_feat_norm[idx], edge_mask_all[idx], glob_all_n[idx], y_all[idx])
            (l, grads) = jax.value_and_grad(lambda p: loss_fn(p, batch)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    test_batch = (node_feat_norm[test_idx], node_mask_all[test_idx], src_all[test_idx],
                 dst_all[test_idx], edge_feat_norm[test_idx], edge_mask_all[test_idx],
                 glob_all_n[test_idx], y_all[test_idx])
    _, test_pred = loss_fn(params, test_batch)
    test_pred = np.array(test_pred)
    y_test = y_all[test_idx]
    baseline_test = baseline_all[test_idx]

    gnn_mae = mean_absolute_error(y_test, test_pred)
    gnn_r2 = r2_score(y_test, test_pred)
    base_mae = mean_absolute_error(y_test, baseline_test)
    base_r2 = r2_score(y_test, baseline_test)

    print(f"\n=== Entry 091: first multi-bell-pair model (n={n}, test n={len(test_idx)}) ===")
    print(f"GNN:                    MAE={gnn_mae*100:.2f} R2={gnn_r2:.3f}")
    print(f"product-of-marginals:   MAE={base_mae*100:.2f} R2={base_r2:.3f}")
    print(f"improvement: {(1 - gnn_mae/base_mae)*100:.1f}% lower MAE than the naive-independence baseline")

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    export = {"params": params_json, "node_mu": node_mu.tolist(), "node_sd": node_sd.tolist(),
             "edge_mu": edge_mu.tolist(), "edge_sd": edge_sd.tolist(),
             "glob_mu": glob_mu.tolist(), "glob_sd": glob_sd.tolist(),
             "max_n": MAX_N, "max_e": MAX_E}
    json.dump(export, open("quantumbridge_data/entry091_multibell_params.json", "w"))
    result = {"n": n, "n_test": len(test_idx), "gnn_mae": float(gnn_mae), "gnn_r2": float(gnn_r2),
             "baseline_mae": float(base_mae), "baseline_r2": float(base_r2)}
    json.dump(result, open("quantumbridge_data/entry091_multibell_results.json", "w"), indent=2)
    print("saved -> quantumbridge_data/entry091_multibell_params.json, entry091_multibell_results.json")
