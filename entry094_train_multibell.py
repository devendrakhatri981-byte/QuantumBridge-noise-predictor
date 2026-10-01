"""
QuantumBridge — Entry 094: multi-bell-pair model retrained on the scaled
3-chip dataset (kyiv/sherbrooke/brisbane, 650 circuits total, up from
Entry 091's kyiv-only 310). Same architecture as Entry 091, but now
adopts the per-chip normalization fix from Entry 048 -- each chip's
node/edge features are z-scored against that chip's OWN mean/std rather
than a single global one, since raw T1/T2/error scales differ across
chips (the same fix that mattered for the single-pair pipeline applies
here for the same reason: nothing about multi-bell circuits changes why
per-chip normalization was necessary).

Reports a per-chip breakdown of the held-out fold, mirroring Entry 077's
combined-model discipline, since an aggregate MAE alone could hide one
chip being systematically worse (as happened with Kyiv/Brisbane in the
single-pair case, Entry 079).

This is the "fold-in" test (all 3 chips seen during training, standard
5-fold CV) -- the cross-chip zero-shot generalization test (train on 2,
predict cold on the 3rd) is Entry 095, run separately after this.
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
CHIPS = ("kyiv", "sherbrooke", "brisbane")

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
chip_all = np.array([g["chip"] for g in raw])
N_GLOB = glob_all.shape[1]

# per-chip normalization (Entry 048 fix, carried over from the single-pair
# pipeline -- raw T1/T2/error/duration scales differ across chips, so a
# single global mean/std lets the largest chip's scale dominate).
node_stats, edge_stats = {}, {}
for chip in CHIPS:
    m = chip_all == chip
    nm = node_mask_all[m].astype(bool)
    em_ = edge_mask_all[m].astype(bool)
    node_stats[chip] = (node_feat_all[m][nm].mean(axis=0), node_feat_all[m][nm].std(axis=0) + 1e-6)
    edge_stats[chip] = (edge_feat_all[m][em_].mean(axis=0), edge_feat_all[m][em_].std(axis=0) + 1e-6)
glob_mu, glob_sd = glob_all.mean(axis=0), glob_all.std(axis=0) + 1e-6

node_feat_norm = np.zeros_like(node_feat_all)
edge_feat_norm = np.zeros_like(edge_feat_all)
for chip in CHIPS:
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

    print(f"\n=== Entry 094: 3-chip multi-bell-pair model (n={n}, test n={len(test_idx)}) ===")
    print(f"GNN:                    MAE={gnn_mae*100:.2f} R2={gnn_r2:.3f}")
    print(f"product-of-marginals:   MAE={base_mae*100:.2f} R2={base_r2:.3f}")
    print(f"improvement: {(1 - gnn_mae/base_mae)*100:.1f}% lower MAE than the naive-independence baseline")

    chip_test = chip_all[test_idx]
    per_chip = {}
    for chip in CHIPS:
        m = chip_test == chip
        if m.sum() == 0:
            continue
        c_gnn_mae = mean_absolute_error(y_test[m], test_pred[m])
        c_base_mae = mean_absolute_error(y_test[m], baseline_test[m])
        per_chip[chip] = {"n": int(m.sum()), "gnn_mae": float(c_gnn_mae), "baseline_mae": float(c_base_mae)}
        print(f"  {chip:12s} n={int(m.sum()):3d}  GNN MAE={c_gnn_mae*100:5.2f}  baseline MAE={c_base_mae*100:5.2f}")

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    export = {"params": params_json,
             "node_stats": {c: [v[0].tolist(), v[1].tolist()] for c, v in node_stats.items()},
             "edge_stats": {c: [v[0].tolist(), v[1].tolist()] for c, v in edge_stats.items()},
             "glob_mu": glob_mu.tolist(), "glob_sd": glob_sd.tolist(),
             "chips": list(CHIPS), "max_n": MAX_N, "max_e": MAX_E}
    json.dump(export, open("quantumbridge_data/entry094_multibell_params.json", "w"))
    result = {"n": n, "n_test": len(test_idx), "gnn_mae": float(gnn_mae), "gnn_r2": float(gnn_r2),
             "baseline_mae": float(base_mae), "baseline_r2": float(base_r2), "per_chip": per_chip}
    json.dump(result, open("quantumbridge_data/entry094_multibell_results.json", "w"), indent=2)
    print("saved -> quantumbridge_data/entry094_multibell_params.json, entry094_multibell_results.json")
