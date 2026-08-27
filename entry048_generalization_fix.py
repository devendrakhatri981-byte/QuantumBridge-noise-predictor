"""
QuantumBridge — Entry 048: fix cross-chip generalization.

Entry 047 found cross-chip R2 collapsing from 0.97 to 0.04-0.37 when
trained on one chip and tested cold on the other. Two likely root
causes, both addressed here:

1. CHIP IDENTITY LEAKAGE: the global feature vector included a
   chip_kyiv/chip_sherbrooke one-hot. When trained on Kyiv only, that
   flag is CONSTANT ([1,0]) for every single training example -- at
   test time on Sherbrooke it flips to [0,1], a value the network never
   saw during training. This is a genuine train/test distribution shift
   baked into the input itself, independent of anything physical.
   FIX: drop the chip one-hot from the global feature vector entirely.

2. GLOBAL (CROSS-CHIP) FEATURE NORMALIZATION: node/edge features
   (T1, T2, readout, gate error, ...) were normalized using mean/std
   computed across BOTH chips combined. If Kyiv and Sherbrooke have
   systematically different absolute calibration scales, a model
   trained only on Kyiv's normalized range sees out-of-range values for
   Sherbrooke at test time even before any physics is involved.
   FIX: normalize every node/edge feature PER CHIP (using that chip's
   own full calibration distribution, known in advance -- this is
   available at deployment time without needing labels), so the same
   normalized value means the same relative thing ("below-average T1
   for this chip") on every chip.

Everything else (architecture, feature set, MAX_N/MAX_E, dataset) is
identical to Entry 047's cross-chip test, so any improvement here is
attributable to these two fixes specifically.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.metrics import mean_absolute_error, r2_score

MAX_N, MAX_E = 48, 64
N_NODE_FEATS, N_EDGE_FEATS = 6, 4
H = 16
IN_PATH = "quantumbridge_data/entry045_graph_dataset.json"

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
        u, v, err, dur, pos, is_final = edge
        src[i] = u; dst[i] = v
        edge_feat[i] = [err, dur, pos, is_final]
    edge_mask = np.zeros(MAX_E, dtype=np.float32); edge_mask[:e] = 1.0

    # FIX 1: no chip one-hot -- only kind + hop-distance + v4.1 pred, all of
    # which are meaningful/comparable across chips (v4.1 is a probability,
    # hop-distance is a graph-topology count, both chips share the same
    # heavy-hex-family coupling structure).
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
chip_all = np.array([g["chip"] for g in raw])
N_GLOB = glob_all.shape[1]

# FIX 2: per-chip normalization stats, computed from each chip's OWN full
# calibration distribution (all circuits of that chip, regardless of
# train/test split -- this is legitimate because a chip's calibration
# distribution is known at deployment time without needing any labels).
node_stats, edge_stats = {}, {}
for chip in ("kyiv", "sherbrooke"):
    m = chip_all == chip
    nm = node_mask_all[m].astype(bool)
    em = edge_mask_all[m].astype(bool)
    node_stats[chip] = (node_feat_all[m][nm].mean(axis=0), node_feat_all[m][nm].std(axis=0) + 1e-6)
    edge_stats[chip] = (edge_feat_all[m][em].mean(axis=0), edge_feat_all[m][em].std(axis=0) + 1e-6)
glob_mu, glob_sd = glob_all.mean(axis=0), glob_all.std(axis=0) + 1e-6

node_feat_norm = np.zeros_like(node_feat_all)
edge_feat_norm = np.zeros_like(edge_feat_all)
for chip in ("kyiv", "sherbrooke"):
    m = chip_all == chip
    mu, sd = node_stats[chip]
    node_feat_norm[m] = (node_feat_all[m] - mu) / sd
    mu, sd = edge_stats[chip]
    edge_feat_norm[m] = (edge_feat_all[m] - mu) / sd
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


def run(name, train_idx, test_idx, epochs=100, chunk_epochs=25, batch_size=64, lr=3e-3, seed=0):
    import os
    ckpt_path = f"quantumbridge_data/entry048_{name}_ckpt.json"
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
                    edge_feat_norm[idx], edge_mask_all[idx], glob_all_n[idx], y_all[idx])
            (l, grads) = jax.value_and_grad(lambda p: loss_fn(p, batch)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    json.dump({"params": params_json, "epoch": target}, open(ckpt_path, "w"))
    if target < epochs:
        print(f"[{name}] checkpoint at epoch {target}/{epochs} -- rerun to continue")
        return None

    test_batch = (node_feat_norm[test_idx], node_mask_all[test_idx], src_all[test_idx], dst_all[test_idx],
                 edge_feat_norm[test_idx], edge_mask_all[test_idx], glob_all_n[test_idx], y_all[test_idx])
    _, test_pred = loss_fn(params, test_batch)
    test_pred = np.array(test_pred)
    mae = mean_absolute_error(y_all[test_idx], test_pred)
    r2 = r2_score(y_all[test_idx], test_pred)
    is_fc = np.array([abs(y_all[i] - 0.5) < 0.02 and raw[i]["bfs_hop_distance"] <= 10 for i in test_idx])
    mae_fc = mean_absolute_error(y_all[test_idx][is_fc], test_pred[is_fc]) if is_fc.sum() > 0 else None
    result = {"name": name, "mae": float(mae), "r2": float(r2),
             "mae_fc": float(mae_fc) if mae_fc is not None else None,
             "n_test": len(test_idx), "n_fc": int(is_fc.sum())}
    print(f"[{name}] DONE: MAE={mae*100:.2f} R2={r2:.3f} "
          f"fc_MAE={(mae_fc*100 if mae_fc is not None else -1):.2f} (n_fc={int(is_fc.sum())})")
    return result


if __name__ == "__main__":
    import sys, os
    direction = sys.argv[1] if len(sys.argv) > 1 else "kyiv_to_sherbrooke"
    idx_all = np.arange(len(raw))
    if direction == "kyiv_to_sherbrooke":
        train_idx = idx_all[chip_all == "kyiv"]; test_idx = idx_all[chip_all == "sherbrooke"]
    else:
        train_idx = idx_all[chip_all == "sherbrooke"]; test_idx = idx_all[chip_all == "kyiv"]

    RESULTS_PATH = "quantumbridge_data/entry048_results.json"
    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
    r = run(direction, train_idx, test_idx)
    if r:
        results[direction] = r
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
