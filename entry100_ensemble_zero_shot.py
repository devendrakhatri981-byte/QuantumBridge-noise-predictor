"""
QuantumBridge — Entry 100: cross-chip zero-shot generalization test for
the multi-bell-pair template, mirroring the discipline established for
the single-pair pipeline (Entries 073/078/088: train on N-1 chips,
predict cold on the held-out Nth, never touched during training).

Only 3 chips exist in the multi-bell dataset so far (kyiv/sherbrooke/
brisbane, Entry 089/094), so this runs all three leave-one-chip-out
folds rather than picking a single dedicated holdout chip the way the
single-pair pipeline did with Quebec -- with only 3 chips, "reserve one
forever" would cut the already-small training set by a third permanently
for no extra rigor over rotating through all three.

For the held-out chip, per-chip normalization stats are unavailable (by
construction -- the whole point is the model never saw that chip), so
each fold applies the AVERAGE of the two training chips' node/edge
stats to the held-out chip's raw features, the same fallback used for
Quebec in Entry 078.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
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
glob_mu, glob_sd = glob_all.mean(axis=0), glob_all.std(axis=0) + 1e-6
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



from functools import partial

@partial(jax.jit, static_argnums=(3,))
def _step(params, opt_state, batch, opt):
    grads = jax.grad(lambda p: loss_fn(p, batch)[0])(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

def run_fold(holdout_chip, seed=0, mode='own', save=True):
    train_chips = [c for c in CHIPS if c != holdout_chip]
    train_mask = np.isin(chip_all, train_chips)
    test_mask = chip_all == holdout_chip

    # per-chip norm for the two TRAINING chips only
    node_stats, edge_stats = {}, {}
    for chip in train_chips:
        m = chip_all == chip
        nm = node_mask_all[m].astype(bool)
        em_ = edge_mask_all[m].astype(bool)
        node_stats[chip] = (node_feat_all[m][nm].mean(axis=0), node_feat_all[m][nm].std(axis=0) + 1e-6)
        edge_stats[chip] = (edge_feat_all[m][em_].mean(axis=0), edge_feat_all[m][em_].std(axis=0) + 1e-6)

    node_feat_norm = np.zeros_like(node_feat_all)
    edge_feat_norm = np.zeros_like(edge_feat_all)
    for chip in train_chips:
        m = chip_all == chip
        mu, sd = node_stats[chip]; node_feat_norm[m] = (node_feat_all[m] - mu) / sd
        mu, sd = edge_stats[chip]; edge_feat_norm[m] = (edge_feat_all[m] - mu) / sd

    # Entry 100 variant: normalize the held-out chip with ITS OWN calibration
    # stats (no labels used -- calibration data is public for any chip, so this
    # is deployable), instead of the Entry 078 averaged-stats fallback.
    if mode == 'own':
        nm_t = node_mask_all[test_mask].astype(bool); em_t = edge_mask_all[test_mask].astype(bool)
        n_mu, n_sd = node_feat_all[test_mask][nm_t].mean(axis=0), node_feat_all[test_mask][nm_t].std(axis=0) + 1e-6
        e_mu, e_sd = edge_feat_all[test_mask][em_t].mean(axis=0), edge_feat_all[test_mask][em_t].std(axis=0) + 1e-6
    else:  # 'avg': Entry 078/095 fallback
        n_mu = np.mean([node_stats[c][0] for c in train_chips], axis=0)
        n_sd = np.mean([node_stats[c][1] for c in train_chips], axis=0)
        e_mu = np.mean([edge_stats[c][0] for c in train_chips], axis=0)
        e_sd = np.mean([edge_stats[c][1] for c in train_chips], axis=0)
    node_feat_norm[test_mask] = (node_feat_all[test_mask] - n_mu) / n_sd
    edge_feat_norm[test_mask] = (edge_feat_all[test_mask] - e_mu) / e_sd

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    opt = optax.adam(3e-3)
    params = init_params(jax.random.PRNGKey(seed))
    opt_state = opt.init(params)
    rng = np.random.RandomState(seed)
    batch_size = 32

    for epoch in range(EPOCHS):
        perm = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (node_feat_norm[idx], node_mask_all[idx], src_all[idx], dst_all[idx],
                    edge_feat_norm[idx], edge_mask_all[idx], glob_all_n[idx], y_all[idx])
            params, opt_state = _step(params, opt_state, batch, opt)

    test_batch = (node_feat_norm[test_idx], node_mask_all[test_idx], src_all[test_idx],
                 dst_all[test_idx], edge_feat_norm[test_idx], edge_mask_all[test_idx],
                 glob_all_n[test_idx], y_all[test_idx])
    _, test_pred = loss_fn(params, test_batch)
    test_pred = np.array(test_pred)
    y_test = y_all[test_idx]
    baseline_test = baseline_all[test_idx]

    import os
    os.makedirs('quantumbridge_data/entry100_preds', exist_ok=True)
    if save: np.savez(f'quantumbridge_data/entry100_preds/{mode}_{holdout_chip}_seed{seed}.npz',
             pred=test_pred, y=y_test, base=baseline_test)
    gnn_mae = mean_absolute_error(y_test, test_pred)
    gnn_r2 = r2_score(y_test, test_pred)
    base_mae = mean_absolute_error(y_test, baseline_test)
    base_r2 = r2_score(y_test, baseline_test)
    return {"holdout": holdout_chip, "train_chips": train_chips, "n_train": len(train_idx),
            "n_test": len(test_idx), "gnn_mae": float(gnn_mae), "gnn_r2": float(gnn_r2),
            "baseline_mae": float(base_mae), "baseline_r2": float(base_r2)}


RESULTS_PATH = "quantumbridge_data/entry099_ownstats_results.json"

if __name__ == "__main__":
    import sys
    import os

    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else []
    done_chips = {r["holdout"] for r in results}

    requested = sys.argv[1] if len(sys.argv) > 1 else None
    targets = [requested] if requested else [c for c in CHIPS if c not in done_chips]

    print("=== Entry 100: multi-bell-pair cross-chip zero-shot (leave-one-chip-out) ===")
    for holdout in targets:
        if holdout in done_chips:
            print(f"skipping {holdout}, already done")
            continue
        r = run_fold(holdout)
        results = [x for x in results if x["holdout"] != holdout] + [r]
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
        better = (1 - r["gnn_mae"] / r["baseline_mae"]) * 100
        print(f"\nholdout={holdout:12s} train={r['train_chips']}  n_test={r['n_test']}")
        print(f"  GNN (zero-shot):        MAE={r['gnn_mae']*100:.2f} R2={r['gnn_r2']:.3f}")
        print(f"  product-of-marginals:   MAE={r['baseline_mae']*100:.2f} R2={r['baseline_r2']:.3f}")
        print(f"  GNN is {better:.1f}% {'better' if better > 0 else 'worse'} than the naive-independence baseline")
        print(f"saved -> {RESULTS_PATH}")
