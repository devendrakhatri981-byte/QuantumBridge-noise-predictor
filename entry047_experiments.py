"""
QuantumBridge — Entry 047: generalization + ablation experiments.

Directly responding to the AI-model review consensus (three independent
reviews of the public demo all converged on the same critique): the GNN's
apparent win over v4.1 could just mean it fits Aer's simulator better,
not that it learned anything transferable. Two experiments to check this
honestly:

1. CROSS-CHIP GENERALIZATION: train on one chip only, test cold on the
   other. If accuracy collapses, the model memorized chip-specific
   quirks rather than learning route/calibration physics that transfers.

2. FEATURE ABLATION: remove each of Entry 044's three added feature
   groups (control/target role, T1/T2 ratio, route-position/is-final)
   one at a time, plus a "no message passing" variant (raw per-node
   features pooled directly, no graph propagation) to check whether the
   graph structure itself is earning its keep or whether a flat feature
   set would do just as well.

All experiments reuse the exact Entry 045 architecture (MAX_N=48,
MAX_E=64, H=16, 2 message-passing rounds unless disabled) and the same
combined 2,317-circuit dataset, so results are directly comparable to
Entry 045's CV numbers (fold 0: MAE=1.32, R2=0.968). To keep this
tractable, every variant here uses a SINGLE fixed 80/20 split (KFold
seed=42, fold 0) rather than full 5-fold CV -- explicitly a faster,
lower-confidence proxy, not a replacement for Entry 045's CV numbers.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.model_selection import KFold
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

    glob = np.array([
        1.0 if g["chip"] == "kyiv" else 0.0,
        1.0 if g["chip"] == "sherbrooke" else 0.0,
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

NODE_NAMES = ["T1", "T2", "readout", "T1_T2_ratio", "is_control", "is_target"]
EDGE_NAMES = ["gate_err", "duration", "route_pos", "is_final"]


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


def make_forward(use_mp=True):
    def forward_single(params, node_feat, node_mask, src, dst, edge_feat, edge_mask, glob):
        h = jax.nn.relu(apply_lin(params["node_embed"], node_feat)) * node_mask[:, None]
        if use_mp:
            h = mp_round(params["msg"], params["update"], h, node_mask, src, dst, edge_feat, edge_mask)
            h = mp_round(params["msg2"], params["update2"], h, node_mask, src, dst, edge_feat, edge_mask)
        pooled = (h * node_mask[:, None]).sum(axis=0) / (node_mask.sum() + 1e-6)
        x = jnp.concatenate([pooled, glob], axis=-1)
        x = jax.nn.relu(apply_lin(params["readout1"], x))
        out = apply_lin(params["readout2"], x)[0]
        return jax.nn.sigmoid(out)
    return jax.vmap(forward_single, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))


def run_experiment(name, train_idx, test_idx, node_keep, edge_keep, use_mp=True,
                   epochs=100, chunk_epochs=25, batch_size=64, lr=3e-3, seed=0):
    """node_keep / edge_keep: boolean masks over feature columns -- excluded
    columns are zeroed (and excluded from normalization stats), which is
    equivalent to removing that feature without reshaping the network."""
    ckpt_path = f"quantumbridge_data/entry047_{name}_ckpt.json"
    import os

    nf = node_feat_all.copy(); nf[:, :, ~node_keep] = 0
    ef = edge_feat_all.copy(); ef[:, :, ~edge_keep] = 0

    node_mask_bool = node_mask_all.astype(bool)
    edge_mask_bool = edge_mask_all.astype(bool)
    node_mu = nf[node_mask_bool].mean(axis=0); node_sd = nf[node_mask_bool].std(axis=0) + 1e-6
    node_mu[~node_keep] = 0; node_sd[~node_keep] = 1
    edge_mu = ef[edge_mask_bool].mean(axis=0); edge_sd = ef[edge_mask_bool].std(axis=0) + 1e-6
    edge_mu[~edge_keep] = 0; edge_sd[~edge_keep] = 1
    glob_mu = glob_all.mean(axis=0); glob_sd = glob_all.std(axis=0) + 1e-6

    nf_n = (nf - node_mu) / node_sd
    ef_n = (ef - edge_mu) / edge_sd
    gl_n = (glob_all - glob_mu) / glob_sd

    forward_batch = make_forward(use_mp)

    def loss_fn(params, batch):
        (a, b, c, d, e, f, g, y) = batch
        pred = forward_batch(params, a, b, c, d, e, f, g)
        return jnp.mean(jnp.abs(pred - y)), pred

    opt = optax.adam(lr)
    if os.path.exists(ckpt_path):
        ck = json.load(open(ckpt_path))
        params = {k: {"W": jnp.array(v["W"]), "b": jnp.array(v["b"])} for k, v in ck["params"].items()}
        opt_state = opt.init(params)
        done_epochs = ck["epoch"]
    else:
        key = jax.random.PRNGKey(seed)
        params = init_params(key)
        opt_state = opt.init(params)
        done_epochs = 0

    rng = np.random.RandomState(seed + done_epochs)
    target = min(epochs, done_epochs + chunk_epochs)
    n_train = len(train_idx)
    for epoch in range(done_epochs, target):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = train_idx[perm[start:start + batch_size]]
            batch = (nf_n[idx], node_mask_all[idx], src_all[idx], dst_all[idx],
                    ef_n[idx], edge_mask_all[idx], gl_n[idx], y_all[idx])
            (l, grads) = jax.value_and_grad(lambda p: loss_fn(p, batch)[0])(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    params_json = {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in params.items()}
    json.dump({"params": params_json, "epoch": target}, open(ckpt_path, "w"))

    if target < epochs:
        print(f"[{name}] checkpoint at epoch {target}/{epochs} -- rerun to continue")
        return None

    test_batch = (nf_n[test_idx], node_mask_all[test_idx], src_all[test_idx], dst_all[test_idx],
                 ef_n[test_idx], edge_mask_all[test_idx], gl_n[test_idx], y_all[test_idx])
    _, test_pred = loss_fn(params, test_batch)
    test_pred = np.array(test_pred)
    mae = mean_absolute_error(y_all[test_idx], test_pred)
    r2 = r2_score(y_all[test_idx], test_pred)
    is_fc = np.array([abs(y_all[i] - 0.5) < 0.02 and raw[i]["bfs_hop_distance"] <= 10 for i in test_idx])
    mae_fc = mean_absolute_error(y_all[test_idx][is_fc], test_pred[is_fc]) if is_fc.sum() > 0 else None
    result = {"name": name, "mae": float(mae), "r2": float(r2),
             "mae_fc": float(mae_fc) if mae_fc is not None else None,
             "n_test": len(test_idx), "n_fc": int(is_fc.sum())}
    if mae_fc is not None:
        print(f"[{name}] DONE: MAE={mae*100:.2f} R2={r2:.3f} fc_MAE={mae_fc*100:.2f} (n_fc={int(is_fc.sum())})")
    else:
        print(f"[{name}] DONE: MAE={mae*100:.2f} R2={r2:.3f} (no fc cases in test)")
    return result


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    idx_all = np.arange(len(raw))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(idx_all))
    fold0_train, fold0_test = splits[0]

    all_node = np.ones(N_NODE_FEATS, dtype=bool)
    all_edge = np.ones(N_EDGE_FEATS, dtype=bool)

    RESULTS_PATH = "quantumbridge_data/entry047_results.json"
    results = json.load(open(RESULTS_PATH)) if __import__("os").path.exists(RESULTS_PATH) else {}

    if mode == "cross_chip":
        direction = sys.argv[2]  # "kyiv_to_sherbrooke" or "sherbrooke_to_kyiv"
        if direction == "kyiv_to_sherbrooke":
            train_idx = idx_all[chip_all == "kyiv"]
            test_idx = idx_all[chip_all == "sherbrooke"]
        else:
            train_idx = idx_all[chip_all == "sherbrooke"]
            test_idx = idx_all[chip_all == "kyiv"]
        r = run_experiment(f"cross_{direction}", train_idx, test_idx, all_node, all_edge)
        if r:
            results[r["name"]] = r
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    elif mode == "ablation":
        variant = sys.argv[2]
        node_keep = all_node.copy(); edge_keep = all_edge.copy(); use_mp = True
        if variant == "no_role":
            node_keep[[4, 5]] = False   # is_control, is_target
        elif variant == "no_ratio":
            node_keep[3] = False        # T1_T2_ratio
        elif variant == "no_routepos":
            edge_keep[[2, 3]] = False   # route_pos, is_final
        elif variant == "no_mp":
            use_mp = False
        elif variant == "full":
            pass
        else:
            print("unknown variant"); sys.exit(1)
        r = run_experiment(f"ablation_{variant}", fold0_train, fold0_test, node_keep, edge_keep, use_mp=use_mp)
        if r:
            results[r["name"]] = r
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    else:
        print("usage: entry047_experiments.py cross_chip <kyiv_to_sherbrooke|sherbrooke_to_kyiv>")
        print("       entry047_experiments.py ablation <full|no_role|no_ratio|no_routepos|no_mp>")
