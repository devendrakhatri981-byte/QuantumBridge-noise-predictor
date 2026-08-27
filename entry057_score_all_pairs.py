"""
QuantumBridge — Entry 057d: re-score all 16,002 pairs with the deployment
MC-Dropout model, producing mean +/- std (uncertainty) for each pair
instead of a bare point estimate. Pure numpy, T=20 stochastic passes per
pair, no Aer simulation needed (reuses Entry 046's precomputed routing
graphs -- max 27 nodes/26 edges, well within the new 90/130 capacity).
"""

import json

import numpy as np

MAX_N, MAX_E = 90, 130
N_NODE_FEATS, N_EDGE_FEATS = 6, 4
H = 16
DROPOUT_RATE = 0.15
T_SAMPLES = 20
SEED = 42

model = json.load(open("quantumbridge_data/entry057_deploy_params.json"))
params = {name: {"W": np.array(layer["W"]), "b": np.array(layer["b"])}
          for name, layer in model["params"].items()}
node_stats = {chip: (np.array(v[0]), np.array(v[1])) for chip, v in model["node_stats"].items()}
edge_stats = {chip: (np.array(v[0]), np.array(v[1])) for chip, v in model["edge_stats"].items()}
glob_mu, glob_sd = np.array(model["glob_mu"]), np.array(model["glob_sd"])


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

    # glob layout matches Entry 048+: [kind_bell, kind_ghz, bfs_hop_distance, v4_1_prediction]
    # -- NO chip one-hot (removed in Entry 048's generalization fix)
    glob = np.array([1.0, 0.0, float(g["bfs_hop_distance"]), float(g["v4_1_prediction"])], dtype=np.float32)
    return node_feat, node_mask, src, dst, edge_feat, edge_mask, glob


def relu(x): return np.maximum(x, 0)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def lin(p, x): return x @ p["W"] + p["b"]


def dropout(x, rng, mask=None):
    keep = rng.random(x.shape) < (1.0 - DROPOUT_RATE)
    out = np.where(keep, x / (1.0 - DROPOUT_RATE), 0.0)
    if mask is not None:
        out = out * mask[:, None]
    return out


def mp_round(pm, pu, h, node_mask, src, dst, edge_feat, edge_mask):
    h_src, h_dst = h[src], h[dst]
    msg_in = np.concatenate([h_src, h_dst, edge_feat], axis=-1)
    msg = relu(lin(pm, msg_in)) * edge_mask[:, None]
    agg = np.zeros((MAX_N, H)); np.add.at(agg, dst, msg)
    deg = np.zeros(MAX_N); np.add.at(deg, dst, edge_mask); deg += 1e-6
    agg = agg / deg[:, None]
    upd_in = np.concatenate([h, agg], axis=-1)
    h_new = relu(lin(pu, upd_in))
    return h_new * node_mask[:, None]


def forward_stochastic(node_feat, node_mask, src, dst, edge_feat, edge_mask, glob, chip, rng):
    nmu, nsd = node_stats[chip]; emu, esd = edge_stats[chip]
    nf = (node_feat - nmu) / nsd
    ef = (edge_feat - emu) / esd
    gl = (glob - glob_mu) / glob_sd
    h = relu(lin(params["node_embed"], nf)) * node_mask[:, None]
    h = dropout(h, rng, node_mask)
    h = mp_round(params["msg"], params["update"], h, node_mask, src, dst, ef, edge_mask)
    h = dropout(h, rng, node_mask)
    h = mp_round(params["msg2"], params["update2"], h, node_mask, src, dst, ef, edge_mask)
    h = dropout(h, rng, node_mask)
    pooled = (h * node_mask[:, None]).sum(axis=0) / (node_mask.sum() + 1e-6)
    x = np.concatenate([pooled, gl], axis=-1)
    x = relu(lin(params["readout1"], x))
    out = lin(params["readout2"], x)[0]
    return float(sigmoid(out))


def predict_mc(g, rng):
    padded = pad_graph(g)
    samples = [forward_stochastic(*padded, g["chip"], rng) for _ in range(T_SAMPLES)]
    samples = np.array(samples)
    return float(samples.mean()), float(samples.std())


def ground_truth_index():
    idx = {}
    for path in ("quantumbridge_data/entry043_combined_dataset.json",
                "quantumbridge_data/entry045_bigghz_dataset.json"):
        for r in json.load(open(path)):
            if r["kind"] != "bell":
                continue
            a, b = r["pairs_flat"]
            idx[(r["chip"], min(a, b), max(a, b))] = r["aer_ground_truth"]
    return idx


if __name__ == "__main__":
    all_pairs = json.load(open("quantumbridge_data/entry046_all_pairs.json"))
    gt = ground_truth_index()
    rng = np.random.default_rng(SEED)

    out = {}
    for chip in ("kyiv", "sherbrooke"):
        rows = []
        for g in all_pairs[chip]:
            a, b = g["a"], g["b"]
            mean_pred, std_pred = predict_mc(g, rng)
            key = (chip, min(a, b), max(a, b))
            real = gt.get(key)
            rows.append([a, b, g["bfs_hop_distance"], g["n_nodes"], len(g["edges"]),
                        round(g["v4_1_prediction"], 4), round(mean_pred, 4), round(std_pred, 4),
                        round(real, 4) if real is not None else None])
        out[chip] = rows
        print(f"{chip}: scored {len(rows)} pairs, "
              f"{sum(1 for r in rows if r[8] is not None)} have real ground truth")

    json.dump(out, open("quantumbridge_data/entry057_lookup_with_uncertainty.json", "w"))
    import os
    sz = os.path.getsize("quantumbridge_data/entry057_lookup_with_uncertainty.json")
    print(f"saved lookup table: {sz/1e6:.2f} MB")
