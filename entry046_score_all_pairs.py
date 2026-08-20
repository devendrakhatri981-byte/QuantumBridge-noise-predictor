"""
QuantumBridge — Entry 046c: run the final GNN over all 16,002 precomputed
pair-graphs and build the compact lookup table the public demo embeds.

Loads the fixed trained weights from entry046_final_gnn_params.json (no
retraining here), batches every pair through the same forward pass used
in training, and merges in real Aer/MPS ground truth wherever a pair
happens to match one of the 2,317 circuits actually simulated -- most
pairs won't have this (only ~2,200 of the 16,002 possible bell pairs
were ever run through Aer), and the demo must say so honestly rather
than implying every number is hardware-verified.
"""

import json

import numpy as np

MAX_N, MAX_E = 48, 64
N_NODE_FEATS, N_EDGE_FEATS = 6, 4
H = 16

model = json.load(open("quantumbridge_data/entry046_final_gnn_params.json"))
params = {name: {"W": np.array(layer["W"]), "b": np.array(layer["b"])}
          for name, layer in model["params"].items()}
norm = model["norm"]
node_mu, node_sd = np.array(norm["node_mu"]), np.array(norm["node_sd"])
edge_mu, edge_sd = np.array(norm["edge_mu"]), np.array(norm["edge_sd"])
glob_mu, glob_sd = np.array(norm["glob_mu"]), np.array(norm["glob_sd"])


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
        1.0, 0.0,  # kind: always bell for this sweep
        float(g["bfs_hop_distance"]), float(g["v4_1_prediction"]),
    ], dtype=np.float32)
    return node_feat, node_mask, src, dst, edge_feat, edge_mask, glob


def relu(x): return np.maximum(x, 0)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def lin(p, x): return x @ p["W"] + p["b"]


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


def forward(node_feat, node_mask, src, dst, edge_feat, edge_mask, glob):
    nf = (node_feat - node_mu) / node_sd
    ef = (edge_feat - edge_mu) / edge_sd
    gl = (glob - glob_mu) / glob_sd
    h = relu(lin(params["node_embed"], nf)) * node_mask[:, None]
    h = mp_round(params["msg"], params["update"], h, node_mask, src, dst, ef, edge_mask)
    h = mp_round(params["msg2"], params["update2"], h, node_mask, src, dst, ef, edge_mask)
    pooled = (h * node_mask[:, None]).sum(axis=0) / (node_mask.sum() + 1e-6)
    x = np.concatenate([pooled, gl], axis=-1)
    x = relu(lin(params["readout1"], x))
    out = lin(params["readout2"], x)[0]
    return float(sigmoid(out))


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

    out = {}
    for chip in ("kyiv", "sherbrooke"):
        rows = []
        for g in all_pairs[chip]:
            a, b = g["a"], g["b"]
            padded = pad_graph(g)
            gnn_pred = forward(*padded)
            key = (chip, min(a, b), max(a, b))
            real = gt.get(key)
            rows.append([a, b, g["bfs_hop_distance"], g["n_nodes"], len(g["edges"]),
                        round(g["v4_1_prediction"], 4), round(gnn_pred, 4),
                        round(real, 4) if real is not None else None])
        out[chip] = rows
        print(f"{chip}: scored {len(rows)} pairs, "
              f"{sum(1 for r in rows if r[7] is not None)} have real ground truth")

    json.dump(out, open("quantumbridge_data/entry046_lookup.json", "w"))
    import os
    sz = os.path.getsize("quantumbridge_data/entry046_lookup.json")
    print(f"saved lookup table: {sz/1e6:.2f} MB")
