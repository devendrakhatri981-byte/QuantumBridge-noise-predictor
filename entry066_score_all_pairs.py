"""
QuantumBridge — Entry 066: score all 24,003 pairs (Kyiv + Sherbrooke +
Brisbane, 8,001 each) with the Entry 065 combined three-chip model.

Note on uncertainty: the Entry 057 MC-Dropout model was trained with
dropout active, which is what makes its uncertainty band calibrated
(validated in Entries 057/057b). Entry 065's combined model was trained
without dropout (same base architecture as Entries 060/061/063), so it
produces a single point estimate here -- applying dropout only at
inference time without having trained with it would produce an
uncertainty-shaped number that was never validated, which this project
has been careful not to do elsewhere. So: point estimate only, honestly
labeled as such in the demo.
"""

import json

import numpy as np

import entry063_train_gnn as base

MAX_N, MAX_E = base.MAX_N, base.MAX_E

params_json = json.load(open("quantumbridge_data/entry065_combined_params.json"))
params = {name: {"W": np.array(layer["W"]), "b": np.array(layer["b"])}
          for name, layer in params_json.items()}
node_stats = base.node_stats
edge_stats = base.edge_stats
glob_mu, glob_sd = np.array(base.glob_mu), np.array(base.glob_sd)


def pad_graph(g):
    n = g["n_nodes"]
    node_feat = np.zeros((MAX_N, base.N_NODE_FEATS), dtype=np.float32)
    for i, f in enumerate(g["nodes"]):
        node_feat[i] = f
    node_mask = np.zeros(MAX_N, dtype=np.float32); node_mask[:n] = 1.0

    e = len(g["edges"])
    src = np.zeros(MAX_E, dtype=np.int32); dst = np.zeros(MAX_E, dtype=np.int32)
    edge_feat = np.zeros((MAX_E, base.N_EDGE_FEATS), dtype=np.float32)
    for i, edge in enumerate(g["edges"]):
        u, v, err, dur, pos, is_final = edge
        src[i] = u; dst[i] = v
        edge_feat[i] = [err, dur, pos, is_final]
    edge_mask = np.zeros(MAX_E, dtype=np.float32); edge_mask[:e] = 1.0

    glob = np.array([1.0, 0.0, float(g["bfs_hop_distance"]), float(g["v4_1_prediction"])], dtype=np.float32)
    return node_feat, node_mask, src, dst, edge_feat, edge_mask, glob


def relu(x): return np.maximum(x, 0)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def lin(p, x): return x @ p["W"] + p["b"]


def mp_round(pm, pu, h, node_mask, src, dst, edge_feat, edge_mask):
    h_src, h_dst = h[src], h[dst]
    msg_in = np.concatenate([h_src, h_dst, edge_feat], axis=-1)
    msg = relu(lin(pm, msg_in)) * edge_mask[:, None]
    agg = np.zeros((MAX_N, base.H)); np.add.at(agg, dst, msg)
    deg = np.zeros(MAX_N); np.add.at(deg, dst, edge_mask); deg += 1e-6
    agg = agg / deg[:, None]
    upd_in = np.concatenate([h, agg], axis=-1)
    h_new = relu(lin(pu, upd_in))
    return h_new * node_mask[:, None]


def forward(node_feat, node_mask, src, dst, edge_feat, edge_mask, glob, chip):
    nmu, nsd = node_stats[chip]; emu, esd = edge_stats[chip]
    nf = (node_feat - nmu) / nsd
    ef = (edge_feat - emu) / esd
    gl = (glob - glob_mu) / glob_sd
    h = relu(lin(params["node_embed"], nf)) * node_mask[:, None]
    h = mp_round(params["msg"], params["update"], h, node_mask, src, dst, ef, edge_mask)
    h = mp_round(params["msg2"], params["update2"], h, node_mask, src, dst, ef, edge_mask)
    pooled = (h * node_mask[:, None]).sum(axis=0) / (node_mask.sum() + 1e-6)
    x = np.concatenate([pooled, gl], axis=-1)
    x = relu(lin(params["readout1"], x))
    out = lin(params["readout2"], x)[0]
    return float(sigmoid(out))


def predict(g):
    return forward(*pad_graph(g), g["chip"])


def ground_truth_index():
    idx = {}
    for path in ("quantumbridge_data/entry043_combined_dataset.json",
                "quantumbridge_data/entry045_bigghz_dataset.json",
                "quantumbridge_data/entry061_brisbane_bell_dataset.json"):
        try:
            recs = json.load(open(path))
        except FileNotFoundError:
            continue
        for r in recs:
            if r["kind"] != "bell":
                continue
            a, b = r["pairs_flat"]
            idx[(r["chip"], min(a, b), max(a, b))] = r["aer_ground_truth"]
    return idx


if __name__ == "__main__":
    all_pairs = json.load(open("quantumbridge_data/entry046_all_pairs.json"))
    gt = ground_truth_index()

    out = {}
    for chip in ("kyiv", "sherbrooke", "brisbane"):
        rows = []
        for g in all_pairs[chip]:
            a, b = g["a"], g["b"]
            pred = predict(g)
            key = (chip, min(a, b), max(a, b))
            real = gt.get(key)
            rows.append([a, b, g["bfs_hop_distance"], g["n_nodes"], len(g["edges"]),
                        round(g["v4_1_prediction"], 4), round(pred, 4),
                        round(real, 4) if real is not None else None])
        out[chip] = rows
        print(f"{chip}: scored {len(rows)} pairs, "
              f"{sum(1 for r in rows if r[7] is not None)} have real ground truth")

    tmp = "quantumbridge_data/entry066_combined_lookup.json.tmp"
    json.dump(out, open(tmp, "w"))
    import os
    os.replace(tmp, "quantumbridge_data/entry066_combined_lookup.json")
    sz = os.path.getsize("quantumbridge_data/entry066_combined_lookup.json")
    print(f"saved lookup table: {sz/1e6:.2f} MB")
