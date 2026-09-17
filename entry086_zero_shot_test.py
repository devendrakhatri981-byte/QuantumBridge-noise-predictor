"""
QuantumBridge — Entry 073: the true zero-shot fourth-chip test.

Builds graphs for the 708 Osaka eval-only bell-pair circuits (never folded
into any training dataset), then scores them with the Entry 071 deployed
model -- trained on Kyiv+Sherbrooke+Brisbane, with ZERO Osaka exposure of
any kind. This is the genuine cold-transfer test this project has not
run before: Brisbane (Entry 061) became a training chip almost immediately
(Entry 065), so no chip has ever been held out this cleanly while still
being evaluated against the actual deployed, in-production model.

Reports MAE, R^2, floor-collapse MAE on Osaka, and compares mean predicted
uncertainty (std) on Osaka vs. the model's three training chips -- if the
model has learned to recognize when it's out of its depth, uncertainty
should widen on Osaka relative to its home turf.
"""

import json

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

import emulator_v3_routing as em
import emulator_v4 as v4
from entry044_build_graphs import BACKENDS, graph_for_record

RECORDS_PATH = "quantumbridge_data/entry084_kyoto_bell_dataset.json"
GRAPHS_PATH = "quantumbridge_data/entry086_kyoto_graph_dataset.json"

MAX_N, MAX_E = 90, 130
N_NODE_FEATS, N_EDGE_FEATS = 6, 4
H = 16
DROPOUT_RATE = 0.15
T_SAMPLES = 20
SEED = 42


def build_graphs():
    import os
    records = json.load(open(RECORDS_PATH))
    out = json.load(open(GRAPHS_PATH)) if os.path.exists(GRAPHS_PATH) else []
    done = len(out)
    print(f"resuming with {done}/{len(records)} kyoto eval graphs already built")

    backend = BACKENDS["kyoto"]()
    graph = em.build_connectivity_graph(em.load_calibration("kyoto"), "kyoto")
    coh = v4.load_coherence("kyoto")

    for i, r in enumerate(records):
        if i < done:
            continue
        g = graph_for_record(backend, graph, coh, r)
        g.update({"chip": "kyoto", "kind": r["kind"],
                  "aer_ground_truth": r["aer_ground_truth"],
                  "v4_1_prediction": r["v4_1_prediction"],
                  "bfs_hop_distance": r["bfs_hop_distance"]})
        out.append(g)

        if (i + 1) % 100 == 0 or i + 1 == len(records):
            tmp = GRAPHS_PATH + ".tmp"
            json.dump(out, open(tmp, "w"))
            os.replace(tmp, GRAPHS_PATH)
            print(f"  {i+1}/{len(records)} graphs built, checkpoint saved")

    tmp = GRAPHS_PATH + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, GRAPHS_PATH)
    print(f"DONE: {len(out)} kyoto eval graphs -> {GRAPHS_PATH}")
    return out


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


def forward_stochastic(node_feat, node_mask, src, dst, edge_feat, edge_mask, glob, chip_stats, rng, params, glob_mu, glob_sd):
    nmu, nsd = chip_stats["node"]; emu, esd = chip_stats["edge"]
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


def predict_mc(g, rng, chip_stats, params, glob_mu, glob_sd):
    padded = pad_graph(g)
    samples = [forward_stochastic(*padded, chip_stats, rng, params, glob_mu, glob_sd) for _ in range(T_SAMPLES)]
    samples = np.array(samples)
    return float(samples.mean()), float(samples.std())


if __name__ == "__main__":
    graphs = build_graphs()

    model = json.load(open("quantumbridge_data/entry085_deploy_params.json"))
    params = {name: {"W": np.array(layer["W"]), "b": np.array(layer["b"])}
              for name, layer in model["params"].items()}
    node_stats = {chip: (np.array(v[0]), np.array(v[1])) for chip, v in model["node_stats"].items()}
    edge_stats = {chip: (np.array(v[0]), np.array(v[1])) for chip, v in model["edge_stats"].items()}
    glob_mu, glob_sd = np.array(model["glob_mu"]), np.array(model["glob_sd"])

    # Kyoto has NO entry in node_stats/edge_stats (never trained on -- it is the new clean holdout) -- use
    # the mean of the three training chips' normalization stats as the best
    # available proxy, since a deployed tool would need *some* answer here
    # for a genuinely new chip with no per-chip calibration baked in yet.
    node_mu_avg = np.mean([node_stats[c][0] for c in ("kyiv", "sherbrooke", "brisbane", "osaka", "quebec")], axis=0)
    node_sd_avg = np.mean([node_stats[c][1] for c in ("kyiv", "sherbrooke", "brisbane", "osaka", "quebec")], axis=0)
    edge_mu_avg = np.mean([edge_stats[c][0] for c in ("kyiv", "sherbrooke", "brisbane", "osaka", "quebec")], axis=0)
    edge_sd_avg = np.mean([edge_stats[c][1] for c in ("kyiv", "sherbrooke", "brisbane", "osaka", "quebec")], axis=0)
    osaka_stats = {"node": (node_mu_avg, node_sd_avg), "edge": (edge_mu_avg, edge_sd_avg)}

    rng = np.random.default_rng(SEED)
    y_true, y_pred, y_std = [], [], []
    for g in graphs:
        mean_p, std_p = predict_mc(g, rng, osaka_stats, params, glob_mu, glob_sd)
        y_true.append(g["aer_ground_truth"])
        y_pred.append(mean_p)
        y_std.append(std_p)

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_std = np.array(y_std)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    is_fc = np.abs(y_true - 0.5) < 0.02
    hop = np.array([g["bfs_hop_distance"] for g in graphs])
    is_fc = is_fc & (hop <= 10)
    mae_fc = mean_absolute_error(y_true[is_fc], y_pred[is_fc]) if is_fc.sum() > 0 else None

    print(f"\n=== TRUE ZERO-SHOT: Entry 085 5-chip model (never saw Kyoto) on {len(graphs)} Kyoto circuits ===")
    print(f"MAE={mae*100:.2f} R2={r2:.3f} mean_std={y_std.mean()*100:.2f}pts "
          f"floor-collapse MAE={(mae_fc*100 if mae_fc else -1):.2f} (n_fc={is_fc.sum()})")

    result = {"chip": "kyoto", "n": len(graphs), "mae": float(mae), "r2": float(r2),
             "mean_std": float(y_std.mean()), "mae_fc": float(mae_fc) if mae_fc is not None else None,
             "n_fc": int(is_fc.sum())}
    tmp = "quantumbridge_data/entry086_zero_shot_results.json.tmp"
    json.dump(result, open(tmp, "w"), indent=2)
    import os
    os.replace(tmp, "quantumbridge_data/entry086_zero_shot_results.json")
    print("saved -> quantumbridge_data/entry086_zero_shot_results.json")
