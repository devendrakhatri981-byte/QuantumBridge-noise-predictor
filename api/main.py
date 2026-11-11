"""
QuantumBridge Live Inference API (v1 productization)

Replaces the static demo's precomputed lookup table with a real backend:
every request does a live Qiskit transpile + SABRE routing + feature
extraction + v4.1 closed-form prediction + Entry 085 GNN inference (with
calibrated MC-Dropout uncertainty), on demand, for any pair of qubits on
any of the five trained chips (Kyiv, Sherbrooke, Brisbane, Osaka, Quebec).

Updated in Entry 097: this was deployed on the 4-chip Entry 077 model
for a while after the 5-chip Entry 085 model existed (Entries 108-112
built and validated entry085, but the live API's MODEL_PATH was never
bumped past entry077, and Quebec kept being refused as if it were still
a zero-shot holdout after it had already been folded into training).
Cusco is now the project's zero-shot holdout chip (Entries 087-088) and
is refused with an explicit message in its place.

This is deliberately scoped, not the full "any circuit, any chip" vision:
- Circuit type: Bell pairs only (H + CX between two qubits), matching what
  the model was trained and validated on. Extending to arbitrary circuits
  is future work (the training data does include GHZ/star/chain circuits,
  but the live feature-extraction path here has only been wired and
  tested for bell pairs).
- Chips: kyiv, sherbrooke, brisbane, osaka, quebec -- the five chips the
  deployed model was trained on. Cusco is the project's current zero-shot
  holdout chip (Entries 087-088) and is refused here with an explicit
  message, rather than silently degrading.
- Validation: all noise data comes from Qiskit's fake-backend snapshots
  run through Aer's simulator, not live queued jobs on physical hardware.
  This is disclosed in every response, not just documentation.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health
    GET  /chips
    POST /predict   {"chip": "kyiv", "qubit_a": 0, "qubit_b": 5}
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke, FakeBrisbane, FakeOsaka, FakeQuebec

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost
from entry044_build_graphs import graph_for_record
from entry092_decomposed_prediction import exact_dwell_cost_decomposed, summarize_failure_mode
from entry093_recommend import recommend as _recommend

MAX_N, MAX_E = 90, 130
N_NODE_FEATS, N_EDGE_FEATS = 6, 4
H = 16
DROPOUT_RATE = 0.15
T_SAMPLES = 20

BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke, "brisbane": FakeBrisbane,
           "osaka": FakeOsaka, "quebec": FakeQuebec}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "quantumbridge_data",
                          "entry085_deploy_params.json")


# ---------------------------------------------------------------------------
# Model load (once, at startup)
# ---------------------------------------------------------------------------

def _load_model():
    import json
    model = json.load(open(MODEL_PATH))
    params = {name: {"W": np.array(layer["W"]), "b": np.array(layer["b"])}
              for name, layer in model["params"].items()}
    node_stats = {chip: (np.array(v[0]), np.array(v[1])) for chip, v in model["node_stats"].items()}
    edge_stats = {chip: (np.array(v[0]), np.array(v[1])) for chip, v in model["edge_stats"].items()}
    glob_mu, glob_sd = np.array(model["glob_mu"]), np.array(model["glob_sd"])
    return params, node_stats, edge_stats, glob_mu, glob_sd


PARAMS, NODE_STATS, EDGE_STATS, GLOB_MU, GLOB_SD = _load_model()

_backend_cache = {}
_graph_cache = {}
_coh_cache = {}


def get_backend_ctx(chip):
    if chip not in _backend_cache:
        _backend_cache[chip] = BACKENDS[chip]()
        _graph_cache[chip] = em.build_connectivity_graph(em.load_calibration(chip), chip)
        _coh_cache[chip] = v4.load_coherence(chip)
    return _backend_cache[chip], _graph_cache[chip], _coh_cache[chip]


# ---------------------------------------------------------------------------
# Live feature extraction + prediction (mirrors entry046_precompute_pairs.py
# and entry072_score_all_pairs.py, but computed on demand instead of offline)
# ---------------------------------------------------------------------------

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


def forward_stochastic(node_feat, node_mask, src, dst, edge_feat, edge_mask, glob, chip, rng):
    nmu, nsd = NODE_STATS[chip]; emu, esd = EDGE_STATS[chip]
    nf = (node_feat - nmu) / nsd
    ef = (edge_feat - emu) / esd
    gl = (glob - GLOB_MU) / GLOB_SD
    h = relu(lin(PARAMS["node_embed"], nf)) * node_mask[:, None]
    h = dropout(h, rng, node_mask)
    h = mp_round(PARAMS["msg"], PARAMS["update"], h, node_mask, src, dst, ef, edge_mask)
    h = dropout(h, rng, node_mask)
    h = mp_round(PARAMS["msg2"], PARAMS["update2"], h, node_mask, src, dst, ef, edge_mask)
    h = dropout(h, rng, node_mask)
    pooled = (h * node_mask[:, None]).sum(axis=0) / (node_mask.sum() + 1e-6)
    x = np.concatenate([pooled, gl], axis=-1)
    x = relu(lin(PARAMS["readout1"], x))
    out = lin(PARAMS["readout2"], x)[0]
    return float(sigmoid(out))


def predict_mc(g, chip, seed):
    rng = np.random.default_rng(seed)
    padded = pad_graph(g)
    samples = [forward_stochastic(*padded, chip, rng) for _ in range(T_SAMPLES)]
    samples = np.array(samples)
    return float(samples.mean()), float(samples.std())


def live_predict(chip, a, b):
    backend, graph, coh = get_backend_ctx(chip)
    nq = backend.num_qubits
    if not (0 <= a < nq and 0 <= b < nq) or a == b:
        raise HTTPException(400, f"qubit_a and qubit_b must be distinct integers in [0, {nq-1}] for {chip}")

    path = em.shortest_path(graph, a, b)
    if path is None:
        raise HTTPException(400, f"no routable path between qubits {a} and {b} on {chip}")
    dist = len(path) - 1

    qc_bare = QuantumCircuit(nq, 2)
    qc_bare.h(a); qc_bare.cx(a, b)
    v41_pred, _, _ = exact_dwell_cost(qc_bare, backend, graph, coh, [(a, b)], [a, b])

    rec = {"chip": chip, "kind": "bell", "pairs_flat": [a, b], "logical_pairs": [[a, b]],
          "bfs_hop_distance": dist}
    g = graph_for_record(backend, graph, coh, rec)
    g.update({"chip": chip, "kind": "bell", "a": a, "b": b,
              "bfs_hop_distance": dist, "v4_1_prediction": v41_pred})

    seed = abs(hash((chip, a, b))) % (2**31)
    gnn_mean, gnn_std = predict_mc(g, chip, seed)

    return {
        "chip": chip, "qubit_a": a, "qubit_b": b,
        "bfs_hop_distance": dist, "n_nodes": g["n_nodes"], "n_edges": len(g["edges"]),
        "v4_1_prediction": round(v41_pred, 4),
        "gnn_prediction": round(gnn_mean, 4),
        "gnn_uncertainty": round(gnn_std, 4),
        "note": ("Live prediction: real Qiskit SABRE-routed transpile for this exact pair on this "
                "chip, real closed-form v4.1 formula, real GNN forward pass with calibrated "
                "MC-Dropout uncertainty (Entry 085, trained on Kyiv+Sherbrooke+Brisbane+Osaka+"
                "Quebec) -- nothing here is precomputed or approximated. Noise data is from "
                "Qiskit's fake-backend snapshot run through Aer's simulator, not live queued jobs "
                "on physical IBM hardware."),
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QuantumBridge Live Inference API",
    description=("Live circuit reliability prediction for IBM quantum chips: v4.1 closed-form "
                "physics model + graph neural network with calibrated uncertainty, computed on "
                "demand rather than served from a precomputed table. Undergraduate research "
                "project, not a production tool -- see /health for scope and caveats."),
    version="1.0.0",
)

# CORS: the live demo is a static page served from GitHub Pages
# (devendrakhatri981-byte.github.io), a different origin than wherever this
# API ends up hosted, so the browser's cross-origin fetch needs this to
# succeed. This is a public, read-only, rate-limit-free prediction endpoint
# with no auth and no user data involved, so a permissive origin policy is
# an acceptable tradeoff here -- tighten this to the specific GitHub Pages
# origin before adding anything that isn't purely read-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    chip: str = Field(..., description="One of: kyiv, sherbrooke, brisbane, osaka, quebec")
    qubit_a: int = Field(..., ge=0, description="First qubit index")
    qubit_b: int = Field(..., ge=0, description="Second qubit index")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supported_chips": list(BACKENDS.keys()),
        "circuit_types_supported": ["bell", "multi-bell (POST /predict/multi)"],
        "model": "entry085_deploy_params.json (unified 5-chip MC-Dropout GNN)",
        "validation_note": ("Trained and validated against Qiskit fake-backend + Aer simulation "
                            "only, not real IBM hardware. Cusco is the current true zero-shot "
                            "holdout chip (never trained on, by design) -- Entry 088 found this "
                            "5-chip model beats the plain physics formula (v4.1) on Cusco by a wide "
                            "margin (MAE=4.01 vs 9.32, R2=0.784 vs 0.364), continuing a monotonic "
                            "trend as training-chip count grows (3-chip: tied; 4-chip: 44% better; "
                            "5-chip: 57% better on each new held-out chip). Quebec was this "
                            "project's zero-shot holdout chip through Entries 076-078 and was later "
                            "folded into training for this model -- it is now a normal trained chip, "
                            "not a zero-shot one. See the project's research log for full numbers "
                            "before relying on this for a chip outside the five listed above."),
    }


@app.get("/chips")
def chips():
    out = {}
    for chip, cls in BACKENDS.items():
        backend, _, _ = get_backend_ctx(chip)
        out[chip] = {"num_qubits": backend.num_qubits}
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    chip = req.chip.lower().strip()
    if chip == "cusco":
        raise HTTPException(
            400,
            "Unsupported chip 'cusco'. Cusco is this project's current zero-shot holdout chip "
            "(Entries 087-088) -- deliberately excluded from every training run so it stays a "
            "clean test of true generalization. It is not served live here; the research log "
            "documents Entry 088's cold-chip results instead of a live or precomputed lookup.")
    if chip not in BACKENDS:
        raise HTTPException(
            400,
            f"Unsupported chip '{req.chip}'. This deployment is trained on {list(BACKENDS.keys())} "
            "only.")
    t0 = time.time()
    result = live_predict(chip, req.qubit_a, req.qubit_b)
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.post("/predict/breakdown")
def predict_breakdown(req: PredictRequest):
    """Entry 092: same live physics as /predict, but returns WHY, not just
    the number -- which failure channel (gate error, decoherence, single-
    qubit gates, readout) is costing the most probability mass, and which
    specific physical edge in the route is the weak link. Validated to
    reproduce exact_dwell_cost's total exactly (see entry092's own
    self-check); this is exposition of existing physics, not a new model."""
    chip = req.chip.lower().strip()
    if chip == "cusco":
        raise HTTPException(400, "Unsupported chip 'cusco'. Cusco is this project's current "
                            "zero-shot holdout chip (Entries 087-088) and is not served live here.")
    if chip not in BACKENDS:
        raise HTTPException(400, f"Unsupported chip '{req.chip}'. Same five chips as /predict: "
                            f"{list(BACKENDS.keys())}.")
    backend, graph, coh = get_backend_ctx(chip)
    nq = backend.num_qubits
    a, b = req.qubit_a, req.qubit_b
    if not (0 <= a < nq and 0 <= b < nq) or a == b:
        raise HTTPException(400, f"qubit_a and qubit_b must be distinct integers in [0, {nq-1}] for {chip}")

    from qiskit import QuantumCircuit
    qc = QuantumCircuit(nq, 2)
    qc.h(a); qc.cx(a, b)
    breakdown = exact_dwell_cost_decomposed(qc, backend, graph, coh, [(a, b)], [a, b])
    summary = summarize_failure_mode(breakdown)
    return {
        "chip": chip, "qubit_a": a, "qubit_b": b,
        "total_success": round(breakdown["total_success"], 4),
        "channel_survival": {
            "gate_error": round(breakdown["gate_error_survival"], 4),
            "decoherence": round(breakdown["decoherence_survival"], 4),
            "single_qubit_gates": round(breakdown["sq_gate_survival"], 4),
            "readout": round(breakdown["readout_survival"], 4),
        },
        "dominant_failure_mode": summary["dominant_failure_mode"],
        "channel_loss_pts": {k: round(v * 100, 3) for k, v in summary["channel_loss"].items()},
        "weak_link": summary["weak_link"],
        "note": ("Physics-informed decomposition of the same v4.1 closed-form formula used "
                "elsewhere (Entry 024-032), not a separate model -- validated to reproduce "
                "the combined success prediction exactly (Entry 092)."),
    }


# ---------------------------------------------------------------------------
# Multi-bell (Entries 089-105): N disjoint bell pairs run in parallel, success
# = every pair correlated in the same shot. Separate 7-node/5-edge-feature
# model (5-seed ensemble, trained on the 5 chips' own circuits).
# ---------------------------------------------------------------------------

_MULTI = None


def _load_multi():
    global _MULTI
    if _MULTI is None:
        import json
        m = json.load(open(os.path.join(os.path.dirname(__file__), "..", "quantumbridge_data",
                                         "entry105_multibell_deploy.json")))
        _MULTI = {
            "members": [{k: {"W": np.array(v["W"]), "b": np.array(v["b"])} for k, v in p.items()}
                        for p in m["members"]],
            "gmu": np.array(m["glob_mu"]), "gsd": np.array(m["glob_sd"]),
            "ns": {c: (np.array(v[0]), np.array(v[1])) for c, v in m["node_stats"].items()},
            "es": {c: (np.array(v[0]), np.array(v[1])) for c, v in m["edge_stats"].items()},
            "mx_n": m["max_n"], "mx_e": m["max_e"], "n_train": m["n_train"],
        }
    return _MULTI


def _multi_forward(p, M, nf, nm, src, dst, ef, em_, gl):
    mx_n = M["mx_n"]

    def mpr(pm, pu, h):
        msg = relu(lin(pm, np.concatenate([h[src], h[dst], ef], axis=-1))) * em_[:, None]
        agg = np.zeros((mx_n, H)); np.add.at(agg, dst, msg)
        deg = np.zeros(mx_n); np.add.at(deg, dst, em_); deg += 1e-6
        return relu(lin(pu, np.concatenate([h, agg / deg[:, None]], axis=-1))) * nm[:, None]

    h = relu(lin(p["node_embed"], nf)) * nm[:, None]
    h = mpr(p["msg"], p["update"], h)
    h = mpr(p["msg2"], p["update2"], h)
    pooled = (h * nm[:, None]).sum(axis=0) / (nm.sum() + 1e-6)
    x = relu(lin(p["readout1"], np.concatenate([pooled, gl])))
    return float(sigmoid(lin(p["readout2"], x)[0]))


class MultiRequest(BaseModel):
    chip: str = Field(..., description="One of: kyiv, sherbrooke, brisbane, osaka, quebec")
    pairs: list[list[int]] = Field(..., description="2-6 disjoint qubit pairs, e.g. [[0,5],[20,27]]")


@app.post("/predict/multi")
def predict_multi(req: MultiRequest):
    """Entry 105: joint success of N disjoint bell pairs run simultaneously
    (all pairs correlated in the same shot). Returns the independence baseline
    (product of each pair's isolated v4.1 prediction), the GNN prediction
    (5-seed ensemble mean) and the ensemble spread as an uncertainty proxy.
    Trained chips only; trained on 2,080 simulated circuits of 2-6 pairs."""
    chip = req.chip.lower().strip()
    if chip not in BACKENDS:
        raise HTTPException(400, f"Unsupported chip '{req.chip}'. Trained chips: {list(BACKENDS.keys())}.")
    pairs = [tuple(p) for p in req.pairs]
    if not (2 <= len(pairs) <= 6) or any(len(p) != 2 for p in pairs):
        raise HTTPException(400, "pairs must contain 2-6 [a, b] pairs.")
    qs = [q for p in pairs for q in p]
    backend, graph, coh = get_backend_ctx(chip)
    nq = backend.num_qubits
    if len(set(qs)) != len(qs) or any(not (0 <= q < nq) for q in qs):
        raise HTTPException(400, f"all qubits must be distinct integers in [0, {nq-1}] for {chip}")
    dists = []
    for a, b in pairs:
        path = em.shortest_path(graph, a, b)
        if path is None:
            raise HTTPException(400, f"no routable path between qubits {a} and {b} on {chip}")
        dists.append(len(path) - 1)

    from entry090_build_multibell_graphs import graph_for_multibell_record
    t0 = time.time()
    prod, per = 1.0, []
    for a, b in pairs:
        qs_ = QuantumCircuit(nq, 2); qs_.h(a); qs_.cx(a, b)
        pr, _, _ = exact_dwell_cost(qs_, backend, graph, coh, [(a, b)], [a, b])
        per.append(round(pr, 4)); prod *= pr
    g = graph_for_multibell_record(backend, graph, coh, {"logical_pairs": [list(p) for p in pairs]})
    M = _load_multi()
    if g["n_nodes"] > M["mx_n"] or len(g["edges"]) > M["mx_e"]:
        raise HTTPException(400, "routing for this pair set is larger than the model's trained capacity "
                            f"({M['mx_n']} nodes / {M['mx_e']} edges); try pairs with shorter routes.")
    n = len(pairs)
    nf = np.zeros((M["mx_n"], 7), np.float32); nm = np.zeros(M["mx_n"], np.float32)
    for i, f in enumerate(g["nodes"]): nf[i] = f
    nm[:g["n_nodes"]] = 1
    src = np.zeros(M["mx_e"], np.int32); dst = np.zeros(M["mx_e"], np.int32)
    ef = np.zeros((M["mx_e"], 5), np.float32); em_ = np.zeros(M["mx_e"], np.float32)
    for i, e in enumerate(g["edges"]):
        src[i], dst[i] = e[0], e[1]; ef[i] = e[2:]
    em_[:len(g["edges"])] = 1
    nmu, nsd = M["ns"][chip]; emu, esd = M["es"][chip]
    gl = (np.array([float(n), prod]) - M["gmu"]) / M["gsd"]
    preds = [_multi_forward(p, M, (nf - nmu) / nsd, nm, src, dst, (ef - emu) / esd, em_, gl)
             for p in M["members"]]
    return {
        "chip": chip, "pairs": req.pairs, "n_pairs": n, "bfs_hop_distances": dists,
        "per_pair_v41": per,
        "independence_baseline": round(prod, 4),
        "gnn_prediction": round(float(np.mean(preds)), 4),
        "gnn_uncertainty": round(float(np.std(preds)), 4),
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "note": ("Multi-bell (Entry 105): 5-seed GNN ensemble trained on 2,080 simulated multi-pair "
                "circuits from these five chips, normalized with each chip's own training-circuit "
                "statistics (no cold-start caveat for these chips; a NEW chip would need an unlabeled "
                "circuit sample first, Entries 099-101). Validated against Aer simulation only, not "
                "real hardware. Uncertainty is ensemble spread, not MC-dropout."),
    }


@app.get("/recommend")
def recommend_endpoint(min_success: float = 0.0, max_hop: int = None,
                       max_uncertainty: float = None, chips: str = None, top_k: int = 5):
    """Entry 093/097: rank (chip, qubit_a, qubit_b) bell-pair candidates by
    predicted success, built on the precomputed 5-chip lookup table (all
    5 trained chips, including Quebec -- no longer zero-shot as of Entry
    085/097) -- ranking/filtering, not a new model. chips is a
    comma-separated list (e.g. 'kyiv,sherbrooke'); omit for all 5. The
    top result includes a live failure-mode breakdown."""
    chip_list = [c.strip() for c in chips.split(",")] if chips else None
    results = _recommend(min_success=min_success, max_hop=max_hop, max_uncertainty=max_uncertainty,
                         chips=chip_list, top_k=top_k, explain_top=True)
    return {"count": len(results), "candidates": results}
