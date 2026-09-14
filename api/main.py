"""
QuantumBridge Live Inference API (v1 productization)

Replaces the static demo's precomputed 24,003-row lookup table with a real
backend: every request does a live Qiskit transpile + SABRE routing +
feature extraction + v4.1 closed-form prediction + Entry 071 GNN inference
(with calibrated MC-Dropout uncertainty), on demand, for any pair of
qubits on any of the three trained chips (Kyiv, Sherbrooke, Brisbane).

This is deliberately scoped, not the full "any circuit, any chip" vision:
- Circuit type: Bell pairs only (H + CX between two qubits), matching what
  the model was trained and validated on. Extending to arbitrary circuits
  is future work (the training data does include GHZ/star/chain circuits,
  but the live feature-extraction path here has only been wired and
  tested for bell pairs).
- Chips: kyiv, sherbrooke, brisbane only -- the three chips the deployed
  model was trained on. Requesting a fourth chip is refused with an
  explicit message pointing at Entry 073's honest zero-shot findings,
  rather than silently degrading.
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
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke, FakeBrisbane

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost
from entry044_build_graphs import graph_for_record

MAX_N, MAX_E = 90, 130
N_NODE_FEATS, N_EDGE_FEATS = 6, 4
H = 16
DROPOUT_RATE = 0.15
T_SAMPLES = 20

BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke, "brisbane": FakeBrisbane}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "quantumbridge_data",
                          "entry071_deploy_params.json")


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
                "MC-Dropout uncertainty (Entry 071) -- nothing here is precomputed or approximated. "
                "Noise data is from Qiskit's fake-backend snapshot run through Aer's simulator, not "
                "live queued jobs on physical IBM hardware."),
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
    chip: str = Field(..., description="One of: kyiv, sherbrooke, brisbane")
    qubit_a: int = Field(..., ge=0, description="First qubit index")
    qubit_b: int = Field(..., ge=0, description="Second qubit index")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supported_chips": list(BACKENDS.keys()),
        "circuit_types_supported": ["bell"],
        "model": "entry071_deploy_params.json (unified 3-chip MC-Dropout GNN)",
        "validation_note": ("Trained and validated against Qiskit fake-backend + Aer simulation "
                            "only, not real IBM hardware. Leave-one-chip-out cross-chip transfer "
                            "and a true zero-shot fourth-chip test (Entry 073) both show real, "
                            "honestly-reported generalization gaps -- see the project's research "
                            "log for full numbers before relying on this for a chip outside the "
                            "three listed above."),
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
    if chip not in BACKENDS:
        raise HTTPException(
            400,
            f"Unsupported chip '{req.chip}'. This deployment is trained on {list(BACKENDS.keys())} "
            "only. Entry 073 tested a genuinely unseen fourth chip and found the model's accuracy "
            "advantage over the simpler v4.1 formula does not clearly survive zero exposure -- "
            "adding a new chip here would need either training data or an explicit, honestly-"
            "labeled degraded mode, not silent extrapolation.")
    t0 = time.time()
    result = live_predict(chip, req.qubit_a, req.qubit_b)
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result
