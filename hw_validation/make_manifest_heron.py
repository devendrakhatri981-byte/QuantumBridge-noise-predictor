"""
Hardware-validation manifest for a LIVE Heron device (chip files from pull_live_heron.py).
Predictions frozen before the run: v4.1 (primary), v4.1 on the stale fake snapshot (for comparison),
and the Eagle-trained GNN with own-stats normalization (zero-shot, secondary). Routes crossing dead
edges (err>=0.5) are excluded. usage: python3 hw_validation/make_manifest_heron.py fez_live [N_SINGLE N_MULTI]
"""
import json, random, sys, time
import numpy as np
sys.path.insert(0, ".")
import entry089_grow_multibell as g89
import api.main as M
import emulator_v3_routing as em
from qiskit import QuantumCircuit
from exact_dwell_routing import exact_dwell_cost, route_with_explicit_swaps

chip = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 20; NM = int(sys.argv[3]) if len(sys.argv) > 3 else 15
M.BACKENDS.update({k: v for k, v in g89.BACKENDS.items() if k not in M.BACKENDS})
backend, graph, coh = M.get_backend_ctx(chip); nq = backend.num_qubits
snap = chip.replace("_live", ""); sb, sg, sc = M.get_backend_ctx(snap)
cal = json.load(open(f"quantumbridge_data/offline_calibration_{chip}_full.json"))["two_qubit_gate_errors"]
dead = {tuple(sorted(x["qubits"])) for x in cal if (x["gate_error"] or 0) >= 0.5}
dead_q = set(json.load(open(f"quantumbridge_data/coherence_{chip}.json")).get("missing_qubits_filled_with_median", []))
rng = random.Random(2026)


def clean(pairs):
    qc = QuantumCircuit(nq, 2 * len(pairs))
    for a, b in pairs: qc.h(a); qc.cx(a, b)
    t = route_with_explicit_swaps(qc, backend)
    used = {t.find_bit(q).index for i in t.data if i.operation.num_qubits == 2 for q in i.qubits}
    return not (used & dead_q) and not any(tuple(sorted(t.find_bit(q).index for q in i.qubits)) in dead
                   for i in t.data if i.operation.num_qubits == 2)


def v41(pairs, b, gr, ch):
    prod = 1.0
    for a, c in pairs:
        q = QuantumCircuit(nq, 2); q.h(a); q.cx(a, c)
        prod *= exact_dwell_cost(q, b, gr, ch, [(a, c)], [a, c])[0]
    return prod


def hops(a, b):
    p = em.shortest_path(graph, a, b); return None if p is None else len(p) - 1


circuits = []
for h in [1, 2, 3, 4, 5, 6]:                       # singles spread over hop distance
    want = NS // 6 + (1 if h <= NS % 6 else 0); got = 0
    while got < want:
        a, b = rng.sample(range(nq), 2)
        if hops(a, b) == h and clean([(a, b)]):
            circuits.append({"id": f"single_{len(circuits):02d}", "kind": "bell", "pairs": [[a, b]], "hops": [h]}); got += 1
n_multi = {2: NM // 2, 3: NM - NM // 2}
for n, want in n_multi.items():
    got = 0
    while got < want:
        qs = rng.sample(range(nq), 2 * n); prs = [(qs[2 * i], qs[2 * i + 1]) for i in range(n)]
        hs = [hops(a, b) for a, b in prs]
        if None in hs or max(hs) > 3 or not clean(prs): continue
        circuits.append({"id": f"multi{n}_{len(circuits):02d}", "kind": "multibell", "pairs": [list(p) for p in prs], "hops": hs}); got += 1

# zero-shot GNN own-stats from unlabeled random pairs on the live chip
gs = []
while len(gs) < 100:
    a, b = rng.sample(range(nq), 2)
    if hops(a, b) and hops(a, b) <= 8:
        r = {"chip": chip, "kind": "bell", "pairs_flat": [a, b], "logical_pairs": [[a, b]], "bfs_hop_distance": hops(a, b)}
        gs.append(M.graph_for_record(backend, graph, coh, r))
nf = np.concatenate([np.array(x["nodes"]) for x in gs]); ef = np.concatenate([np.array(x["edges"])[:, 2:] for x in gs])
M.NODE_STATS[chip] = (nf.mean(0), nf.std(0) + 1e-6); M.EDGE_STATS[chip] = (ef.mean(0), ef.std(0) + 1e-6)

for c in circuits:
    prs = [tuple(p) for p in c["pairs"]]
    c["v41"] = v41(prs, backend, graph, coh); c["v41_stale_snapshot"] = v41(prs, sb, sg, sc)
    if c["kind"] == "bell":
        a, b = prs[0]
        r = {"chip": chip, "kind": "bell", "pairs_flat": [a, b], "logical_pairs": [[a, b]], "bfs_hop_distance": c["hops"][0]}
        gr = M.graph_for_record(backend, graph, coh, r); gr.update({"v4_1_prediction": c["v41"], "bfs_hop_distance": c["hops"][0]})
        c["gnn"] = M.predict_mc(gr, chip, 1)[0]
    else:
        c["gnn"] = None                                # multi-bell GNN is Eagle-only; v4.1 product is the prediction
json.dump({"chip_model": chip, "sim_chip": snap, "num_qubits": nq, "created": time.strftime("%Y-%m-%d %H:%M"),
           "shots": 8192, "dead_edges_excluded": sorted(dead), "dead_qubits_excluded": sorted(dead_q),
           "note": "predictions frozen before hardware run; v41 uses the live calibration pulled just before",
           "circuits": circuits}, open(f"hw_validation/manifest_{chip}.json", "w"), indent=1)
v = np.array([c["v41"] for c in circuits]); w = np.array([c["v41_stale_snapshot"] for c in circuits])
print(len(circuits), "circuits | v4.1 live mean %.3f, stale mean %.3f, mean |live-stale| %.3f" % (v.mean(), w.mean(), np.abs(v - w).mean()))
