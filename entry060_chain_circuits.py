"""
QuantumBridge — Entry 060: Sherbrooke chain-topology generation, fixed.

Entry 058 introduced chain-topology GHZ circuits (q0-q1-...-qk-1, entangled
sequentially) as a second large-circuit shape alongside the existing
star topology. It worked for Kyiv but stalled completely on Sherbrooke,
and was shipped Kyiv-only after two full time-budget timeouts.

Root cause (diagnosed here): Entry 058 built each chain by sampling k
qubits uniformly at random from the whole chip and chaining them in that
random order. That means most consecutive pairs in the chain are physically
far apart, so after SWAP routing the circuit's entangling structure is
*not* spatially local -- and the Aer matrix-product-state simulator (which
is fast specifically because it exploits 1D-local entanglement) blows up:
a single k=10 Sherbrooke trial measured at 48s-120s+ in isolation, well
past the old 150s *total* time budget per chip.

Fix: build each chain by walking the real coupling graph itself (random
start qubit, then repeatedly step to a random unvisited physical neighbor,
falling back to nearest unvisited qubit by BFS distance if the current
node is boxed in). This is still a "random" chain -- the start point and
walk choices are seeded and vary -- but it stays spatially local on the
chip, which is what actually made Kyiv's chains fast in Entry 058 (Kyiv's
random samples apparently landed local more often by chance; Sherbrooke's
didn't). Measured speedup on a single Sherbrooke k=10 trial: 48s -> 0.5s.

Same Aer/MPS ground truth methodology as Entries 052/056/058, same
resumable have_per_size top-up logic, output folds into the same
entry058_chain_dataset.json so training scripts don't need to change.
"""

import json
import os
import random
import time
from collections import deque

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost

SHOTS_BY_SIZE = {10: 256, 12: 128, 14: 32}
OUT_PATH = "quantumbridge_data/entry058_chain_dataset.json"  # same file, both entries append
BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke}
SIZES = [10, 12, 14]
PER_SIZE = 4

random.seed(60)


def load(path):
    if os.path.exists(path):
        return json.load(open(path))
    return []


def save(records, path):
    json.dump(records, open(path, "w"), indent=2, default=str)


def done_keys(records):
    return {(r["chip"], r.get("topology", r["kind"]), tuple(sorted(r["pairs_flat"]))) for r in records}


def bfs_dist(graph, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v, _ in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def local_chain(k, rng, graph):
    """Build a k-qubit chain that stays spatially local on the real coupling
    graph: random start, then random-neighbor walk, falling back to nearest
    unvisited qubit by BFS distance when boxed in."""
    start = rng.choice(list(graph.keys()))
    path = [start]
    visited = {start}
    cur = start
    for _ in range(k - 1):
        nbrs = [v for v, _ in graph.get(cur, []) if v not in visited]
        if not nbrs:
            dist = bfs_dist(graph, cur)
            cands = sorted([n for n in dist if n not in visited], key=lambda n: dist[n])
            if not cands:
                break
            cur = cands[0]
        else:
            cur = rng.choice(nbrs)
        path.append(cur)
        visited.add(cur)
    return path


def process(chip, records, time_budget_s=150):
    backend = BACKENDS[chip]()
    nq = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    coh = v4.load_coherence(chip)
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=nm, method="matrix_product_state")
    have = done_keys(records)
    chip_offset = sum(ord(c) for c in chip)
    rng = random.Random(60 + chip_offset)

    have_per_size = {}
    for r in records:
        if r["chip"] == chip and r.get("topology") == "chain":
            k = r.get("n_logical_qubits", len(r["pairs_flat"]))
            have_per_size[k] = have_per_size.get(k, 0) + 1

    t0 = time.time()
    for k in SIZES:
        target_new = PER_SIZE - min(have_per_size.get(k, 0), PER_SIZE)
        made, attempts = 0, 0
        while made < target_new and attempts < target_new * 20:
            if time.time() - t0 > time_budget_s:
                print(f"{chip}: time budget hit mid-size k={k}")
                return False
            attempts += 1
            qubits = local_chain(k, rng, graph)
            if len(qubits) < k:
                continue  # walk got boxed in with no reachable unvisited qubits
            key = (chip, "chain", tuple(sorted(qubits)))
            if key in have:
                continue
            qc = QuantumCircuit(nq, k)
            qc.h(qubits[0])
            for i in range(k - 1):
                qc.cx(qubits[i], qubits[i + 1])
            for i, q in enumerate(qubits):
                qc.measure(q, i)
            t_start = time.time()
            tt = transpile(qc, backend=backend, initial_layout=list(range(nq)),
                          optimization_level=3, seed_transpiler=1)
            ideal = {"0" * k, "1" * k}
            shots = SHOTS_BY_SIZE[k]
            counts = sim.run(tt, shots=shots, seed_simulator=1).result().get_counts()
            tot = sum(counts.values())
            ok = sum(cnt for b, cnt in counts.items() if b.replace(" ", "") in ideal)
            ref = ok / tot

            edges_used = sorted(set(tuple(sorted([tt.find_bit(x).index for x in inst.qubits]))
                                    for inst in tt.data if inst.operation.num_qubits == 2))
            n_2q = sum(1 for i in tt.data if i.operation.num_qubits == 2)

            qc_bare = QuantumCircuit(nq, k)
            qc_bare.h(qubits[0])
            for i in range(k - 1):
                qc_bare.cx(qubits[i], qubits[i + 1])
            logical_pairs = [(qubits[i], qubits[i + 1]) for i in range(k - 1)]
            pred, notes, _ = exact_dwell_cost(qc_bare, backend, graph, coh, logical_pairs, qubits)
            dists = [len(em.shortest_path(graph, qubits[i], qubits[i + 1])) - 1 for i in range(k - 1)]

            rec = {"chip": chip, "kind": "ghz", "topology": "chain", "pairs_flat": qubits,
                  "logical_pairs": [list(p) for p in logical_pairs],
                  "n_logical_qubits": k, "bfs_hop_distance": max(dists),
                  "n_real_edges_used": len(edges_used), "n_2q_gates_total": n_2q,
                  "worst_raw_edge_error_on_bfs_path": None,
                  "aer_ground_truth": ref, "v4_1_prediction": pred,
                  "gap": abs(ref - pred), "runtime_s": round(time.time() - t_start, 2)}
            records.append(rec)
            save(records, OUT_PATH)
            have.add(key)
            made += 1
            print(f"[{chip}] chain k={k:>2} aer={ref*100:6.2f}% pred={pred*100:6.2f}% "
                 f"n2q={n_2q:>3} nodes~{len(set(qubits)|set(v for e in edges_used for v in e))} "
                 f"({rec['runtime_s']}s)")
    return True


if __name__ == "__main__":
    import sys
    records = load(OUT_PATH)
    print(f"resuming with {len(records)} existing records")
    chips = sys.argv[1:] if len(sys.argv) > 1 else ["kyiv", "sherbrooke"]
    for chip in chips:
        finished = process(chip, records)
        if not finished:
            print("rerun to continue")
            break
    print(f"TOTAL: {len(records)} -> {OUT_PATH}")
