"""
QuantumBridge — Entry 034: the batch-circuit dataset.

WHY THIS EXISTS
---------------
Every entry from 022 through 033 debugged the model against 5 hand-picked
circuits per chip (adjacent / near / mid / far / one local GHZ). That was
the right way to chase individual bugs, but it also means every finding so
far rests on 10 total data points. This script does the opposite: it
samples MANY qubit pairs per chip, stratified across the real route-length
spectrum (BFS hop distance on the actual coupling graph), runs each one
through the same Aer/MPS ground-truth pipeline used since Entry 022, and
records the v4.1 (exact-dwell, Entry 032) prediction alongside it. Two
purposes:

  1. Find systematic failure patterns by FREQUENCY and MAGNITUDE (does the
     gap actually grow with hop count, or was that impression from 5
     circuits an accident?) instead of one circuit at a time.
  2. Lay the foundation dataset for the ML/GNN phase: circuit + calibration
     features -> measured success probability.

Kyiv and Sherbrooke only (both uniform ecr-only basis). Cairo stays
excluded from full-circuit Aer/MPS validation, per Entry 022's original
finding and this session's direct confirmation that forcing an ecr-only
basis disconnects Cairo's coupling graph.

RESUMABLE BY DESIGN
--------------------
Each circuit's result is appended to the output JSON immediately after it
finishes, and the script skips any (chip, kind, qubits) key already present
on startup -- a long batch can be safely re-run in chunks.
"""

import json
import os
import random
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost

SHOTS = 4096
SEEDS = (1, 2, 3, 4)
OUT_PATH = "quantumbridge_data/entry034_batch_dataset.json"

# hop-distance bins (inclusive) we stratify Bell-pair sampling across, and
# how many pairs to draw per chip from each bin.
BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 25), (26, 999)]
PER_BIN = 10          # -> up to 60 bell circuits per chip
N_GHZ = 10             # 3-qubit ghz circuits per chip, random spread

BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke}

random.seed(7)


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save(records, path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)


def done_keys(records):
    return {(r["chip"], r["kind"], tuple(r["pairs_flat"])) for r in records}


def sample_bell_pairs(graph, nq, per_bin, bins):
    """Stratified sample of (a, b) pairs by BFS hop distance."""
    chosen = []
    seen = set()
    for lo, hi in bins:
        got = 0
        attempts = 0
        while got < per_bin and attempts < per_bin * 60:
            attempts += 1
            a, b = random.randrange(nq), random.randrange(nq)
            if a == b or (a, b) in seen or (b, a) in seen:
                continue
            path = em.shortest_path(graph, a, b)
            if path is None:
                continue
            dist = len(path) - 1
            if lo <= dist <= hi:
                chosen.append((a, b, dist))
                seen.add((a, b))
                got += 1
    return chosen


def sample_ghz_triples(graph, nq, n, seen_bell):
    chosen = []
    attempts = 0
    while len(chosen) < n and attempts < n * 60:
        attempts += 1
        c, q1, q2 = random.sample(range(nq), 3)
        key = (c, q1, q2)
        if key in seen_bell:
            continue
        d1 = len(em.shortest_path(graph, c, q1)) - 1
        d2 = len(em.shortest_path(graph, c, q2)) - 1
        chosen.append((c, q1, q2, d1, d2))
        seen_bell.add(key)
    return chosen


def ground_truth(sim, backend, qc, meas):
    t = transpile(qc, backend=backend, initial_layout=list(range(backend.num_qubits)),
                 optimization_level=3, seed_transpiler=1)
    ideal = {"0" * len(meas), "1" * len(meas)}
    vals = []
    for sd in SEEDS:
        counts = sim.run(t, shots=SHOTS, seed_simulator=sd).result().get_counts()
        tot = sum(counts.values())
        ok = sum(c for b, c in counts.items() if b.replace(" ", "") in ideal)
        vals.append(ok / tot)
    edges_used = sorted(set(tuple(sorted([t.find_bit(x).index for x in inst.qubits]))
                            for inst in t.data if inst.operation.num_qubits == 2))
    n_2q_total = sum(1 for i in t.data if i.operation.num_qubits == 2)
    return float(np.mean(vals)), len(edges_used), n_2q_total


def process_chip(chip, records):
    print(f"\n=== {chip} ===")
    backend = BACKENDS[chip]()
    nq = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    coh = v4.load_coherence(chip)
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=nm, method="matrix_product_state")

    have = done_keys(records)

    bell_pairs = sample_bell_pairs(graph, nq, PER_BIN, BINS)
    seen_flat = {(a, b) for a, b, _ in bell_pairs}
    ghz_triples = sample_ghz_triples(graph, nq, N_GHZ, set(seen_flat))

    total = len(bell_pairs) + len(ghz_triples)
    done_now = 0
    t_start = time.time()

    for a, b, dist in bell_pairs:
        key = (chip, "bell", (a, b))
        if key in have:
            done_now += 1
            continue
        qc = QuantumCircuit(nq, 2)
        qc.h(a); qc.cx(a, b)
        qc.measure(a, 0); qc.measure(b, 1)
        t0 = time.time()
        ref, n_edges, n_2q = ground_truth(sim, backend, qc, [a, b])
        qc_bare = QuantumCircuit(nq, 2)
        qc_bare.h(a); qc_bare.cx(a, b)
        pred, notes, _ = exact_dwell_cost(qc_bare, backend, graph, coh, [(a, b)], [a, b])
        raw_edge_err, worst_t1 = None, None
        try:
            path = em.shortest_path(graph, a, b)
            raw_edge_err = max(em.edge_error(graph, path[i], path[i + 1])
                               for i in range(len(path) - 1))
            worst_t1 = min(coh["T1"].get(q, 1e9) for q in path)
        except Exception:
            pass
        rec = {"chip": chip, "kind": "bell", "pairs_flat": [a, b],
              "logical_pairs": [[a, b]], "bfs_hop_distance": dist,
              "n_real_edges_used": n_edges, "n_2q_gates_total": n_2q,
              "worst_raw_edge_error_on_bfs_path": raw_edge_err,
              "worst_t1_on_bfs_path": worst_t1,
              "aer_ground_truth": ref, "v4_1_prediction": pred,
              "gap": abs(ref - pred), "runtime_s": round(time.time() - t0, 2)}
        records.append(rec)
        save(records, OUT_PATH)
        done_now += 1
        print(f"[{done_now}/{total}] bell {a}->{b} dist={dist:>3}  "
              f"aer={ref*100:6.2f}%  pred={pred*100:6.2f}%  "
              f"gap={rec['gap']*100:5.2f}  ({rec['runtime_s']}s)")

    for c, q1, q2, d1, d2 in ghz_triples:
        key = (chip, "ghz", (c, q1, q2))
        if key in have:
            done_now += 1
            continue
        qc = QuantumCircuit(nq, 3)
        qc.h(c); qc.cx(c, q1); qc.cx(c, q2)
        qc.measure(c, 0); qc.measure(q1, 1); qc.measure(q2, 2)
        t0 = time.time()
        ref, n_edges, n_2q = ground_truth(sim, backend, qc, [c, q1, q2])
        qc_bare = QuantumCircuit(nq, 3)
        qc_bare.h(c); qc_bare.cx(c, q1); qc_bare.cx(c, q2)
        pred, notes, _ = exact_dwell_cost(qc_bare, backend, graph, coh,
                                          [(c, q1), (c, q2)], [c, q1, q2])
        rec = {"chip": chip, "kind": "ghz", "pairs_flat": [c, q1, q2],
              "logical_pairs": [[c, q1], [c, q2]],
              "bfs_hop_distance": max(d1, d2), "n_real_edges_used": n_edges,
              "n_2q_gates_total": n_2q, "worst_raw_edge_error_on_bfs_path": None,
              "aer_ground_truth": ref, "v4_1_prediction": pred,
              "gap": abs(ref - pred), "runtime_s": round(time.time() - t0, 2)}
        records.append(rec)
        save(records, OUT_PATH)
        done_now += 1
        print(f"[{done_now}/{total}] ghz  {c}->({q1},{q2}) dist={max(d1,d2):>3}  "
              f"aer={ref*100:6.2f}%  pred={pred*100:6.2f}%  "
              f"gap={rec['gap']*100:5.2f}  ({rec['runtime_s']}s)")

    print(f"{chip}: {done_now}/{total} done, {time.time()-t_start:.0f}s this run")


if __name__ == "__main__":
    records = load(OUT_PATH)
    print(f"resuming with {len(records)} existing records")
    for chip in ("kyiv", "sherbrooke"):
        process_chip(chip, records)
    print(f"\nTOTAL records: {len(records)} -> {OUT_PATH}")
