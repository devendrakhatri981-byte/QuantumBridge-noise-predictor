"""
QuantumBridge — Entry 061: star + chain large circuits for Brisbane, the
third chip. Same methodology as Entry 056 (star) and Entry 060 (chain,
using the spatially-local chain-walk fix so Brisbane doesn't hit the same
MPS-blowup stall Sherbrooke did in Entry 058)."""

import json
import os
import random
import time
from collections import deque

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost

SHOTS_BY_SIZE = {10: 256, 12: 128, 14: 32}
STAR_OUT = "quantumbridge_data/entry061_brisbane_star_dataset.json"
CHAIN_OUT = "quantumbridge_data/entry061_brisbane_chain_dataset.json"
BACKENDS = {"brisbane": FakeBrisbane}
SIZES = [10, 12, 14]
PER_SIZE = 4


def load(path):
    if os.path.exists(path):
        return json.load(open(path))
    return []


def save(records, path):
    # Atomic write -- see entry061_grow_brisbane.py's _atomic_save for why
    # (a SIGKILL mid-json.dump corrupted a checkpoint file and destroyed
    # prior progress, not just the in-progress write).
    tmp = path + ".tmp"
    json.dump(records, open(tmp, "w"), indent=2, default=str)
    os.replace(tmp, path)


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


def done_keys(records, topo):
    return {(r["chip"], r.get("topology", r["kind"]), tuple(sorted(r["pairs_flat"]))) for r in records}


def local_star_targets(control, k, rng, graph):
    """Pick k-1 targets physically near the control qubit (nearest by BFS
    distance on the real coupling graph, with light randomization among
    near-tied candidates) instead of uniformly at random from the whole
    chip. A star's hub-and-spoke entanglement is not spatially local in
    Hilbert space regardless of qubit ordering -- unlike the chain fix
    (Entry 060), reordering can't fix this. But keeping the targets
    physically close to the control keeps the *routed* circuit shallow
    (few/no SWAPs needed), which is what actually determines MPS cost.
    Measured effect on Brisbane: a k=10 star with fully random targets
    didn't finish Aer/MPS simulation in 170s; with nearby targets, 0.5s."""
    dist = bfs_dist(graph, control)
    cands = sorted([n for n in dist if n != control], key=lambda n: dist[n])
    # take a little extra than needed and shuffle within that pool so we're
    # not always picking the literal single nearest set every time
    pool = cands[:max(k - 1, min(len(cands), (k - 1) * 3))]
    rng.shuffle(pool)
    return pool[:k - 1]


def process_star(chip, backend, nq, graph, coh, sim, records, seed_base, time_budget_s=150):
    have = done_keys(records, "star")
    rng = random.Random(seed_base)
    have_per_size = {}
    for r in records:
        if r["chip"] == chip:
            k = r.get("n_logical_qubits", len(r["pairs_flat"]))
            have_per_size[k] = have_per_size.get(k, 0) + 1

    t0 = time.time()
    for k in SIZES:
        target_new = PER_SIZE - min(have_per_size.get(k, 0), PER_SIZE)
        made, attempts = 0, 0
        while made < target_new and attempts < target_new * 20:
            if time.time() - t0 > time_budget_s:
                print(f"[star] {chip}: time budget hit mid-size k={k}")
                return False
            attempts += 1
            c = rng.choice(list(graph.keys()))
            targets = local_star_targets(c, k, rng, graph)
            if len(targets) < k - 1:
                continue
            qubits = [c] + targets
            key = (chip, "ghz", tuple(sorted(qubits)))
            if key in have:
                continue
            qc = QuantumCircuit(nq, k)
            qc.h(c)
            for t in targets:
                qc.cx(c, t)
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
            qc_bare.h(c)
            for t in targets:
                qc_bare.cx(c, t)
            logical_pairs = [(c, t) for t in targets]
            pred, notes, _ = exact_dwell_cost(qc_bare, backend, graph, coh, logical_pairs, qubits)
            dists = [len(em.shortest_path(graph, c, t)) - 1 for t in targets]

            rec = {"chip": chip, "kind": "ghz", "pairs_flat": qubits,
                  "logical_pairs": [list(p) for p in logical_pairs],
                  "n_logical_qubits": k, "bfs_hop_distance": max(dists),
                  "n_real_edges_used": len(edges_used), "n_2q_gates_total": n_2q,
                  "worst_raw_edge_error_on_bfs_path": None,
                  "aer_ground_truth": ref, "v4_1_prediction": pred,
                  "gap": abs(ref - pred), "runtime_s": round(time.time() - t_start, 2)}
            records.append(rec)
            save(records, STAR_OUT)
            have.add(key)
            made += 1
            print(f"[star {chip}] k={k:>2} aer={ref*100:6.2f}% pred={pred*100:6.2f}% "
                 f"n2q={n_2q:>3} ({rec['runtime_s']}s)")
    return True


def process_chain(chip, backend, nq, graph, coh, sim, records, seed_base, time_budget_s=150):
    have = done_keys(records, "chain")
    rng = random.Random(seed_base)
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
                print(f"[chain] {chip}: time budget hit mid-size k={k}")
                return False
            attempts += 1
            qubits = local_chain(k, rng, graph)
            if len(qubits) < k:
                continue
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
            save(records, CHAIN_OUT)
            have.add(key)
            made += 1
            print(f"[chain {chip}] k={k:>2} aer={ref*100:6.2f}% pred={pred*100:6.2f}% "
                 f"n2q={n_2q:>3} ({rec['runtime_s']}s)")
    return True


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    chip = "brisbane"
    backend = BACKENDS[chip]()
    nq = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    coh = v4.load_coherence(chip)
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=nm, method="matrix_product_state")

    if mode in ("star", "both"):
        records = load(STAR_OUT)
        print(f"star: resuming with {len(records)} existing records")
        finished = process_star(chip, backend, nq, graph, coh, sim, records, seed_base=161)
        print(f"star TOTAL: {len(records)} -> {STAR_OUT}" + ("" if finished else " (rerun to continue)"))

    if mode in ("chain", "both"):
        records = load(CHAIN_OUT)
        print(f"chain: resuming with {len(records)} existing records")
        finished = process_chain(chip, backend, nq, graph, coh, sim, records, seed_base=261)
        print(f"chain TOTAL: {len(records)} -> {CHAIN_OUT}" + ("" if finished else " (rerun to continue)"))
