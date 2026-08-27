"""
QuantumBridge — Entry 052: push real circuit size from ~42 nodes (Entry 045's
max) toward 60-70 physical qubits, per updated project direction.

Same star-GHZ approach as Entry 045 (1 control + k-1 targets), extended to
larger k (10, 12, 14 logical qubits). Single-seed reduced-shot sampling,
same documented precision tradeoff as Entry 045's large-circuit batch --
these are expensive circuits (deep routing, many SWAPs) that would exceed
the sandbox's per-command time ceiling at full 4-seed precision.

Resumable via the same have_per_size top-up logic as Entry 045.
"""

import json
import os
import random
import time

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost

# NOTE: k=10-14 star-GHZ circuits are far more expensive for Aer's MPS
# simulator than anything generated before -- a k=10 star measured at
# ~0.25s PER SHOT (227 real 2-qubit gates from routing+entangling a
# 10-way star), so the usual 4096 shots would take ~17 minutes, well
# beyond a single command in this sandbox. Shots are reduced per size
# to fit one sim.run() call in the time budget -- a real, larger
# precision tradeoff than earlier entries, documented here rather than
# silently applied.
SHOTS_BY_SIZE = {10: 256, 12: 128, 14: 32}
OUT_PATH = "quantumbridge_data/entry052_biggerghz_dataset.json"
BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke}
SIZES = [10, 12, 14]
PER_SIZE = 2

random.seed(52)


def load(path):
    if os.path.exists(path):
        return json.load(open(path))
    return []


def save(records, path):
    json.dump(records, open(path, "w"), indent=2, default=str)


def done_keys(records):
    return {(r["chip"], r["kind"], tuple(sorted(r["pairs_flat"]))) for r in records}


def process(chip, records, time_budget_s=150):
    backend = BACKENDS[chip]()
    nq = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    coh = v4.load_coherence(chip)
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=nm, method="matrix_product_state")
    have = done_keys(records)
    chip_offset = sum(ord(c) for c in chip)
    rng = random.Random(52 + chip_offset)

    have_per_size = {}
    for r in records:
        if r["chip"] == chip and r["kind"] == "ghz":
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
            qubits = rng.sample(range(nq), k)
            key = (chip, "ghz", tuple(sorted(qubits)))
            if key in have:
                continue
            c = qubits[0]
            targets = qubits[1:]
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
            save(records, OUT_PATH)
            have.add(key)
            made += 1
            print(f"[{chip}] k={k:>2} ctrl={c:>3} aer={ref*100:6.2f}% pred={pred*100:6.2f}% "
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
