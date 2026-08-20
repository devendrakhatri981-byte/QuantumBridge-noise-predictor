"""
QuantumBridge — Entry 045a: bigger star-GHZ circuits to actually stress
the model's node/edge capacity.

WHY THIS EXISTS
---------------
Every GHZ circuit generated so far (Entries 034-040) used exactly 3
logical qubits (1 control + 2 targets). Even after SWAP-routing overhead,
that structure only ever produced routes up to 37 physical nodes -- and
most circuits (bell pairs especially) are far smaller (2-10 nodes). The
GNN's MAX_N=MAX_E=40 padding was sized to fit what existed, not to any
real ceiling on what the chip or the model could represent.

This script generates star-topology GHZ circuits with MANY targets
(4 to 12 logical qubits per circuit, control + up to 11 targets spread
across the chip), which forces real routed graphs to touch far more
physical qubits at once -- a genuine test of "can the model represent a
bigger slice of the chip," not just a bigger padding constant for its
own sake.

Same Aer/MPS ground-truth methodology as every prior entry (noise model
from backend, optimization_level=3, seed_transpiler=1), using the
single-call shots=SHOTS*len(SEEDS) optimization proven statistically
identical and ~18% faster in Entry 042/043.

Resumable: skips any (chip, kind, sorted(pairs_flat)) key already present.
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

SHOTS = 4096
SEEDS = (1,)   # NOTE: reduced from the usual 4-seed/16384-shot standard for
               # this capacity-probe batch only -- an 8-target star GHZ on a
               # 127-qubit device takes 100-250s+ per full-precision Aer/MPS
               # call (large routed circuit, deep bond dimension), which
               # exceeds this sandbox's ~170s per-command ceiling at the
               # usual precision. Single-seed 4096-shot sampling noise is
               # still small relative to the effect sizes here (success
               # probabilities in the 5-70% range); this tradeoff is
               # confined to the handful of large-k circuits in this file,
               # not applied to the main 2,288-circuit dataset.
OUT_PATH = "quantumbridge_data/entry045_bigghz_dataset.json"
BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke}
SIZES = [6, 7, 8]   # logical qubits per GHZ star (1 control + k-1 targets)
PER_SIZE = 1                          # circuits per chip per size

random.seed(45)


def load(path):
    if os.path.exists(path):
        return json.load(open(path))
    return []


def save(records, path):
    json.dump(records, open(path, "w"), indent=2, default=str)


def done_keys(records):
    return {(r["chip"], r["kind"], tuple(sorted(r["pairs_flat"]))) for r in records}


def sample_star(nq, k, rng):
    return rng.sample(range(nq), k)


def process(chip, records):
    backend = BACKENDS[chip]()
    nq = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    coh = v4.load_coherence(chip)
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=nm, method="matrix_product_state")
    have = done_keys(records)
    chip_offset = sum(ord(c) for c in chip)
    rng = random.Random(45 + chip_offset)

    # count how many we already have per size for this chip, so resuming
    # tops up to PER_SIZE instead of blindly adding PER_SIZE more each run
    have_per_size = {}
    for r in records:
        if r["chip"] == chip and r["kind"] == "ghz":
            have_per_size[r.get("n_logical_qubits", len(r["pairs_flat"]))] = \
                have_per_size.get(r.get("n_logical_qubits", len(r["pairs_flat"])), 0) + 1

    total = len(SIZES) * PER_SIZE
    done_now = sum(min(have_per_size.get(k, 0), PER_SIZE) for k in SIZES)
    attempts_by_size = {k: 0 for k in SIZES}
    for k in SIZES:
        target_new = PER_SIZE - min(have_per_size.get(k, 0), PER_SIZE)
        made = 0
        while made < target_new and attempts_by_size[k] < target_new * 20:
            attempts_by_size[k] += 1
            qubits = sample_star(nq, k, rng)
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
            t0 = time.time()
            tt = transpile(qc, backend=backend, initial_layout=list(range(nq)),
                           optimization_level=3, seed_transpiler=1)
            ideal = {"0" * k, "1" * k}
            counts = sim.run(tt, shots=SHOTS * len(SEEDS),
                             seed_simulator=SEEDS[0]).result().get_counts()
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
            pred, notes, _ = exact_dwell_cost(qc_bare, backend, graph, coh,
                                              logical_pairs, qubits)

            dists = [len(em.shortest_path(graph, c, t)) - 1 for t in targets]
            rec = {"chip": chip, "kind": "ghz", "pairs_flat": qubits,
                  "logical_pairs": [list(p) for p in logical_pairs],
                  "n_logical_qubits": k,
                  "bfs_hop_distance": max(dists),
                  "n_real_edges_used": len(edges_used), "n_2q_gates_total": n_2q,
                  "worst_raw_edge_error_on_bfs_path": None,
                  "aer_ground_truth": ref, "v4_1_prediction": pred,
                  "gap": abs(ref - pred), "runtime_s": round(time.time() - t0, 2)}
            records.append(rec)
            save(records, OUT_PATH)
            have.add(key)
            done_now += 1
            made += 1
            print(f"[{chip} {done_now}/{total}] k={k:>2} ctrl={c:>3} "
                  f"aer={ref*100:6.2f}% pred={pred*100:6.2f}% "
                  f"gap={rec['gap']*100:5.2f} n2q={n_2q:>3} ({rec['runtime_s']}s)")


if __name__ == "__main__":
    import sys
    records = load(OUT_PATH)
    print(f"resuming with {len(records)} existing records")
    chips = sys.argv[1:] if len(sys.argv) > 1 else ["kyiv", "sherbrooke"]
    for chip in chips:
        process(chip, records)
    print(f"\nTOTAL records: {len(records)} -> {OUT_PATH}")
