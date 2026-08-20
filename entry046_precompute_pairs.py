"""
QuantumBridge — Entry 046b: precompute v4.1 + graph features for EVERY
possible Bell-pair circuit on both chips (all C(127,2)=8,001 pairs per
chip, 16,002 total).

WHY THIS EXISTS
---------------
The public demo needs to answer instantly for ANY two qubits the user
picks -- but running Qiskit's real transpiler live in a browser isn't
possible, and approximating the router in JS risks giving people
plausible-looking but fabricated numbers. This script does the honest
thing instead: run the REAL pipeline (real SABRE-routed transpile, the
real v4.1 closed-form formula, the real graph feature extraction) once,
offline, for full coverage of every possible pair -- then the demo is a
lookup, not a guess.

Resumable: checkpoints every 500 records per chip.
"""

import json
import os
import time

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost
from entry044_build_graphs import graph_for_record

BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke}
OUT_PATH = "quantumbridge_data/entry046_all_pairs.json"


def load():
    if os.path.exists(OUT_PATH):
        return json.load(open(OUT_PATH))
    return {"kyiv": [], "sherbrooke": []}


def save(d):
    json.dump(d, open(OUT_PATH, "w"))


def process(chip, d, time_budget_s=150):
    out_list = d[chip]
    backend = BACKENDS[chip]()
    nq = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    coh = v4.load_coherence(chip)

    all_pairs = [(a, b) for a in range(nq) for b in range(a + 1, nq)]
    done = len(out_list)
    t0 = time.time()
    i = done
    while i < len(all_pairs):
        if time.time() - t0 > time_budget_s:
            break
        a, b = all_pairs[i]
        qc = QuantumCircuit(nq, 2)
        qc.h(a); qc.cx(a, b)
        pred, _, _ = exact_dwell_cost(qc, backend, graph, coh, [(a, b)], [a, b])

        rec = {"chip": chip, "kind": "bell", "pairs_flat": [a, b],
              "logical_pairs": [[a, b]],
              "bfs_hop_distance": len(em.shortest_path(graph, a, b)) - 1}
        g = graph_for_record(backend, graph, coh, rec)
        g.update({"chip": chip, "kind": "bell", "a": a, "b": b,
                  "bfs_hop_distance": rec["bfs_hop_distance"],
                  "v4_1_prediction": pred})
        out_list.append(g)
        i += 1

        if i % 500 == 0 or i == len(all_pairs):
            save(d)
            print(f"  {chip}: {i}/{len(all_pairs)} done, checkpoint saved "
                  f"({time.time()-t0:.0f}s elapsed)")
    print(f"{chip}: {len(out_list)}/{len(all_pairs)} total")
    return len(out_list) >= len(all_pairs)


if __name__ == "__main__":
    import sys
    d = load()
    chips = sys.argv[1:] if len(sys.argv) > 1 else ["kyiv", "sherbrooke"]
    all_done = True
    for chip in chips:
        finished = process(chip, d)
        save(d)
        if not finished:
            print("time budget hit -- rerun to continue")
            all_done = False
            break
    if all_done:
        print("ALL PAIRS DONE for requested chips")
