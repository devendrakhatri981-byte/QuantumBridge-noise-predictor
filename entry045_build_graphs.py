"""
QuantumBridge — Entry 045b: extend the feature-engineered graph dataset
with the larger star-GHZ circuits from Entry 045a.

Reuses Entry 044's graph_for_record() unchanged (it was already written
generally enough to handle any number of GHZ target legs off one control
qubit, not just the 3-qubit case) -- only the INPUT records change: the
same 2,288 circuits as Entry 044, plus the 29 new 6/7/8-target star-GHZ
circuits from entry045_bigghz_dataset.json. This directly answers "how
many qubits can this model represent" by giving it real training examples
that use far more physical qubits per circuit than anything in the
dataset before.
"""

import json
import os

from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke

import emulator_v3_routing as em
import emulator_v4 as v4
from entry044_build_graphs import BACKENDS, graph_for_record

IN_PATHS = [
    "quantumbridge_data/entry043_combined_dataset.json",
    "quantumbridge_data/entry045_bigghz_dataset.json",
]
OUT_PATH = "quantumbridge_data/entry045_graph_dataset.json"


def main():
    records = []
    for p in IN_PATHS:
        records += json.load(open(p))
    print(f"input records: {len(records)}")

    out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
    done = len(out)
    print(f"resuming with {done}/{len(records)} graphs already built")

    cache = {}
    for i, r in enumerate(records):
        if i < done:
            continue
        chip = r["chip"]
        if chip not in cache:
            backend = BACKENDS[chip]()
            graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
            coh = v4.load_coherence(chip)
            cache[chip] = (backend, graph, coh)
        backend, graph, coh = cache[chip]

        g = graph_for_record(backend, graph, coh, r)
        g.update({"chip": chip, "kind": r["kind"],
                  "aer_ground_truth": r["aer_ground_truth"],
                  "v4_1_prediction": r["v4_1_prediction"],
                  "bfs_hop_distance": r["bfs_hop_distance"]})
        out.append(g)

        if (i + 1) % 100 == 0 or i + 1 == len(records):
            json.dump(out, open(OUT_PATH, "w"))
            print(f"  {i+1}/{len(records)} graphs built, saved checkpoint")

    json.dump(out, open(OUT_PATH, "w"))
    sizes = [g["n_nodes"] for g in out]
    print(f"DONE: {len(out)} graphs -> {OUT_PATH}")
    print(f"max n_nodes={max(sizes)}, max n_edges={max(len(g['edges']) for g in out)}")


if __name__ == "__main__":
    main()
