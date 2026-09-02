"""QuantumBridge — Entry 061: fold Brisbane's new circuits (740 bell pairs +
12 chain-topology large circuits; star-topology large circuits skipped this
round -- see research log for why) into the Entry 060 graph dataset
(3,241 -> 3,993), reusing graph_for_record unchanged. First non-Kyiv/
non-Sherbrooke data in the training pipeline."""

import json
import os

import emulator_v3_routing as em
import emulator_v4 as v4
from entry044_build_graphs import BACKENDS, graph_for_record

BELL_PATH = "quantumbridge_data/entry061_brisbane_bell_dataset.json"
CHAIN_PATH = "quantumbridge_data/entry061_brisbane_chain_dataset.json"
BASE_GRAPHS_PATH = "quantumbridge_data/entry060_graph_dataset.json"
OUT_PATH = "quantumbridge_data/entry061_graph_dataset.json"


def main():
    base = json.load(open(BASE_GRAPHS_PATH))
    bell = json.load(open(BELL_PATH))
    chain = json.load(open(CHAIN_PATH))
    new_records = bell + chain
    print(f"base graphs: {len(base)}, new records to add: {len(new_records)} "
          f"({len(bell)} bell + {len(chain)} chain)")

    out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else list(base)
    done = len(out) - len(base)
    if done < 0:
        done = 0
        out = list(base)
    print(f"resuming with {done}/{len(new_records)} new graphs already built")

    cache = {}
    for i, r in enumerate(new_records):
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
        g.update({"chip": chip, "kind": r["kind"], "topology": r.get("topology", "star"),
                  "aer_ground_truth": r["aer_ground_truth"],
                  "v4_1_prediction": r["v4_1_prediction"],
                  "bfs_hop_distance": r["bfs_hop_distance"]})
        out.append(g)

        if (i + 1) % 20 == 0 or i + 1 == len(new_records):
            json.dump(out, open(OUT_PATH, "w"))
            print(f"  {i+1}/{len(new_records)} new graphs built, saved checkpoint")

    json.dump(out, open(OUT_PATH, "w"))
    sizes = [g["n_nodes"] for g in out]
    print(f"DONE: {len(out)} graphs -> {OUT_PATH}")
    print(f"max n_nodes={max(sizes)}, max n_edges={max(len(g['edges']) for g in out)}")


if __name__ == "__main__":
    main()
