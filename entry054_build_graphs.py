"""QuantumBridge — Entry 054: fold the 482 new deduped bell-pair circuits
(from the 932->1,592 growth of entry042_parallel_dataset.json) into the
Entry 053 graph dataset (2,329 -> 2,811), reusing graph_for_record unchanged."""

import json
import os

import emulator_v3_routing as em
import emulator_v4 as v4
from entry044_build_graphs import BACKENDS, graph_for_record

NEW_RECORDS_PATH = "quantumbridge_data/entry054_new_bell_only.json"
BASE_GRAPHS_PATH = "quantumbridge_data/entry053_graph_dataset.json"
OUT_PATH = "quantumbridge_data/entry054_graph_dataset.json"


def main():
    base = json.load(open(BASE_GRAPHS_PATH))
    new_records = json.load(open(NEW_RECORDS_PATH))
    print(f"base graphs: {len(base)}, new records to add: {len(new_records)}")

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
        g.update({"chip": chip, "kind": r["kind"],
                  "aer_ground_truth": r["aer_ground_truth"],
                  "v4_1_prediction": r["v4_1_prediction"],
                  "bfs_hop_distance": r["bfs_hop_distance"]})
        out.append(g)

        if (i + 1) % 100 == 0 or i + 1 == len(new_records):
            json.dump(out, open(OUT_PATH, "w"))
            print(f"  {i+1}/{len(new_records)} new graphs built, saved checkpoint")

    json.dump(out, open(OUT_PATH, "w"))
    sizes = [g["n_nodes"] for g in out]
    print(f"DONE: {len(out)} graphs -> {OUT_PATH}")
    print(f"max n_nodes={max(sizes)}, max n_edges={max(len(g['edges']) for g in out)}")


if __name__ == "__main__":
    main()
