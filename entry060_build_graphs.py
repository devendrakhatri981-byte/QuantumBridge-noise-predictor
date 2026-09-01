"""QuantumBridge — Entry 060: fold the 20 new chain-topology circuits
(entry060_chain_circuits.py fixed the MPS-blowup bug that stalled Sherbrooke
chain generation in Entry 058) into the Entry 058 graph dataset
(3,221 -> 3,241). The first 4 records in entry058_chain_dataset.json are the
original Kyiv-only Entry 058 additions (already folded into
entry058_graph_dataset.json); records[4:] are new from this entry."""

import json
import os

import emulator_v3_routing as em
import emulator_v4 as v4
from entry044_build_graphs import BACKENDS, graph_for_record

ALL_RECORDS_PATH = "quantumbridge_data/entry058_chain_dataset.json"
BASE_GRAPHS_PATH = "quantumbridge_data/entry058_graph_dataset.json"
OUT_PATH = "quantumbridge_data/entry060_graph_dataset.json"
ALREADY_FOLDED = 4  # entry058's original Kyiv-only records


def main():
    base = json.load(open(BASE_GRAPHS_PATH))
    all_records = json.load(open(ALL_RECORDS_PATH))
    new_records = all_records[ALREADY_FOLDED:]
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
        g.update({"chip": chip, "kind": r["kind"], "topology": r.get("topology", "star"),
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
