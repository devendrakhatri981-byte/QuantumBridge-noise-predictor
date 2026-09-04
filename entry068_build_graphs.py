"""QuantumBridge — Entry 068: fold the 804 new Sherbrooke bell-pair circuits
grown in Entry 067 (723 -> 1,527 records, balancing Sherbrooke with Kyiv
~1,510 and Brisbane ~1,621) into the graph dataset (4,886 -> 5,690)."""

import json
import os

import emulator_v3_routing as em
import emulator_v4 as v4
from entry044_build_graphs import BACKENDS, graph_for_record

NEW_RECORDS_PATH = "quantumbridge_data/entry067_sherbrooke_bell_dataset.json"
BASE_GRAPHS_PATH = "quantumbridge_data/entry063_graph_dataset.json"
OUT_PATH = "quantumbridge_data/entry068_graph_dataset.json"


def main():
    base = json.load(open(BASE_GRAPHS_PATH))
    new_records = json.load(open(NEW_RECORDS_PATH))
    print(f"base graphs: {len(base)}, new sherbrooke records: {len(new_records)}")

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
        g.update({"chip": chip, "kind": r["kind"], "topology": r.get("topology", "bell"),
                  "aer_ground_truth": r["aer_ground_truth"],
                  "v4_1_prediction": r["v4_1_prediction"],
                  "bfs_hop_distance": r["bfs_hop_distance"]})
        out.append(g)

        if (i + 1) % 40 == 0 or i + 1 == len(new_records):
            tmp = OUT_PATH + ".tmp"
            json.dump(out, open(tmp, "w"))
            os.replace(tmp, OUT_PATH)
            print(f"  {i+1}/{len(new_records)} new graphs built, saved checkpoint")

    tmp = OUT_PATH + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, OUT_PATH)
    sizes = [g["n_nodes"] for g in out]
    print(f"DONE: {len(out)} graphs -> {OUT_PATH}")
    print(f"max n_nodes={max(sizes)}, max n_edges={max(len(g['edges']) for g in out)}")


if __name__ == "__main__":
    main()
