"""
QuantumBridge — Entry 042 continued, sequential fallback.

The multiprocessing.Pool approach in entry042_grow_parallel.py worked for
several rounds (782 circuits added) then started reliably hanging for
unclear reasons (possibly resource contention from repeated interrupted
spawns in this sandbox). Rather than keep debugging process pools, this
reuses the exact same run_bell() function -- same precision, same
methodology -- single-threaded, appending to the same output file. Same
proven approach as Entries 034/035/038/039/040, just continuing to feed
the growing entry042_parallel_dataset.json.
"""

import json
import os
import random
import sys

import entry042_grow_parallel as m

TARGET_PER_BIN = 40


def main(target_per_bin=TARGET_PER_BIN, seed=None):
    seed = seed if seed is not None else random.randrange(1_000_000)
    records = m.load()
    existing = {(r["chip"], r["pairs_flat"][0], r["pairs_flat"][1]) for r in records}
    print(f"resuming with {len(records)} existing records, seed={seed}")

    tasks = []
    for chip in ("kyiv", "sherbrooke"):
        backend = m.BACKENDS[chip]()
        graph = m.em.build_connectivity_graph(m.em.load_calibration(chip), chip)
        tasks += m.sample_tasks(chip, backend.num_qubits, graph, target_per_bin, seed, existing)
    random.Random(seed).shuffle(tasks)
    print(f"queued {len(tasks)} new circuits (sequential)")

    for i, task in enumerate(tasks):
        rec = m.run_bell(task)
        records.append(rec)
        if (i + 1) % 10 == 0:
            json.dump(records, open(m.OUT_PATH, "w"))
            print(f"  {i+1}/{len(tasks)} done, total so far {len(records)}")

    json.dump(records, open(m.OUT_PATH, "w"))
    print(f"DONE this run: {len(tasks)} attempted, {len(records)} total -> {m.OUT_PATH}")


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_PER_BIN
    main(target_per_bin=target)
