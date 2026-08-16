"""
QuantumBridge — Entry 038: growing the dataset, oversampled on short routes.

WHY THIS EXISTS
---------------
Entry 037's ML baseline showed a real but modest gain on floor-collapse
cases, limited by having only 12 such examples across 238 circuits. Entry
036 showed floor-collapse doesn't need a long route -- the worst
independence-breakdown gap measured in this project came from a 3-edge
route. That means growing the dataset efficiently means oversampling SHORT
routes (1-15 hops), which are also the cheapest to simulate (no long SWAP
chains), rather than spreading effort evenly across the full hop-distance
spectrum the way Entries 034/035 did.

This reuses entry034_batch_dataset.py's exact methodology (same Aer/MPS
ground truth pipeline, same exact_dwell_cost prediction) with a third,
independent seed, restricted to short/mid bins, and skips the GHZ batch
(Entries 034/035 already cover structural variety there) to spend the
compute budget on more Bell-pair coverage in the region that matters.
"""

import entry034_batch_dataset as base

base.random.seed(2024)
base.OUT_PATH = "quantumbridge_data/entry038_shortroute_dataset.json"
base.BINS = [(1, 1), (2, 3), (4, 7), (8, 15)]
base.PER_BIN = 40
base.N_GHZ = 0

if __name__ == "__main__":
    records = base.load(base.OUT_PATH)
    print(f"resuming with {len(records)} existing records")
    for chip in ("kyiv", "sherbrooke"):
        base.process_chip(chip, records)
    print(f"\nTOTAL records: {len(records)} -> {base.OUT_PATH}")
