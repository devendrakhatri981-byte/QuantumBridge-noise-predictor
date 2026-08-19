"""QuantumBridge — Entry 040a: fifth growth batch, short/mid bins (fast, high-volume)."""

import entry034_batch_dataset as base

base.random.seed(777)
base.OUT_PATH = "quantumbridge_data/entry040_short_dataset.json"
base.BINS = [(1, 1), (2, 3), (4, 7), (8, 15)]
base.PER_BIN = 60
base.N_GHZ = 0

if __name__ == "__main__":
    records = base.load(base.OUT_PATH)
    print(f"resuming with {len(records)} existing records")
    for chip in ("kyiv", "sherbrooke"):
        base.process_chip(chip, records)
    print(f"\nTOTAL records: {len(records)} -> {base.OUT_PATH}")
