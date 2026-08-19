"""QuantumBridge — Entry 040b: fifth growth batch, far bins + GHZ (balance)."""

import entry034_batch_dataset as base

base.random.seed(888)
base.OUT_PATH = "quantumbridge_data/entry040_far_dataset.json"
base.BINS = [(16, 25), (26, 999)]
base.PER_BIN = 30
base.N_GHZ = 20

if __name__ == "__main__":
    records = base.load(base.OUT_PATH)
    print(f"resuming with {len(records)} existing records")
    for chip in ("kyiv", "sherbrooke"):
        base.process_chip(chip, records)
    print(f"\nTOTAL records: {len(records)} -> {base.OUT_PATH}")
