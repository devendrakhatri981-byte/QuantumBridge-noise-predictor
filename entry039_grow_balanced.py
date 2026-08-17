"""
QuantumBridge — Entry 039: a fourth, balanced growth batch.

Entry 038 deliberately skewed toward short/mid routes to build up the
floor-collapse sample cheaply. This batch restores balance by sampling
all six hop-distance bins again (like Entries 034/035) plus a GHZ batch,
with a fourth independent seed, to keep growing the dataset toward the
low thousands without losing coverage of the long-route regime the
project's earlier entries (022-033) were built on.
"""

import entry034_batch_dataset as base

base.random.seed(3141)
base.OUT_PATH = "quantumbridge_data/entry039_balanced_dataset.json"
base.BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 25), (26, 999)]
base.PER_BIN = 20
base.N_GHZ = 15

if __name__ == "__main__":
    records = base.load(base.OUT_PATH)
    print(f"resuming with {len(records)} existing records")
    for chip in ("kyiv", "sherbrooke"):
        base.process_chip(chip, records)
    print(f"\nTOTAL records: {len(records)} -> {base.OUT_PATH}")
