"""
QuantumBridge — Entry 035: an independent replication batch.

WHY THIS EXISTS
---------------
Entry 034's batch (116 circuits) surfaced 7 short routes that collapse
completely to the 50% floor, concentrated on Sherbrooke qubits 5 and 6.
Before spending an entry chasing a root cause, the obvious question is
whether that is a property of THOSE SPECIFIC QUBITS, or a general
phenomenon that would show up again on a completely independent sample.

This script re-runs the exact same batch methodology (entry034_batch_
dataset.py) with a different random seed, sampling a fresh, non-
overlapping set of qubit pairs on the same two chips, into a SEPARATE
output file. entry035_compare.py (run after this finishes) then checks:
does the floor-collapse pattern reappear on the SAME qubits (5/6), on
DIFFERENT qubits (a general phenomenon), or not at all (034's outliers
were a sampling fluke)? That comparison decides whether to fix this as
one shared bug or treat the two batches' findings separately.
"""

import entry034_batch_dataset as base

base.random.seed(99)               # different draw from Entry 034's seed(7)
base.OUT_PATH = "quantumbridge_data/entry035_replication_dataset.json"
base.PER_BIN = 10                  # same shape as Entry 034 for a fair comparison
base.N_GHZ = 10

if __name__ == "__main__":
    records = base.load(base.OUT_PATH)
    print(f"resuming with {len(records)} existing records")
    for chip in ("kyiv", "sherbrooke"):
        base.process_chip(chip, records)
    print(f"\nTOTAL records: {len(records)} -> {base.OUT_PATH}")
