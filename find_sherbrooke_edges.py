"""
QuantumBridge — Entry 020: Find contrasting low/high error edges on Sherbrooke,
to mirror the (24,25) low-error / (19,22) high-error pair used on FakeCairoV2.
"""

import json

with open("quantumbridge_data/offline_calibration_sherbrooke_full.json") as f:
    calibration = json.load(f)

with open("quantumbridge_data/real_topology_sherbrooke.json") as f:
    real_edges = json.load(f)

error_lookup = {}
for g in calibration["two_qubit_gate_errors"]:
    q1, q2 = g["qubits"]
    err = g["gate_error"]
    if err is not None:
        error_lookup[tuple(sorted((q1, q2)))] = err

# Only consider edges that are BOTH real topology edges AND have measured calibration
real_edge_set = {tuple(e) for e in real_edges}
calibrated_real_edges = [(edge, err) for edge, err in error_lookup.items() if edge in real_edge_set]
calibrated_real_edges.sort(key=lambda x: x[1])

print(f"Total calibrated, real edges: {len(calibrated_real_edges)}\n")
print("5 LOWEST error edges:")
for edge, err in calibrated_real_edges[:5]:
    print(f"  {edge}: {err*100:.4f}%")

print("\n5 HIGHEST error edges:")
for edge, err in calibrated_real_edges[-5:]:
    print(f"  {edge}: {err*100:.4f}%")

errs = [e for _, e in calibrated_real_edges]
print(f"\nMean: {sum(errs)/len(errs)*100:.4f}%")
print(f"Median: {sorted(errs)[len(errs)//2]*100:.4f}%")
