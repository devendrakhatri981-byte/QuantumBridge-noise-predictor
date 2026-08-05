"""
QuantumBridge — Entry 021 Setup: Third chip for ratio-of-ratios confirmation.
Same pipeline as Entry 013 (Cairo) and Entry 020 (Sherbrooke).
"""

import json
from qiskit_ibm_runtime.fake_provider import FakeKyiv

backend = FakeKyiv()
print(f"Chip: {backend.name}")
print(f"Qubits: {backend.num_qubits}")
print(f"Basis gates: {backend.operation_names}\n")

edges_from_cm = set(map(tuple, backend.coupling_map.get_edges()))
edges_from_config = set(map(tuple, backend.configuration().coupling_map))
all_edges = edges_from_cm | edges_from_config
undirected_edges = sorted({(min(a, b), max(a, b)) for a, b in all_edges})

print(f"Edges from coupling_map.get_edges(): {len(edges_from_cm)}")
print(f"Edges from configuration():          {len(edges_from_config)}")
print(f"Union (undirected, deduped):         {len(undirected_edges)}")

with open("quantumbridge_data/real_topology_kyiv.json", "w") as f:
    json.dump(undirected_edges, f, indent=2)
print("Saved to quantumbridge_data/real_topology_kyiv.json")

props = backend.properties()
two_qubit_gate_errors = []
for gate in props.gates:
    if len(gate.qubits) == 2:
        err = next((p.value for p in gate.parameters if p.name == "gate_error"), None)
        if err is not None:
            two_qubit_gate_errors.append({"qubits": list(gate.qubits), "gate_error": err})

calibration = {"chip": "kyiv", "two_qubit_gate_errors": two_qubit_gate_errors}
with open("quantumbridge_data/offline_calibration_kyiv_full.json", "w") as f:
    json.dump(calibration, f, indent=2)
print(f"Saved {len(two_qubit_gate_errors)} two-qubit gate error entries")

# Filter placeholders and show low/high candidates, same as Entry 020
error_lookup = {}
for g in two_qubit_gate_errors:
    q1, q2 = g["qubits"]
    err = g["gate_error"]
    if err is not None:
        error_lookup[tuple(sorted((q1, q2)))] = err

real_edge_set = {tuple(e) for e in undirected_edges}
calibrated_real_edges = [(edge, err) for edge, err in error_lookup.items()
                          if edge in real_edge_set and err < 0.5]
calibrated_real_edges.sort(key=lambda x: x[1])

print(f"\nTotal plausible calibrated edges: {len(calibrated_real_edges)}")
print("\n5 LOWEST error edges:")
for edge, err in calibrated_real_edges[:5]:
    print(f"  {edge}: {err*100:.4f}%")
print("\n5 HIGHEST plausible error edges:")
for edge, err in calibrated_real_edges[-5:]:
    print(f"  {edge}: {err*100:.4f}%")
