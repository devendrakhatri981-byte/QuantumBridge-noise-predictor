"""
QuantumBridge — Entry 061 Setup: Third Chip for Generalization Test

Exports FakeBrisbane's real topology and calibration data, following the
same pattern as setup_sherbrooke.py (Entry 020). Brisbane is the same
127-qubit heavy-hex class as Kyiv and Sherbrooke, chosen specifically so
the existing generation/training pipeline (built around 127-qubit chips)
needs minimal adaptation, while still being a genuinely different chip
with its own real calibration snapshot -- the point is to test whether
cross-chip generalization holds up across more than one chip pair, not
to introduce a confound from a different chip scale at the same time.
"""

import json
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

backend = FakeBrisbane()

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

with open("quantumbridge_data/real_topology_brisbane.json", "w") as f:
    json.dump(undirected_edges, f, indent=2)
print("Saved to quantumbridge_data/real_topology_brisbane.json")

props = backend.properties()
two_qubit_gate_errors = []
for gate in props.gates:
    if len(gate.qubits) == 2:
        err = next((p.value for p in gate.parameters if p.name == "gate_error"), None)
        if err is not None:
            two_qubit_gate_errors.append({"qubits": list(gate.qubits), "gate_error": err})

calibration = {"chip": "brisbane", "two_qubit_gate_errors": two_qubit_gate_errors}
with open("quantumbridge_data/offline_calibration_brisbane_full.json", "w") as f:
    json.dump(calibration, f, indent=2)
print(f"Saved {len(two_qubit_gate_errors)} two-qubit gate error entries "
      f"to quantumbridge_data/offline_calibration_brisbane_full.json")

missing_from_cm = set(undirected_edges) - {(min(a, b), max(a, b)) for a, b in edges_from_cm}
print(f"\nEdges present in union but missing from coupling_map.get_edges() alone: {len(missing_from_cm)}")
