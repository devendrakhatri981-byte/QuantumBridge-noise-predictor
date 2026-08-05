"""
QuantumBridge — Entry 020 Setup: Second Chip for Generalization Test

Exports FakeSherbrooke's real topology (union of coupling_map.get_edges()
and configuration().coupling_map, same undercount check as Entry 013) and
pulls its calibration data, so the same forgiveness-ratio experiment can
be repeated on a structurally different chip: 127 qubits, heavy-hex,
native ecr gate instead of cx.
"""

import json
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

backend = FakeSherbrooke()

print(f"Chip: {backend.name}")
print(f"Qubits: {backend.num_qubits}")
print(f"Basis gates: {backend.operation_names}\n")

# --- Topology: same union-of-sources check as Entry 013 ---
edges_from_cm = set(map(tuple, backend.coupling_map.get_edges()))
edges_from_config = set(map(tuple, backend.configuration().coupling_map))
all_edges = edges_from_cm | edges_from_config
undirected_edges = sorted({(min(a, b), max(a, b)) for a, b in all_edges})

print(f"Edges from coupling_map.get_edges(): {len(edges_from_cm)}")
print(f"Edges from configuration():          {len(edges_from_config)}")
print(f"Union (undirected, deduped):         {len(undirected_edges)}")

with open("quantumbridge_data/real_topology_sherbrooke.json", "w") as f:
    json.dump(undirected_edges, f, indent=2)
print("Saved to quantumbridge_data/real_topology_sherbrooke.json")

# --- Calibration: pull two-qubit gate errors directly from backend properties ---
props = backend.properties()
two_qubit_gate_errors = []
for gate in props.gates:
    if len(gate.qubits) == 2:
        err = next((p.value for p in gate.parameters if p.name == "gate_error"), None)
        if err is not None:
            two_qubit_gate_errors.append({"qubits": list(gate.qubits), "gate_error": err})

calibration = {"chip": "sherbrooke", "two_qubit_gate_errors": two_qubit_gate_errors}
with open("quantumbridge_data/offline_calibration_sherbrooke_full.json", "w") as f:
    json.dump(calibration, f, indent=2)
print(f"Saved {len(two_qubit_gate_errors)} two-qubit gate error entries "
      f"to quantumbridge_data/offline_calibration_sherbrooke_full.json")

# --- Sanity check: does this chip show the same edge-undercount issue? ---
missing_from_cm = set(undirected_edges) - {(min(a,b), max(a,b)) for a,b in edges_from_cm}
print(f"\nEdges present in union but missing from coupling_map.get_edges() alone: {len(missing_from_cm)}")
if missing_from_cm:
    print(f"  (confirms the same undercount pattern seen on FakeCairoV2 in Entry 013)")
