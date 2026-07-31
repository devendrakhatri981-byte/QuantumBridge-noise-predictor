"""
QuantumBridge — Pull COMPLETE Calibration Data (all connections, not just 15)
"""

from qiskit_ibm_runtime.fake_provider import FakeCairoV2
import json

backend = FakeCairoV2()

qubit_data = []
for q in range(backend.num_qubits):
    try:
        readout_err = round(backend.target["measure"][(q,)].error, 5)
    except Exception:
        readout_err = None
    try:
        sx_err = round(backend.target["sx"][(q,)].error, 6)
    except Exception:
        sx_err = None
    qubit_data.append({"qubit": q, "readout_error": readout_err, "sx_gate_error": sx_err})

gate_name = "cx" if "cx" in backend.target else "ecr"
cnot_data = []
for pair, props in backend.target[gate_name].items():
    err = round(props.error, 5) if props and props.error else None
    cnot_data.append({"qubits": list(pair), "gate_error": err})

print(f"Chip: {backend.name} — {backend.num_qubits} qubits")
print(f"Captured ALL {len(cnot_data)} two-qubit connections (was only 15 before)")

output = {
    "chip": backend.name,
    "num_qubits": backend.num_qubits,
    "qubit_calibration": qubit_data,
    "two_qubit_gate_errors": cnot_data,
    "two_qubit_gate_type": gate_name,
}
with open("quantumbridge_data/offline_calibration_cairo_full.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved to quantumbridge_data/offline_calibration_cairo_full.json")
