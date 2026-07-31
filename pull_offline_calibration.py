"""
QuantumBridge — Pull Multi-Parameter Calibration Data (100% offline)

Extracts T1, T2, readout error, and per-gate error rates from a real
historical IBM chip snapshot bundled in Qiskit. No internet needed.
"""

from qiskit_ibm_runtime.fake_provider import FakeCairoV2
import json

backend = FakeCairoV2()
props = backend.target

print(f"Chip: {backend.name}  —  {backend.num_qubits} qubits\n")
print(f"{'Qubit':<7}{'T1 (us)':<12}{'T2 (us)':<12}{'Readout Err':<14}{'SX Gate Err':<14}")
print("-" * 60)

qubit_data = []
for q in range(backend.num_qubits):
    try:
        qprops = backend.qubit_properties[q]
        t1 = round(qprops.t1 * 1e6, 2) if qprops.t1 else None
        t2 = round(qprops.t2 * 1e6, 2) if qprops.t2 else None
    except Exception:
        t1, t2 = None, None

    # Readout error
    try:
        readout_err = round(backend.target["measure"][(q,)].error, 5)
    except Exception:
        readout_err = None

    # Single-qubit gate error (SX is the standard basis gate)
    try:
        sx_err = round(backend.target["sx"][(q,)].error, 6)
    except Exception:
        sx_err = None

    qubit_data.append({
        "qubit": q, "t1_us": t1, "t2_us": t2,
        "readout_error": readout_err, "sx_gate_error": sx_err
    })
    print(f"{q:<7}{str(t1):<12}{str(t2):<12}{str(readout_err):<14}{str(sx_err):<14}")

# CNOT (ECR/CX) gate errors — the expensive two-qubit gate
print(f"\n{'Qubit pair':<14}{'CNOT/ECR Error':<16}")
print("-" * 30)
cnot_data = []
gate_name = "cx" if "cx" in backend.target else "ecr"
for pair, props in list(backend.target[gate_name].items())[:15]:
    err = round(props.error, 5) if props and props.error else None
    cnot_data.append({"qubits": list(pair), "gate_error": err})
    print(f"{str(pair):<14}{str(err):<16}")

output = {
    "chip": backend.name,
    "num_qubits": backend.num_qubits,
    "qubit_calibration": qubit_data,
    "two_qubit_gate_errors": cnot_data,
    "two_qubit_gate_type": gate_name,
}
with open("quantumbridge_data/offline_calibration_cairo.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to quantumbridge_data/offline_calibration_cairo.json")
