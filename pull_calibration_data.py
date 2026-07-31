"""
QuantumBridge — Pull Real Chip Calibration Data (FREE, no quota used)

This queries IBM's calibration data directly — T1, T2, gate errors,
readout errors, per qubit. No job submission, no quota cost.
"""

from qiskit_ibm_runtime import QiskitRuntimeService
import json
from datetime import datetime

TOKEN = "whudFfVHj_V1izGAIXKVulrRhub49tQc-nrr8prWqVSA"

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=TOKEN,
    instance="open-instance"
)

for backend_name in ["ibm_fez", "ibm_kingston"]:
    print(f"\n{'='*60}")
    print(f"Pulling calibration data: {backend_name}")
    print(f"{'='*60}")

    try:
        backend = service.backend(backend_name)
        props = backend.properties()

        if props is None:
            print(f"No properties available for {backend_name} right now.")
            continue

        qubit_data = []
        num_qubits_to_check = min(10, backend.num_qubits)  # first 10 qubits

        for q in range(num_qubits_to_check):
            t1 = props.t1(q) * 1e6 if props.t1(q) else None  # microseconds
            t2 = props.t2(q) * 1e6 if props.t2(q) else None
            readout_err = props.readout_error(q)

            qubit_data.append({
                "qubit": q,
                "t1_microseconds": round(t1, 2) if t1 else None,
                "t2_microseconds": round(t2, 2) if t2 else None,
                "readout_error": round(readout_err, 5) if readout_err else None,
            })
            print(f"  Qubit {q}: T1={t1:.1f}us, T2={t2:.1f}us, readout_err={readout_err:.4f}")

        output = {
            "backend": backend_name,
            "pulled_at": datetime.now().isoformat(),
            "total_qubits_on_chip": backend.num_qubits,
            "qubit_calibration": qubit_data,
        }

        filename = f"quantumbridge_data/calibration_{backend_name}.json"
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {filename}")

    except Exception as e:
        print(f"Could not pull data for {backend_name}: {e}")

print("\nDone. This used ZERO quota — pure metadata query.")
