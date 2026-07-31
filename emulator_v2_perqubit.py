"""
QuantumBridge Emulator v2 — Per-Qubit Aware Noise Model

Instead of one flat error rate per chip, this traces a circuit's actual
gates through real per-qubit calibration data (SX error, CNOT/ECR error,
readout error) and multiplies success probabilities gate by gate.

Uses real offline chip calibration data — no IBM account needed.
"""

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
import json


def load_calibration():
    with open("quantumbridge_data/offline_calibration_cairo.json") as f:
        return json.load(f)


def predict_error_perqubit(circuit: QuantumCircuit, qubit_mapping: list, calibration: dict) -> dict:
    """
    Predict error rate by tracing the circuit's actual gates through
    real per-qubit calibration data, instead of using one flat number.

    qubit_mapping: which PHYSICAL chip qubits this circuit is placed on,
                   e.g. [0, 1, 2] means logical qubit 0 -> physical qubit 0, etc.
    """
    qubit_cal = {q["qubit"]: q for q in calibration["qubit_calibration"]}
    cnot_cal = {tuple(g["qubits"]): g["gate_error"] for g in calibration["two_qubit_gate_errors"]}

    success_prob = 1.0
    gate_log = []

    for instr in circuit.data:
        gate_name = instr.operation.name
        qubit_indices = [circuit.find_bit(q).index for q in instr.qubits]
        physical_qubits = [qubit_mapping[i] for i in qubit_indices]

        if gate_name in ["h", "x", "sx", "rz", "id"]:
            # Single-qubit gate: use that physical qubit's SX error as a proxy
            q = physical_qubits[0]
            err = qubit_cal.get(q, {}).get("sx_gate_error") or 0.0003  # fallback
            success_prob *= (1 - err)
            gate_log.append(f"{gate_name} on q{q}: err={err}")

        elif gate_name in ["cx", "cnot", "ecr"]:
            pair = tuple(physical_qubits)
            err = cnot_cal.get(pair) or cnot_cal.get(pair[::-1])
            if err is None:
                err = 0.01  # fallback average if this exact pair wasn't in our sample
            success_prob *= (1 - err)
            gate_log.append(f"{gate_name} on q{pair}: err={err}")

    # Add readout error for every qubit that gets measured
    measured_qubits = set()
    for instr in circuit.data:
        if instr.operation.name == "measure":
            for q in instr.qubits:
                idx = circuit.find_bit(q).index
                measured_qubits.add(qubit_mapping[idx])

    for q in measured_qubits:
        err = qubit_cal.get(q, {}).get("readout_error") or 0.02
        success_prob *= (1 - err)
        gate_log.append(f"readout q{q}: err={err}")

    error_rate = round((1 - success_prob) * 100, 2)

    return {
        "physical_qubits_used": qubit_mapping,
        "predicted_error_rate": error_rate,
        "gate_trace": gate_log,
    }


if __name__ == "__main__":
    calibration = load_calibration()
    print(f"Loaded calibration for: {calibration['chip']} ({calibration['num_qubits']} qubits)\n")

    # Bell state, same circuit you've tested many times
    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])

    print("="*60)
    print("Same Bell circuit, placed on DIFFERENT physical qubits")
    print("="*60)

    # Try it on a cheap pair vs an expensive pair from the real data
    test_placements = [
        ("Cheap pair (24,25)", [24, 25]),
        ("Expensive pair (22,19)", [22, 19]),
        ("Default (0,1) - not directly connected!", [0, 1]),
    ]

    for label, mapping in test_placements:
        result = predict_error_perqubit(bell, mapping, calibration)
        print(f"\n{label}")
        print(f"  Predicted error rate: {result['predicted_error_rate']}%")
        for line in result['gate_trace']:
            print(f"    {line}")
