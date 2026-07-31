"""
QuantumBridge Emulator — v2 (fixed noise injection)
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random

INTERCEPT = 3.003
CNOT_WEIGHT = 1.618
KINGSTON_WEIGHT = -2.311


def count_cnots(circuit: QuantumCircuit) -> int:
    return sum(1 for instr in circuit.data if instr.operation.name == "cx")


def predict_error_rate(circuit: QuantumCircuit, backend: str = "ibm_fez") -> float:
    cnots = count_cnots(circuit)
    is_kingston = 1 if backend == "ibm_kingston" else 0
    predicted = INTERCEPT + (CNOT_WEIGHT * cnots) + (KINGSTON_WEIGHT * is_kingston)
    return max(0.0, predicted)


def emulate(circuit: QuantumCircuit, backend: str = "ibm_fez", shots: int = 1024) -> dict:
    sim = AerSimulator()
    ideal_circuit = circuit.copy()
    ideal_circuit.measure_all(add_bits=False)
    ideal_result = sim.run(ideal_circuit, shots=shots).result()
    ideal_counts = ideal_result.get_counts()

    error_rate = predict_error_rate(circuit, backend)
    error_fraction = error_rate / 100.0

    num_qubits = circuit.num_qubits
    all_possible = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]
    ideal_outcomes = list(ideal_counts.keys())
    ideal_weights = list(ideal_counts.values())

    # WRONG outcomes only — outcomes that were NOT in the ideal distribution
    wrong_outcomes = [o for o in all_possible if o not in ideal_outcomes]

    noisy_counts = {}
    for _ in range(shots):
        if random.random() < error_fraction and wrong_outcomes:
            # A real noisy shot: MUST land on a wrong outcome
            outcome = random.choice(wrong_outcomes)
        else:
            outcome = random.choices(ideal_outcomes, weights=ideal_weights)[0]
        noisy_counts[outcome] = noisy_counts.get(outcome, 0) + 1

    return {
        "backend": f"quantumbridge_emulated_{backend}",
        "predicted_error_rate": round(error_rate, 2),
        "cnot_count": count_cnots(circuit),
        "counts": noisy_counts,
    }


if __name__ == "__main__":
    print("QuantumBridge Emulator — Demo\n" + "="*50)

    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)

    print("\nCircuit: Bell state (1 CNOT)\n")

    for backend in ["ibm_fez", "ibm_kingston"]:
        result = emulate(bell, backend=backend, shots=1024)
        print(f"--- Emulated on {backend} ---")
        print(f"Predicted error rate: {result['predicted_error_rate']}%")
        print(f"Emulated counts: {result['counts']}")
        error_count = sum(v for k, v in result['counts'].items() if k not in ['00','11'])
        actual_rate = round(error_count/1024*100, 2)
        print(f"Actual noise injected: {actual_rate}% (should now match predicted closely)\n")

    print("="*50)
    print("Compare this to your REAL Entry 003 result on ibm_fez:")
    print("Real hardware: {'00': 488, '11': 486, '10': 32, '01': 18}  (4.88% error)")
