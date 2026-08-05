"""
QuantumBridge — Entry 020: Forgiveness ratio decay curves on FakeSherbrooke,
using a low-error edge (60,61) and high-error edge (66,73), chosen to be
comparable in magnitude to the FakeCairoV2 pair from Entries 017/019.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
import json
import numpy as np

backend = FakeSherbrooke()
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator(noise_model=noise_model)

IDEAL_OUTCOMES = {"00", "11"}
SHOTS = 4096

def build_circuit(q1, q2, extra_cx_pairs):
    qc = QuantumCircuit(backend.num_qubits, 2)
    qc.h(q1)
    qc.cx(q1, q2)
    for _ in range(extra_cx_pairs):
        qc.cx(q1, q2)
        qc.cx(q1, q2)
    qc.measure([q1, q2], [0, 1])
    return qc

def success_probability(qc):
    transpiled = transpile(qc, backend=backend, optimization_level=0)
    result = sim.run(transpiled, shots=SHOTS).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return sum(c for b, c in counts.items()
                if b.replace(" ", "") in IDEAL_OUTCOMES) / total

def run_curve(q1, q2, raw_edge_err, label):
    print(f"\n{'='*50}\n{label}: edge ({q1},{q2}), raw error {raw_edge_err*100:.4f}%\n{'='*50}")
    extra_pair_counts = [0, 2, 4, 6, 8, 10, 13, 16, 20]
    results = []
    for extra in extra_pair_counts:
        total_cx = 1 + 2 * extra
        qc = build_circuit(q1, q2, extra)
        prob = success_probability(qc)
        results.append((total_cx, prob))
        print(f"  {total_cx:>3} CX gates: {prob*100:.2f}%")

    n_vals = np.array([r[0] for r in results])
    success_vals = np.array([r[1] for r in results])
    k_fit, log_A = np.polyfit(n_vals, np.log(success_vals), 1)
    k_fit = -k_fit
    ratio = k_fit / raw_edge_err

    print(f"\n  Fitted decay constant: {k_fit*100:.4f}% per gate")
    print(f"  Forgiveness ratio:     {ratio:.3f}")
    return results, k_fit, ratio

if __name__ == "__main__":
    low_results, low_k, low_ratio = run_curve(60, 61, 0.003470, "LOW-ERROR EDGE")
    high_results, high_k, high_ratio = run_curve(66, 73, 0.030667, "HIGH-ERROR EDGE")

    print(f"\n{'='*50}\nSUMMARY — Sherbrooke vs Cairo\n{'='*50}")
    print(f"Sherbrooke low-error ratio:  {low_ratio:.3f}   (Cairo: 0.53)")
    print(f"Sherbrooke high-error ratio: {high_ratio:.3f}   (Cairo: 0.334)")

    with open("quantumbridge_data/sherbrooke_decay_curves.json", "w") as f:
        json.dump({
            "low_edge": {"qubits": [60, 61], "raw_error": 0.003470, "results": low_results, "k": low_k, "ratio": low_ratio},
            "high_edge": {"qubits": [66, 73], "raw_error": 0.030667, "results": high_results, "k": high_k, "ratio": high_ratio},
        }, f, indent=2)
    print("\nSaved to quantumbridge_data/sherbrooke_decay_curves.json")
