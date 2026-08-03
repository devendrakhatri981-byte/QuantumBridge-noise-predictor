"""
QuantumBridge — GHZ Gate-Count Decay Curve (Generalization Test)

Same cancellation-based method as the Bell-state decay curve (Entry 017),
but on a 3-qubit GHZ circuit instead. Tests whether the 0.53 forgiveness
ratio fitted on Bell states holds for a structurally different circuit,
or whether it's specific to that one circuit shape.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
import json
import numpy as np

backend = FakeCairoV2()
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator(noise_model=noise_model)

IDEAL_OUTCOMES = {"000", "111"}
Q1, Q2, Q3 = 24, 25, 22  # real connected chain — confirm via graph if unsure
SHOTS = 4096

def build_circuit(extra_cx_pairs):
    """GHZ state with `extra_cx_pairs` redundant CX-CX insertions on the
    first entangling edge (Q1,Q2). The second real edge (Q2,Q3) stays
    untouched, so this isolates the same single-edge decay behavior as
    the Bell-state experiment."""
    qc = QuantumCircuit(27, 3)
    qc.h(Q1)
    qc.cx(Q1, Q2)
    for _ in range(extra_cx_pairs):
        qc.cx(Q1, Q2)
        qc.cx(Q1, Q2)
    qc.cx(Q2, Q3)
    qc.measure([Q1, Q2, Q3], [0, 1, 2])
    return qc

def success_probability(qc):
    transpiled = transpile(qc, backend=backend, optimization_level=0)
    result = sim.run(transpiled, shots=SHOTS).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return sum(c for b, c in counts.items()
                if b.replace(" ", "") in IDEAL_OUTCOMES) / total

if __name__ == "__main__":
    extra_pair_counts = [0, 2, 4, 6, 8, 10, 13, 16, 20]

    print(f"{'Total CX count':>15} {'Success prob':>14}")
    print("=" * 32)

    results = []
    for extra in extra_pair_counts:
        total_cx = 2 + 2 * extra  # 2 real gates (Q1-Q2, Q2-Q3) + cancelling pairs
        qc = build_circuit(extra)
        prob = success_probability(qc)
        results.append((total_cx, prob))
        print(f"{total_cx:>15} {prob*100:>13.2f}%")

    with open("quantumbridge_data/ghz_decay_curve.json", "w") as f:
        json.dump(results, f, indent=2)

    # Fit and compare against the Bell-state ratio
    n_vals = np.array([r[0] for r in results])
    success_vals = np.array([r[1] for r in results])
    k_fit, log_A = np.polyfit(n_vals, np.log(success_vals), 1)
    k_fit = -k_fit

    with open("quantumbridge_data/fitted_decay_model.json") as f:
        bell_model = json.load(f)

    print(f"\nGHZ fitted decay constant:  {k_fit*100:.4f}% per gate")
    print(f"Bell fitted decay constant: {bell_model['k']*100:.4f}% per gate")
    print(f"Ratio (GHZ / Bell):         {k_fit / bell_model['k']:.3f}")
    print("\nSaved to quantumbridge_data/ghz_decay_curve.json")
