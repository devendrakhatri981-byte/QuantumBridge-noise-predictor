"""
QuantumBridge — Empirical Gate-Count Decay Curve

Isolates how success probability actually falls off with CX gate count,
using Aer's realistic noise model. Circuit design: a Bell state on a real
adjacent pair, with N redundant CX-CX pairs inserted (which cancel
logically, so the ideal outcome never changes) between the same two
qubits. optimization_level=0 prevents Qiskit from cancelling the
redundant pairs before simulation. This isolates gate-count decay from
everything else (routing, topology, ideal-outcome definition).
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

backend = FakeCairoV2()
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator(noise_model=noise_model)

IDEAL_OUTCOMES = {"00", "11"}
Q1, Q2 = 24, 25  # real adjacent, low-noise pair (per Entry 013/014)
SHOTS = 4096

def build_circuit(extra_cx_pairs):
    """Bell state with `extra_cx_pairs` redundant CX-CX insertions."""
    qc = QuantumCircuit(27, 2)
    qc.h(Q1)
    qc.cx(Q1, Q2)  # the "real" entangling gate
    for _ in range(extra_cx_pairs):
        qc.cx(Q1, Q2)
        qc.cx(Q1, Q2)  # cancels the one above, ideal outcome unchanged
    qc.measure([Q1, Q2], [0, 1])
    return qc

def success_probability(qc):
    transpiled = transpile(qc, backend=backend, optimization_level=0)
    result = sim.run(transpiled, shots=SHOTS).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return sum(c for b, c in counts.items()
                if b.replace(" ", "") in IDEAL_OUTCOMES) / total

if __name__ == "__main__":
    # Total real CX count = 1 (the actual Bell gate) + 2*extra_pairs
    extra_pair_counts = [0, 2, 4, 6, 8, 10, 13, 16, 20]

    print(f"{'Total CX count':>15} {'Success prob':>14}")
    print("=" * 32)

    results = []
    for extra in extra_pair_counts:
        total_cx = 1 + 2 * extra
        qc = build_circuit(extra)
        prob = success_probability(qc)
        results.append((total_cx, prob))
        print(f"{total_cx:>15} {prob*100:>13.2f}%")

    import json
    with open("quantumbridge_data/gate_count_decay_curve.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to quantumbridge_data/gate_count_decay_curve.json")
