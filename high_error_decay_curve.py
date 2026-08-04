"""
QuantumBridge — Testing Forgiveness Ratio on a High-Error Edge (Entry 019)

The original 0.53 forgiveness ratio (Entry 017) was fitted on edge (24,25),
a LOW-error edge (~0.60%). This repeats the same cancellation-based decay
curve method on edge (19,22), a HIGH-error edge (~3.13%) that sits on
bell_scattered_0_26's real route, to test whether forgiveness applies
uniformly regardless of the underlying edge error magnitude.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
import json
import numpy as np

from emulator_v3_routing import load_calibration, build_connectivity_graph, edge_error

backend = FakeCairoV2()
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator(noise_model=noise_model)

calibration = load_calibration()
graph = build_connectivity_graph(calibration)

IDEAL_OUTCOMES = {"00", "11"}
Q1, Q2 = 19, 22  # the HIGH-error edge flagged in the investigation
SHOTS = 4096

raw_edge_err = edge_error(graph, Q1, Q2)
print(f"Raw calibrated error for edge ({Q1},{Q2}): {raw_edge_err*100:.4f}%\n")

def build_circuit(extra_cx_pairs):
    qc = QuantumCircuit(27, 2)
    qc.h(Q1)
    qc.cx(Q1, Q2)
    for _ in range(extra_cx_pairs):
        qc.cx(Q1, Q2)
        qc.cx(Q1, Q2)
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

    n_vals = np.array([r[0] for r in results])
    success_vals = np.array([r[1] for r in results])
    k_fit, log_A = np.polyfit(n_vals, np.log(success_vals), 1)
    k_fit = -k_fit

    forgiveness_ratio_high = k_fit / raw_edge_err

    print(f"\nFitted decay constant on HIGH-error edge: {k_fit*100:.4f}% per gate")
    print(f"Raw calibrated error (this edge):          {raw_edge_err*100:.4f}%")
    print(f"Forgiveness ratio (this edge):              {forgiveness_ratio_high:.3f}")
    print(f"\nOriginal forgiveness ratio (24,25, low-error): 0.53")
    print(f"Difference: {abs(forgiveness_ratio_high - 0.53):.3f}")

    with open("quantumbridge_data/high_error_decay_curve.json", "w") as f:
        json.dump({"results": results, "edge": [Q1, Q2],
                   "raw_edge_error": raw_edge_err,
                   "forgiveness_ratio": forgiveness_ratio_high}, f, indent=2)
    print("\nSaved to quantumbridge_data/high_error_decay_curve.json")
