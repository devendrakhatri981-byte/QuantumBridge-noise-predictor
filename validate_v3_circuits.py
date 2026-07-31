"""
QuantumBridge Emulator v3 — Circuit-Level Validation

Runs full multi-qubit circuits through v3's SWAP-aware error model and
compares its predicted success probability against a realistic noise-model
simulation of the same circuit (Qiskit Aer + NoiseModel.from_backend on
FakeCairoV2). Fully offline -- no IBM Cloud access required.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

from emulator_v3_routing import load_calibration, build_connectivity_graph, cnot_error_for_pair

SQ_GATE_COST = 0.00224  # ~0.224 pts/gate, from Entry 010's deconfounded measurement


def v3_predicted_success_probability(circuit, graph):
    """Walk the circuit's gate list and multiply per-gate success probabilities
    using v3's routing-aware CNOT cost model plus the single-qubit gate cost
    measured in Entry 010."""
    success_prob = 1.0
    details = []
    for instruction in circuit.data:
        gate = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if gate.num_qubits == 2:
            q1, q2 = qubits
            err, notes = cnot_error_for_pair(graph, q1, q2)
            success_prob *= (1 - err)
            details.append((gate.name, q1, q2, round(err * 100, 2), notes[0]))
        elif gate.num_qubits == 1 and gate.name not in ("measure", "barrier"):
            success_prob *= (1 - SQ_GATE_COST)
    return success_prob, details


from qiskit.transpiler import CouplingMap
import json

with open("quantumbridge_data/real_topology_cairo.json") as f:
    _real_edges = json.load(f)
FIXED_COUPLING_MAP = CouplingMap(couplinglist=_real_edges + [[b, a] for a, b in _real_edges])


def real_noise_success_probability(circuit, backend, ideal_outcomes, shots=4096):
    """Transpile against our validated 28-edge topology and simulate under a
    realistic noise model. Success = fraction of shots landing on ANY of the
    circuit's ideal outcomes (e.g. both 00 and 11 for a Bell state), not just
    the single most frequent bitstring."""
    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=noise_model)
    transpiled = transpile(
        circuit,
        coupling_map=FIXED_COUPLING_MAP,
        basis_gates=backend.operation_names,
        initial_layout=list(range(circuit.num_qubits)),
        optimization_level=1,
    )
    result = sim.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    total = sum(counts.values())
    success = sum(c for bitstring, c in counts.items()
                   if bitstring.replace(" ", "") in ideal_outcomes) / total
    return success, counts


if __name__ == "__main__":
    calibration = load_calibration()
    graph = build_connectivity_graph(calibration)
    backend = FakeCairoV2()
    qc1 = QuantumCircuit(27, 3)
    qc1.h(0); qc1.cx(0, 1); qc1.cx(1, 2)
    qc1.measure([0, 1, 2], [0, 1, 2])

    qc2 = QuantumCircuit(27, 3)
    qc2.h(0); qc2.cx(0, 26); qc2.cx(26, 22)
    qc2.measure([0, 26, 22], [0, 1, 2])

    qc3 = QuantumCircuit(27, 2)
    qc3.h(24); qc3.cx(24, 25)
    qc3.measure([24, 25], [0, 1])

    qc4 = QuantumCircuit(27, 2)
    qc4.h(0); qc4.cx(0, 26)
    qc4.measure([0, 26], [0, 1])
    test_circuits = {
        "ghz_local_0_1_2":       (qc1, {"000", "111"}),
        "ghz_scattered_0_26_22": (qc2, {"000", "111"}),
        "bell_adjacent_24_25":   (qc3, {"00", "11"}),
        "bell_scattered_0_26":   (qc4, {"00", "11"}),
    }

    print(f"{'Circuit':<26} {'v3 predicted':>14} {'Aer+noise (real)':>18} {'Diff':>8}")
    print("=" * 70)

    diffs = []
    for name, (qc, ideal) in test_circuits.items():
        v3_prob, details = v3_predicted_success_probability(qc, graph)
        real_prob, counts = real_noise_success_probability(qc, backend, ideal)
        diff = abs(v3_prob - real_prob)
        diffs.append(diff)
        print(f"{name:<26} {v3_prob*100:>12.2f}%  {real_prob*100:>16.2f}%  {diff*100:>6.2f}pts")

    print(f"\nMean absolute difference: {sum(diffs)/len(diffs)*100:.2f} percentage points")