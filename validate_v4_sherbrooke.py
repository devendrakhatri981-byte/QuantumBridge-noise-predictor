"""
QuantumBridge — Entry 030: extending validation to Sherbrooke.

The v4.1 (exact-dwell) model was built, debugged, and validated entirely on
Kyiv (Entries 022-029). This script asks the obvious next question: does it
generalize to a chip it has never seen?

Sherbrooke was chosen over Cairo because it shares Kyiv's uniform ecr-only
basis (127 qubits, no native cx) -- Cairo genuinely mixes cx and ecr per
edge (forcing an ecr-only basis on a Cairo circuit throws a disconnected-
coupling-map error), which Entry 022 already flagged as unsuitable for this
kind of circuit-level Aer/MPS ground-truth validation.

bell_near_57_61 is not an arbitrary choice: qubit 57 is Sherbrooke's single
worst T2 (2.6us) by a wide margin, but its T1 is healthy (301.9us). Entry
028 corrected the emulator's decoherence term from T2 to T1 on the argument
that pure T2 dephasing is invisible to a computational-basis Bell
measurement. This circuit is a direct, independent test of that claim on a
qubit the fix was never tuned against.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost

SHOTS = 4096
SEEDS = (1, 2, 3, 4)

# Chosen via BFS over the real Sherbrooke coupling graph: an adjacent pair,
# a 4-hop route through qubit 57 (the T2-vs-T1 test), a 12-hop and a 26-hop
# pair, and a local GHZ near qubit 0.
CIRCUITS = {
    "bell_adjacent_0_1": [(0, 1)],
    "bell_near_57_61": [(57, 61)],
    "bell_mid_0_58": [(0, 58)],
    "bell_far_0_126": [(0, 126)],
    "ghz_local_0_1_14": [(0, 1), (0, 14)],
}


def build(pairs, backend):
    qubits = sorted(set(q for pr in pairs for q in pr))
    qc = QuantumCircuit(backend.num_qubits, len(qubits))
    qc.h(pairs[0][0])
    for a, b in pairs:
        qc.cx(a, b)
    for i, q in enumerate(qubits):
        qc.measure(q, i)
    return qc, qubits


if __name__ == "__main__":
    backend = FakeSherbrooke()
    NQ = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration("sherbrooke"), "sherbrooke")
    coh = v4.load_coherence("sherbrooke")
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=nm, method="matrix_product_state")

    print(f"{'Circuit':<24}{'Aer':>8}{'v3':>9}{'v4(BFS)':>9}{'v4.1(exact)':>13}"
          f"{'gap v3':>8}{'gap v4':>8}{'gap v4.1':>9}")
    print("=" * 90)

    results = []
    gaps3, gaps4, gaps41 = [], [], []
    for name, pairs in CIRCUITS.items():
        qc, meas = build(pairs, backend)
        t = transpile(qc, backend=backend, initial_layout=list(range(NQ)),
                     optimization_level=3, seed_transpiler=1)
        vals = []
        for sd in SEEDS:
            counts = sim.run(t, shots=SHOTS, seed_simulator=sd).result().get_counts()
            tot = sum(counts.values())
            ideal = {"0" * len(meas), "1" * len(meas)}
            ok = sum(c for b, c in counts.items() if b.replace(" ", "") in ideal)
            vals.append(ok / tot)
        ref = float(np.mean(vals))

        qc_bare = QuantumCircuit(NQ, len(meas))
        qc_bare.h(pairs[0][0])
        for a, b in pairs:
            qc_bare.cx(a, b)

        p3 = 1.0
        for inst in qc_bare.data:
            if inst.operation.num_qubits == 2:
                i = [qc_bare.find_bit(x).index for x in inst.qubits]
                p3 *= (1 - em.cnot_error_for_pair(graph, i[0], i[1])[0])
            elif inst.operation.num_qubits == 1:
                p3 *= (1 - v4.SQ_GATE_COST)

        p4 = v4.predict(qc_bare, graph, coh, measured_qubits=meas)
        p41, notes, _ = exact_dwell_cost(qc_bare, backend, graph, coh, pairs, meas)

        g3, g4, g41 = abs(p3 - ref), abs(p4 - ref), abs(p41 - ref)
        gaps3.append(g3); gaps4.append(g4); gaps41.append(g41)
        results.append({"name": name, "pairs": pairs, "aer_ref": ref,
                        "v3": p3, "v4_bfs": p4, "v4_1_exact": p41,
                        "gap_v3": g3, "gap_v4": g4, "gap_v41": g41})

        print(f"{name:<24}{ref*100:7.2f}%{p3*100:8.2f}%{p4*100:8.2f}%{p41*100:12.2f}%"
              f"{g3*100:7.2f}{g4*100:8.2f}{g41*100:9.2f}")

    print("=" * 90)
    print(f"MEAN gap:   v3={np.mean(gaps3)*100:.2f}  v4={np.mean(gaps4)*100:.2f}  "
          f"v4.1={np.mean(gaps41)*100:.2f}")
    print(f"MEDIAN gap: v3={np.median(gaps3)*100:.2f}  v4={np.median(gaps4)*100:.2f}  "
          f"v4.1={np.median(gaps41)*100:.2f}")
    print(f"WORST gap:  v3={max(gaps3)*100:.2f}  v4={max(gaps4)*100:.2f}  "
          f"v4.1={max(gaps41)*100:.2f}")

    import json
    json.dump({"chip": "sherbrooke", "shots": SHOTS, "seeds": list(SEEDS),
              "results": results,
              "mean_gap_v3": float(np.mean(gaps3)), "mean_gap_v4": float(np.mean(gaps4)),
              "mean_gap_v41": float(np.mean(gaps41)),
              "worst_gap_v41": float(max(gaps41))},
              open("quantumbridge_data/entry030_sherbrooke_validation.json", "w"),
              indent=2, default=str)
    print("\nSaved to quantumbridge_data/entry030_sherbrooke_validation.json")
