"""
QuantumBridge — Entry 022: first trustworthy circuit-level validation of v3.

WHY THIS REPLACES validate_v3_circuits.py
-----------------------------------------
The Cairo harness was unusable for two independent reasons, both found in
Entry 022:

1. BASIS MISMATCH. FakeCairoV2 is a mixed cx/ecr snapshot. Passing the union
   of basis gates let the transpiler emit `cx` on ecr-only edges, where the
   noise model has no matching entry, so those gates ran with no error at all
   -- 53% of all two-qubit gates in the suite. See diagnose_basis_mismatch.py.

2. CALIBRATION COVERAGE. Only 12 of Cairo's 28 topology edges carry real
   calibrated errors (42.9%). The other 16 used v3's 1% fallback constant, so
   most of the "model" was a guess on that chip.

FakeKyiv has neither problem: uniformly `ecr`, noise-model coverage on all
144 edges, and 139/144 (96.5%) real calibration coverage.

This script also AUDITS ITSELF. Every run reports how many two-qubit gates
carried noise versus ran silently. If that silent count is ever non-zero, the
accuracy numbers below are void and the script says so.
"""

import collections
import json

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeKyiv

import emulator_v3_routing as em
from validate_v3_circuits import SQ_GATE_COST

CHIP = "kyiv"
SEEDS = (1, 2, 3, 4)
SHOTS = 4096

backend = FakeKyiv()
noise_model = NoiseModel.from_backend(backend)

# method="matrix_product_state" is REQUIRED, not a preference. Aer's
# "automatic" method silently returns an empty result for the deeper routed
# circuits here (76 ecr gates, depth 178, across 127 qubits) -- run.success is
# True and status is COMPLETED, but get_counts() raises "No counts for
# experiment". A validation harness that trusted `automatic` would simply lose
# its hardest circuits. MPS handles them because SWAP-routed Bell and GHZ
# states stay low-entanglement.
sim = AerSimulator(noise_model=noise_model, method="matrix_product_state")
NQ = backend.num_qubits


def noise_coverage():
    cov = collections.defaultdict(set)
    for err in noise_model.to_dict()["errors"]:
        ops = err.get("operations", [])
        if ops and len(err["gate_qubits"][0]) == 2:
            cov[ops[0]].add(tuple(err["gate_qubits"][0]))
    return cov


COV = noise_coverage()


def v3_predict(circuit, graph):
    """v3's routing-aware prediction, walking the LOGICAL circuit."""
    p = 1.0
    for inst in circuit.data:
        gate = inst.operation
        if gate.name in ("measure", "barrier"):
            continue
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        if gate.num_qubits == 2:
            err, _ = em.cnot_error_for_pair(graph, qubits[0], qubits[1])
            p *= (1 - err)
        elif gate.num_qubits == 1:
            p *= (1 - SQ_GATE_COST)
    return p


def aer_reference(qc, ideal):
    """Transpile against the backend itself, audit noise coverage, simulate.

    initial_layout pins logical qubit i to physical qubit i. Without it the
    transpiler is free to relabel -- it will happily map qubits 0 and 126 onto
    an adjacent physical pair, producing a single gate and no routing at all.
    v3 reasons about the physical qubits it was handed, so the reference has to
    use the same placement or the two are answering different questions."""
    t = transpile(qc, backend=backend, initial_layout=list(range(NQ)),
                  optimization_level=3, seed_transpiler=1)
    audit = collections.Counter()
    for inst in t.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(t.find_bit(q).index for q in inst.qubits)
            name = inst.operation.name
            hit = pair in COV[name] or pair[::-1] in COV[name]
            audit["noisy" if hit else "silent"] += 1
    vals = []
    for sd in SEEDS:
        counts = sim.run(t, shots=SHOTS, seed_simulator=sd).result().get_counts()
        tot = sum(counts.values())
        vals.append(sum(c for b, c in counts.items()
                        if b.replace(" ", "") in ideal) / tot)
    return float(np.mean(vals)), float(np.std(vals)), audit, t.count_ops()


def build_circuits(graph):
    """Bell and GHZ circuits at increasing routing distance on Kyiv."""
    def hops(a, b):
        return len(em.shortest_path(graph, a, b)) - 1

    specs = [
        ("bell_adjacent_77_78", [(77, 78)], {"00", "11"}),
        ("bell_near_77_82", [(77, 82)], {"00", "11"}),
        ("bell_mid_77_100", [(77, 100)], {"00", "11"}),
        ("bell_far_0_126", [(0, 126)], {"00", "11"}),
    ]
    out = {}
    for name, pairs, ideal in specs:
        a, b = pairs[0]
        qc = QuantumCircuit(NQ, 2)
        qc.h(a); qc.cx(a, b); qc.measure([a, b], [0, 1])
        out[name] = (qc, ideal, hops(a, b))

    # GHZ: local chain vs scattered
    qc = QuantumCircuit(NQ, 3)
    qc.h(77); qc.cx(77, 78); qc.cx(78, 79); qc.measure([77, 78, 79], [0, 1, 2])
    out["ghz_local_77_78_79"] = (qc, {"000", "111"}, hops(77, 78) + hops(78, 79))

    qc = QuantumCircuit(NQ, 3)
    qc.h(0); qc.cx(0, 63); qc.cx(63, 126); qc.measure([0, 63, 126], [0, 1, 2])
    out["ghz_scattered_0_63_126"] = (qc, {"000", "111"}, hops(0, 63) + hops(63, 126))
    return out


if __name__ == "__main__":
    graph = em.build_connectivity_graph(em.load_calibration(CHIP), CHIP,
                                        report_coverage=True)
    print(f"forgiveness law: ratio = {em.FORGIVENESS_COEFFICIENT} * "
          f"err^({em.FORGIVENESS_EXPONENT})\n")

    circuits = build_circuits(graph)
    print(f"{'Circuit':<26}{'hops':>6}{'2q':>5}{'silent':>7}"
          f"{'Aer ref':>10}{'+/-sd':>7}{'v3':>9}{'gap':>8}")
    print("=" * 78)

    gaps, total_silent = [], 0
    rows = []
    for name, (qc, ideal, h) in circuits.items():
        ref, sd, audit, ops = aer_reference(qc, ideal)
        pred = v3_predict(qc, graph)
        gap = abs(pred - ref)
        gaps.append(gap)
        total_silent += audit["silent"]
        n2q = audit["noisy"] + audit["silent"]
        rows.append((name, h, n2q, audit["silent"], ref, sd, pred, gap))
        print(f"{name:<26}{h:>6}{n2q:>5}{audit['silent']:>7}"
              f"{ref * 100:9.2f}%{sd * 100:6.2f}%{pred * 100:8.2f}%{gap * 100:7.2f}")

    print("=" * 78)
    print(f"  Mean absolute gap: {np.mean(gaps) * 100:.2f} percentage points")
    print(f"  Median:            {np.median(gaps) * 100:.2f} pts")
    print(f"  Worst:             {max(gaps) * 100:.2f} pts")

    if total_silent:
        print(f"\n  *** {total_silent} two-qubit gates ran WITHOUT noise. "
              f"These numbers are NOT valid. ***")
    else:
        print(f"\n  All two-qubit gates carried noise. Reference is sound.")

    json.dump({"chip": CHIP, "shots": SHOTS, "seeds": list(SEEDS),
               "silent_gates": total_silent,
               "results": [{"circuit": r[0], "hops": r[1], "two_qubit_gates": r[2],
                            "silent": r[3], "aer_ref": r[4], "aer_sd": r[5],
                            "v3_pred": r[6], "gap": r[7]} for r in rows],
               "mean_gap": float(np.mean(gaps))},
              open("quantumbridge_data/entry022_kyiv_validation.json", "w"), indent=2)
    print("\nSaved to quantumbridge_data/entry022_kyiv_validation.json")
