"""
QuantumBridge — Seeded A/B test: old Cairo two-point forgiveness law vs the
Entry 021 Kyiv four-point law, measured against the same reference.

Why this file exists: validate_v3_circuits.py does not seed the Aer simulator,
so consecutive runs of the same circuit differ by a few tenths of a point. That
is the same order as the effect being measured here, which makes a before/after
comparison run on two separate invocations meaningless.

This script fixes that by averaging the Aer reference over 8 seeds at 8192 shots
and reporting its standard deviation, then evaluating both laws against that one
fixed reference. v3's own prediction is deterministic, so the only noise in the
comparison is in the reference, and it is quantified.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

import emulator_v3_routing as em
from validate_v3_circuits import v3_predicted_success_probability, FIXED_COUPLING_MAP

SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
SHOTS = 8192

# (coefficient, exponent) for ratio = c * raw_edge_error^e
LAWS = {
    "OLD (Cairo, 2pt)": (0.1264, -0.2804),   # Entries 017/019 — high anchor contaminated
    "NEW (Kyiv, 4pt)": (0.0898, -0.3873),    # Entry 021 — D-screened, R^2 = 0.9648
}

# bell_scattered_0_26 is a known pre-existing failure of v3's SWAP routing
# (see investigate_bell_scattered.py). It is reported but also excluded from a
# second mean, because it dominates the average and is not a forgiveness issue.
KNOWN_OUTLIER = "bell_scattered_0_26"

backend = FakeCairoV2()
sim = AerSimulator(noise_model=NoiseModel.from_backend(backend))
graph = em.build_connectivity_graph(em.load_calibration())


def aer_reference(qc, ideal_outcomes):
    """Mean and standard deviation of success probability across SEEDS."""
    vals = []
    for sd in SEEDS:
        t = transpile(qc, coupling_map=FIXED_COUPLING_MAP,
                      basis_gates=backend.operation_names,
                      initial_layout=list(range(qc.num_qubits)),
                      optimization_level=3, seed_transpiler=sd)
        counts = sim.run(t, shots=SHOTS, seed_simulator=sd).result().get_counts()
        total = sum(counts.values())
        vals.append(sum(c for b, c in counts.items()
                        if b.replace(" ", "") in ideal_outcomes) / total)
    return float(np.mean(vals)), float(np.std(vals))


def build_tests():
    qc1 = QuantumCircuit(27, 3); qc1.h(0); qc1.cx(0, 1); qc1.cx(1, 2)
    qc1.measure([0, 1, 2], [0, 1, 2])
    qc2 = QuantumCircuit(27, 3); qc2.h(0); qc2.cx(0, 26); qc2.cx(26, 22)
    qc2.measure([0, 26, 22], [0, 1, 2])
    qc3 = QuantumCircuit(27, 2); qc3.h(24); qc3.cx(24, 25)
    qc3.measure([24, 25], [0, 1])
    qc4 = QuantumCircuit(27, 2); qc4.h(0); qc4.cx(0, 26)
    qc4.measure([0, 26], [0, 1])
    return {
        "ghz_local_0_1_2": (qc1, {"000", "111"}),
        "ghz_scattered_0_26_22": (qc2, {"000", "111"}),
        "bell_adjacent_24_25": (qc3, {"00", "11"}),
        KNOWN_OUTLIER: (qc4, {"00", "11"}),
    }


if __name__ == "__main__":
    tests = build_tests()
    print(f"{'Circuit':<24}{'Aer ref':>10}{'+/-sd':>8}"
          f"{'OLD':>9}{'gap':>8}{'NEW':>9}{'gap':>8}{'better':>9}")
    print("=" * 85)

    gaps = {k: {} for k in LAWS}
    for name, (qc, ideal) in tests.items():
        ref, sd = aer_reference(qc, ideal)
        row = []
        for label, (c, e) in LAWS.items():
            em.FORGIVENESS_COEFFICIENT, em.FORGIVENESS_EXPONENT = c, e
            pred, _ = v3_predicted_success_probability(qc, graph)
            gap = abs(pred - ref)
            gaps[label][name] = gap
            row.append((pred, gap))
        better = "NEW" if row[1][1] < row[0][1] else "old"
        print(f"{name:<24}{ref * 100:9.2f}%{sd * 100:7.2f}%"
              f"{row[0][0] * 100:8.2f}%{row[0][1] * 100:7.2f}"
              f"{row[1][0] * 100:8.2f}%{row[1][1] * 100:7.2f}{better:>9}")

    print("=" * 85)
    for label in LAWS:
        allg = list(gaps[label].values())
        clean = [g for n, g in gaps[label].items() if n != KNOWN_OUTLIER]
        print(f"  {label:<20} all circuits = {np.mean(allg) * 100:5.2f} pts   "
              f"excluding {KNOWN_OUTLIER} = {np.mean(clean) * 100:5.2f} pts")

    d_all = (np.mean(list(gaps['OLD (Cairo, 2pt)'].values()))
             - np.mean(list(gaps['NEW (Kyiv, 4pt)'].values()))) * 100
    d_clean = (np.mean([g for n, g in gaps['OLD (Cairo, 2pt)'].items() if n != KNOWN_OUTLIER])
               - np.mean([g for n, g in gaps['NEW (Kyiv, 4pt)'].items() if n != KNOWN_OUTLIER])) * 100
    print(f"\n  Re-anchoring changes the mean gap by {d_all:+.2f} pts overall, "
          f"{d_clean:+.2f} pts excluding the known routing outlier.")
    print("  (Positive = the Kyiv law is closer to the reference.)")
