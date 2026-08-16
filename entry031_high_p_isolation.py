"""
QuantumBridge — Entry 031: does the closed-form baseline hold at high p?

THE QUESTION
------------
Entry 030 isolated gate error alone for Sherbrooke's bell_near_57_61 route
and found edge (57,58) -- reported gate_error 11.68%, ~15x the chip median
and the worst edge in that reference set -- produces far less real damage
(98.71% isolated survival) than the closed-form baseline predicts for a
depolarizing channel of that magnitude (83.5%). Every edge the closed form
was previously verified against (Entry 025's Kyiv edge, the six D-screened
edges in Entry 021, Entry 026's ten-edge residual check) topped out around
4.3% raw error. This entry asks directly: does success = 0.5 + 0.5*(1-p)^n
still hold when p itself is this large, or does the formula only work in
the regime it happened to be tested in?

METHOD
------
Identical to Entry 025's isolation: an Aer noise model with ONLY a
depolarizing_error(p, 2) on edge (57,58), nothing else -- no thermal
relaxation, no readout. Same gate applied n times on a Bell pair via
cancelling CX pairs (ideal outcome stays {00,11}). Compare Aer's real
composed channel against both the naive (1-p)^n and the closed-form
0.5 + 0.5*(1-p)^n at the SAME p that produced the 83.5%/98.71% mismatch.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

import emulator_v3_routing as em

SHOTS = 8192
SEEDS = (1, 2, 3, 4)
FLOOR = 0.5  # 2-of-4 ideal outcome set, same as every Bell circuit in this project


def isolated_noise_model(backend, edge, p):
    nm = NoiseModel()
    err = depolarizing_error(p, 2)
    for gate_name in ("cx", "ecr"):
        if gate_name in backend.operation_names:
            nm.add_quantum_error(err, gate_name, list(edge))
    return nm


def build_circuit(q1, q2, extra_pairs, nq):
    qc = QuantumCircuit(nq, 2)
    qc.h(q1)
    qc.cx(q1, q2)
    for _ in range(extra_pairs):
        qc.cx(q1, q2)
        qc.cx(q1, q2)
    qc.measure([q1, q2], [0, 1])
    return qc


def run_curve(backend, edge, p, gate_counts):
    nm = isolated_noise_model(backend, edge, p)
    sim = AerSimulator(noise_model=nm)
    results = []
    for n in gate_counts:
        extra = (n - 1) // 2
        qc = build_circuit(*edge, extra, backend.num_qubits)
        t = transpile(qc, backend=backend, initial_layout=list(range(backend.num_qubits)),
                     optimization_level=0)
        vals = []
        for sd in SEEDS:
            counts = sim.run(t, shots=SHOTS, seed_simulator=sd).result().get_counts()
            tot = sum(counts.values())
            vals.append(sum(c for b, c in counts.items()
                           if b.replace(" ", "") in {"00", "11"}) / tot)
        results.append((n, float(np.mean(vals))))
    return results


if __name__ == "__main__":
    backend = FakeSherbrooke()
    edge = (57, 58)
    graph = em.build_connectivity_graph(em.load_calibration("sherbrooke"), "sherbrooke")
    p = em.edge_error(graph, *edge)

    gate_counts = [1, 3, 5, 9, 13, 21, 33, 51, 75]
    print(f"edge {edge}, raw calibrated depolarizing error p = {p * 100:.4f}%")
    print(f"{'n gates':>8}{'Aer':>10}{'naive(1-p)^n':>14}{'closed-form':>13}"
          f"{'diff naive':>12}{'diff closed':>13}")
    print("=" * 75)

    results = run_curve(backend, edge, p, gate_counts)
    rows = []
    for n, aer in results:
        naive = (1 - p) ** n
        closed = FLOOR + (1 - FLOOR) * (1 - p) ** n
        diff_naive = (aer - naive) * 100
        diff_closed = (aer - closed) * 100
        rows.append({"n": n, "aer": aer, "naive": naive, "closed_form": closed,
                    "diff_naive_pts": diff_naive, "diff_closed_pts": diff_closed})
        print(f"{n:>8}{aer*100:>9.3f}%{naive*100:>13.3f}%{closed*100:>12.3f}%"
              f"{diff_naive:>+11.3f}{diff_closed:>+12.3f}")

    mean_abs_closed = float(np.mean([abs(r["diff_closed_pts"]) for r in rows]))
    worst_closed = max(rows, key=lambda r: abs(r["diff_closed_pts"]))

    print(f"\nMean |closed-form error|: {mean_abs_closed:.3f} pts")
    print(f"Worst point: n={worst_closed['n']}, diff={worst_closed['diff_closed_pts']:+.3f} pts")

    if mean_abs_closed < 2.0:
        verdict = (f"MATCH: the closed-form baseline (0.5 + 0.5*(1-p)^n) still holds at "
                  f"p={p*100:.2f}% to within {mean_abs_closed:.2f} points on average -- "
                  f"the formula is NOT the source of Entry 030's bell_near_57_61 gap. "
                  f"The 83.5%-vs-98.71% mismatch found there must come from something "
                  f"else: how the isolated single-edge result combines with the rest of "
                  f"the route (the SWAP-power-3 compounding, or an interaction with "
                  f"decoherence/readout), not from the baseline formula breaking down "
                  f"at high p in isolation.")
    else:
        verdict = (f"MISMATCH: the closed-form baseline breaks down at p={p*100:.2f}%, "
                  f"missing Aer's real result by {mean_abs_closed:.2f} points on average. "
                  f"The absorbing-state derivation (Entry 025) assumed the depolarizing "
                  f"approximation holds uniformly across error magnitudes -- it does not "
                  f"at this scale. Kyiv's edges (topping out near 4.3%) never tested this "
                  f"regime. Any future chip with an edge above ~5-8% raw error should be "
                  f"treated as suspect until re-verified this way.")

    print(f"\n{verdict}")

    import json
    json.dump({"chip": "sherbrooke", "edge": list(edge), "raw_calibrated_error": p,
              "shots": SHOTS, "seeds": list(SEEDS), "gate_counts": gate_counts,
              "results": rows, "mean_abs_closed_form_error_pts": mean_abs_closed,
              "verdict": verdict},
              open("quantumbridge_data/entry031_high_p_isolation.json", "w"), indent=2)
    print("\nSaved to quantumbridge_data/entry031_high_p_isolation.json")
