"""
QuantumBridge — Entry 025: isolating the compounding formula itself.

THE OPEN QUESTION (Entries 015/016, reopened by Entry 024)
------------------------------------------------------------
v3 and v4 both predict the success probability of a chain of n two-qubit
gates as (1 - p)^n, where p is each gate's calibrated error. Entry 016 found
this over-penalizes long chains on real hardware noise models, independent of
routing accuracy or real gate count -- and Entry 024 just confirmed routing
and gate count are NOT the problem (the exact route matched the real
transpiled circuit exactly). That leaves the formula itself as the last
unexamined piece.

But testing it directly against FakeKyiv's full noise model conflates THREE
things: gate error, decoherence (T1/T2), and readout error. If the naive
formula is wrong, is it wrong about compounding gate error, or is decoherence
just quietly doing the real work while gate error is a minor factor?

THE ISOLATION
--------------
This entry strips the question down to one variable. Build an Aer noise
model with ONLY a depolarizing_error on a single real Kyiv edge -- no
thermal relaxation, no readout error, no other gates. Apply that SAME gate
n times on a Bell pair (cancelling pairs, so the ideal outcome never
changes -- the method from Entries 017-021). Compare Aer's actual composed
channel against the naive (1-p)^n prediction.

If they match: the compounding formula is fine, and decoherence/readout were
doing all the damage in Entries 022-024. If they diverge: the formula itself
is wrong even in the cleanest possible case, with nothing else confounding
it.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeKyiv

import emulator_v3_routing as em

SHOTS = 8192
SEEDS = (1, 2, 3, 4)


def isolated_noise_model(backend, edge, p):
    """A noise model with exactly one source of error: a depolarizing
    channel of strength p on the two-qubit gate for `edge`. No thermal
    relaxation, no readout error, no error on any other edge or gate."""
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
    backend = FakeKyiv()
    edge = (77, 78)
    graph = em.build_connectivity_graph(em.load_calibration("kyiv"), "kyiv")
    p = em.edge_error(graph, *edge)  # raw calibrated gate error, no forgiveness correction

    gate_counts = [1, 3, 5, 9, 13, 21, 33, 51, 75]
    print(f"edge {edge}, raw calibrated depolarizing error p = {p * 100:.4f}%")
    print(f"{'n gates':>8}{'Aer (isolated)':>16}{'naive (1-p)^n':>16}{'ratio':>9}{'diff, pts':>11}")
    print("=" * 62)

    results = run_curve(backend, edge, p, gate_counts)
    rows = []
    for n, aer in results:
        naive = (1 - p) ** n
        ratio = aer / naive if naive else float("nan")
        diff = (aer - naive) * 100
        rows.append({"n": n, "aer": aer, "naive": naive, "ratio": ratio, "diff_pts": diff})
        print(f"{n:>8}{aer * 100:>15.3f}%{naive * 100:>15.3f}%{ratio:>9.4f}{diff:>+10.3f}")

    # fit an effective per-gate error from the Aer curve, same method as
    # Entries 017-021's decay-curve fits
    n_arr = np.array([r["n"] for r in rows], dtype=float)
    aer_arr = np.array([r["aer"] for r in rows], dtype=float)
    slope, log_a = np.polyfit(n_arr, np.log(aer_arr), 1)
    k_fit = -slope
    print(f"\nFitted effective decay constant from Aer: {k_fit * 100:.4f}% per gate")
    print(f"Nominal calibrated error:                  {p * 100:.4f}% per gate")
    print(f"Ratio (Aer effective / nominal):            {k_fit / p:.4f}")

    if abs(k_fit / p - 1) < 0.05:
        verdict = ("MATCH: the naive (1-p)^n formula is structurally correct for a "
                  "single isolated depolarizing channel. Compounding math was never "
                  "the problem -- decoherence and readout in the full noise model "
                  "were doing the work Entries 015/016/024 attributed to routing.")
    else:
        verdict = (f"MISMATCH: even with decoherence and readout completely removed, "
                  f"a chain of the SAME depolarizing gate does not compound as "
                  f"(1-p)^n. The effective per-gate cost is "
                  f"{'higher' if k_fit > p else 'lower'} than the calibrated value by "
                  f"a factor of {k_fit / p:.2f}. This is the compounding-formula bug "
                  f"Entry 016 named, isolated from every other confound.")

    print(f"\n{verdict}")

    import json
    json.dump({"edge": list(edge), "raw_calibrated_error": p, "shots": SHOTS,
              "seeds": list(SEEDS), "gate_counts": gate_counts, "results": rows,
              "fitted_effective_error": k_fit, "ratio_effective_to_nominal": k_fit / p,
              "verdict": verdict},
              open("quantumbridge_data/entry025_compounding_isolation.json", "w"), indent=2)
    print("\nSaved to quantumbridge_data/entry025_compounding_isolation.json")
