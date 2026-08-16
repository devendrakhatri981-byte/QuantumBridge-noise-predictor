"""
QuantumBridge — Entry 036: isolating fresh floor-collapse instances by channel.

WHY THIS EXISTS
---------------
Entry 035 ruled out "bad qubits 5/6" -- floor-collapse recurs on entirely
different qubits in an independent sample, at a much higher rate on
Sherbrooke than Kyiv (14.5% vs 3.1% of short routes). This entry picks
fresh instances (none involving qubits 5 or 6, since those are now known
not to be special) and isolates gate error, decoherence, and readout
separately against Aer -- the same ablation Entries 028, 029, and 033 used
-- to see whether the collapse traces to one specific channel, or is
another case of the channel-independence breakdown, or something new.

INSTANCES CHOSEN (from the combined Entry 034 + 035 outlier pool, filtered
to cases where the model predicted well ABOVE the floor -- i.e. genuine
misses, not the handful where the model already (correctly) predicted
near-floor):
    sherbrooke  3 -> 6   (dist=3, Aer 50.0%, predicted 88.5%)  [Entry 034]
    sherbrooke 40 -> 56  (dist=5, Aer 50.7%, predicted 81.9%)  [Entry 035]
    kyiv       95 -> 72  (dist=8, Aer 50.0%, predicted 60.9%)  [Entry 034]
"""

import json

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import route_with_explicit_swaps, exact_dwell_cost

SHOTS = 4096
SEEDS = (1, 2, 3, 4)
BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke}

INSTANCES = [
    ("sherbrooke", 3, 6),
    ("sherbrooke", 40, 56),
    ("kyiv", 95, 72),
]


def isolate(nm, t):
    sim = AerSimulator(noise_model=nm, method="matrix_product_state")
    vals = []
    for sd in SEEDS:
        counts = sim.run(t, shots=SHOTS, seed_simulator=sd).result().get_counts()
        tot = sum(counts.values())
        ok = sum(c for b, c in counts.items() if b.replace(" ", "") in {"00", "11"})
        vals.append(ok / tot)
    return float(sum(vals) / len(vals))


def run_one(chip, a, b):
    backend = BACKENDS[chip]()
    NQ = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    coh = v4.load_coherence(chip)

    qc = QuantumCircuit(NQ, 2)
    qc.h(a); qc.cx(a, b)
    qc.measure(a, 0); qc.measure(b, 1)
    t = transpile(qc, backend=backend, initial_layout=list(range(NQ)),
                 optimization_level=3, seed_transpiler=1)
    edges_used = sorted(set(tuple(sorted([t.find_bit(x).index for x in inst.qubits]))
                            for inst in t.data if inst.operation.num_qubits == 2))
    n_2q = sum(1 for i in t.data if i.operation.num_qubits == 2)

    real_combined = isolate(NoiseModel.from_backend(backend), t)

    # gate-error-only
    nm_gate = NoiseModel()
    for pp, qq in edges_used:
        err = em.edge_error(graph, pp, qq)
        nm_gate.add_quantum_error(depolarizing_error(err, 2), "ecr", [pp, qq])
        nm_gate.add_quantum_error(depolarizing_error(err, 2), "ecr", [qq, pp])
    aer_gate = isolate(nm_gate, t)

    # decoherence-only
    nm_th = NoiseModel()
    for pp, qq in edges_used:
        dur = v4.gate_duration(coh, pp, qq)
        t1a, t2a = coh["T1"][pp], coh["T2"][pp]
        t1b, t2b = coh["T1"][qq], coh["T2"][qq]
        err = thermal_relaxation_error(t1a, t2a, dur).expand(thermal_relaxation_error(t1b, t2b, dur))
        nm_th.add_quantum_error(err, "ecr", [pp, qq])
        nm_th.add_quantum_error(err, "ecr", [qq, pp])
    aer_deco = isolate(nm_th, t)

    # readout-only, at the real final physical location
    qc2 = QuantumCircuit(NQ, 2)
    qc2.h(a); qc2.cx(a, b)
    rt = route_with_explicit_swaps(qc2, backend)
    loc = {a: a, b: b}
    for inst in rt.data:
        op = inst.operation
        if op.num_qubits == 2 and op.name == "swap":
            p, q = [rt.find_bit(x).index for x in inst.qubits]
            for lg, ph in list(loc.items()):
                if ph == p: loc[lg] = q
                elif ph == q: loc[lg] = p
    nm_ro = NoiseModel()
    for q in (loc[a], loc[b]):
        ro = coh["readout"][q]
        nm_ro.add_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]), [q])
    aer_ro = isolate(nm_ro, t)

    qc_bare = QuantumCircuit(NQ, 2)
    qc_bare.h(a); qc_bare.cx(a, b)
    pred_full, _, _ = exact_dwell_cost(qc_bare, backend, graph, coh, [(a, b)], [a, b])

    naive_product_aer = aer_gate * aer_deco * aer_ro

    print(f"\n=== {chip} {a} -> {b}  ({len(edges_used)} edges, {n_2q} 2q gates) ===")
    print(f"{'channel':<16}{'Aer isolated':>14}")
    print(f"{'gate error':<16}{aer_gate*100:>13.2f}%")
    print(f"{'decoherence':<16}{aer_deco*100:>13.2f}%")
    print(f"{'readout':<16}{aer_ro*100:>13.2f}%")
    print(f"{'naive product':<16}{naive_product_aer*100:>13.2f}%")
    print(f"{'REAL combined':<16}{real_combined*100:>13.2f}%")
    print(f"{'v4.1 predicts':<16}{pred_full*100:>13.2f}%")
    print(f"gap from naive product of isolated channels to real: "
          f"{(naive_product_aer-real_combined)*100:+.2f} pts")

    return {"chip": chip, "a": a, "b": b, "n_edges": len(edges_used), "n_2q_gates": n_2q,
           "aer_gate_only": aer_gate, "aer_decoherence_only": aer_deco,
           "aer_readout_only": aer_ro, "naive_product_of_isolated": naive_product_aer,
           "real_combined_aer": real_combined, "v4_1_prediction": pred_full,
           "gap_naive_product_vs_real": naive_product_aer - real_combined}


if __name__ == "__main__":
    results = [run_one(chip, a, b) for chip, a, b in INSTANCES]
    json.dump(results, open("quantumbridge_data/entry036_isolate_floor_collapse.json", "w"),
              indent=2, default=str)
    print("\nSaved to quantumbridge_data/entry036_isolate_floor_collapse.json")
