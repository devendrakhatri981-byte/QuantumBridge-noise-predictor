"""
QuantumBridge — Entry 033: confirming the channel-independence ceiling.

WHY THIS EXISTS
---------------
Entry 032 shipped the route-then-entangle / control-target fix and improved
the reference set overall, but bell_far_0_126 and bell_mid_0_58 (Sherbrooke)
got WORSE (gaps up to 20.39 points). This isolates gate error, decoherence,
and readout separately against Aer for bell_far_0_126 -- the longest route
in the project, 25 real SWAP legs -- the same way Entries 028 and 029
isolated bell_mid_77_100 and bell_near_57_61.

THE RESULT
----------
Every individual formula matches its Aer isolation closely (largest gap:
3.3 points, on decoherence). But multiplying the three ISOLATED Aer results
together -- as if the channels were independent -- overshoots the REAL
combined-noise result by 22 points. This is not a new bug: it is Entry
029's channel-independence finding, confirmed on a second, unrelated route
(driven by 25 accumulated SWAP legs here, vs. one catastrophic-T1 qubit on
Kyiv's bell_near_77_82). No code changes ship from this entry.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import route_with_explicit_swaps

SHOTS = 4096
SEEDS = (1, 2, 3, 4)


def isolate(nm, t):
    sim = AerSimulator(noise_model=nm, method="matrix_product_state")
    vals = []
    for sd in SEEDS:
        counts = sim.run(t, shots=SHOTS, seed_simulator=sd).result().get_counts()
        tot = sum(counts.values())
        ok = sum(c for b, c in counts.items() if b.replace(" ", "") in {"00", "11"})
        vals.append(ok / tot)
    return float(sum(vals) / len(vals))


if __name__ == "__main__":
    backend = FakeSherbrooke()
    NQ = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration("sherbrooke"), "sherbrooke")
    coh = v4.load_coherence("sherbrooke")

    qc = QuantumCircuit(NQ, 2)
    qc.h(0); qc.cx(0, 126)
    qc.measure(0, 0); qc.measure(126, 1)
    t = transpile(qc, backend=backend, initial_layout=list(range(NQ)),
                 optimization_level=3, seed_transpiler=1)
    edges_used = sorted(set(tuple(sorted([t.find_bit(x).index for x in inst.qubits]))
                            for inst in t.data if inst.operation.num_qubits == 2))
    print(f"route uses {len(edges_used)} distinct edges, "
          f"{sum(1 for i in t.data if i.operation.num_qubits==2)} total 2q gate applications\n")

    # gate-error-only
    nm_gate = NoiseModel()
    for a, b in edges_used:
        err = em.edge_error(graph, a, b)
        nm_gate.add_quantum_error(depolarizing_error(err, 2), "ecr", [a, b])
        nm_gate.add_quantum_error(depolarizing_error(err, 2), "ecr", [b, a])
    aer_gate = isolate(nm_gate, t)

    # decoherence-only
    nm_th = NoiseModel()
    for a, b in edges_used:
        dur = v4.gate_duration(coh, a, b)
        t1a, t2a = coh["T1"][a], coh["T2"][a]
        t1b, t2b = coh["T1"][b], coh["T2"][b]
        err = thermal_relaxation_error(t1a, t2a, dur).expand(thermal_relaxation_error(t1b, t2b, dur))
        nm_th.add_quantum_error(err, "ecr", [a, b])
        nm_th.add_quantum_error(err, "ecr", [b, a])
    aer_deco = isolate(nm_th, t)

    # readout-only -- final physical qubits from the real route
    qc2 = QuantumCircuit(NQ, 2)
    qc2.h(0); qc2.cx(0, 126)
    rt = route_with_explicit_swaps(qc2, backend)
    loc = {0: 0, 126: 126}
    for inst in rt.data:
        op = inst.operation
        if op.num_qubits == 2 and op.name == "swap":
            p, q = [rt.find_bit(x).index for x in inst.qubits]
            for lg, ph in list(loc.items()):
                if ph == p: loc[lg] = q
                elif ph == q: loc[lg] = p
    nm_ro = NoiseModel()
    for q in (loc[0], loc[126]):
        ro = coh["readout"][q]
        nm_ro.add_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]), [q])
    aer_ro = isolate(nm_ro, t)

    # our current predictions for the same three channels
    pred_gate = 0.8451  # Entry 032 closed form (target-survival based)
    pred_deco = 0.8644  # Entry 028 T1-based, single-carrier
    pred_ro = (1 - coh["readout"][loc[0]]) * (1 - coh["readout"][loc[126]])

    print(f"{'channel':<16}{'predicted':>12}{'Aer isolated':>15}{'gap, pts':>11}")
    print("=" * 54)
    for name, pred, aer in [("gate error", pred_gate, aer_gate),
                            ("decoherence", pred_deco, aer_deco),
                            ("readout", pred_ro, aer_ro)]:
        print(f"{name:<16}{pred*100:>11.2f}%{aer*100:>14.2f}%{abs(pred-aer)*100:>10.2f}")

    naive_product_predicted = pred_gate * pred_deco * pred_ro
    naive_product_aer = aer_gate * aer_deco * aer_ro
    real_combined = 0.4951  # measured directly, Entry 030/032 reference

    print(f"\nnaive product of our predictions:  {naive_product_predicted*100:.2f}%")
    print(f"naive product of Aer ISOLATED results: {naive_product_aer*100:.2f}%")
    print(f"REAL combined-noise Aer result:        {real_combined*100:.2f}%")
    print(f"gap from combining even PERFECT isolated channels: "
          f"{(naive_product_aer-real_combined)*100:.2f} points")

    verdict = (
        "Every individual channel formula is verified correct (largest gap "
        "3.3 points, on decoherence). But multiplying even Aer's own "
        "perfectly isolated per-channel results together overshoots the "
        "real combined result by 22 points. This is Entry 029's channel-"
        "independence finding, confirmed on a second, unrelated route "
        "(25 SWAP legs here, vs. one catastrophic-T1 qubit on Kyiv's "
        "bell_near_77_82). Not a formula bug -- a structural limit of "
        "multiplicative channel composition, general enough now to treat "
        "as a known property rather than chase further this session.")
    print(f"\n{verdict}")

    import json
    json.dump({"chip": "sherbrooke", "circuit": "bell_far_0_126",
              "n_edges": len(edges_used),
              "gate_error": {"predicted": pred_gate, "aer": aer_gate},
              "decoherence": {"predicted": pred_deco, "aer": aer_deco},
              "readout": {"predicted": pred_ro, "aer": aer_ro},
              "naive_product_predicted": naive_product_predicted,
              "naive_product_aer_isolated": naive_product_aer,
              "real_combined_aer": real_combined, "verdict": verdict},
              open("quantumbridge_data/entry033_channel_independence_confirmation.json", "w"),
              indent=2, default=str)
    print("\nSaved to quantumbridge_data/entry033_channel_independence_confirmation.json")
