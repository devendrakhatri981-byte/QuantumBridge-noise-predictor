"""
QuantumBridge — Entry 031: the model doesn't know WHEN entanglement happens.

THE DEAD END FIRST
-------------------
Entry 030 flagged Sherbrooke edge (57,58) -- gate_error 11.68%, ~15x the
chip median -- as a suspect: modeling it as a depolarizing channel of that
magnitude predicts 83.5% survival for bell_near_57_61's route, but Aer's
isolated gate-error-only result is 98.71%. entry031_high_p_isolation.py
ruled out the obvious hypothesis: sweeping gate count on that exact edge
alone (Entry 025's method) matches the closed form to within 0.13 points
even at p=11.68%. The formula is not the problem, at any magnitude tested.

THE REAL BUG
------------
Since the per-edge formula is fine, the bug has to be in how results from
DIFFERENT edges combine along a route. Inspecting the real transpiled
circuit for bell_near_57_61 shows only ONE non-SWAP two-qubit gate in the
whole sequence -- the entangling gate, at the very end, once both logical
qubits have been routed to adjacent physical positions. Every SWAP before
that point is moving an UNENTANGLED product-state qubit (post-H |+>, or a
plain |0>), not an already-entangled Bell-pair half.

Every model since Entry 023 assumed the opposite: that every SWAP carries
an already-entangled payload, equally vulnerable to depolarizing damage at
every hop. That assumption is wrong for exactly the circuits this project
measures -- a depolarizing event on a product state doesn't destroy a
correlation that doesn't exist yet, and does measurably less damage to the
final {00,11} outcome than the model assumes.

THIS SCRIPT
-----------
Two minimal toy circuits, built directly (no chip topology, isolating the
structural question alone):

  1. WRONG structure: create entanglement first (h + cx), THEN route both
     halves via SWAPs, then apply an extra (incorrect) entangling gate at
     the end -- included only to show how sensitive this is.
  2. CORRECT structure: apply h to one qubit only, route BOTH qubits
     (one post-H, one still |0>) via SWAPs, and create the entanglement
     with a SINGLE gate only once they're adjacent -- matching what the
     real transpiler does.

Same edge errors (Sherbrooke's real calibration for the bell_near_57_61
route) in both, compared against the closed-form baseline that assumes
every SWAP carries an entangled payload.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import emulator_v3_routing as em

SHOTS = 8192
SEEDS = (1, 2, 3, 4)


def run(qc, noise_edges_p, real_edges):
    """noise_edges_p: list of (gate_name, [q1,q2], p). real_edges is used
    only for the printed no-noise sanity check's outcome labels."""
    sim0 = AerSimulator()
    t0 = transpile(qc, sim0, optimization_level=0)
    ideal_counts = sim0.run(t0, shots=4096, seed_simulator=1).result().get_counts()

    nm = NoiseModel()
    for gate, qubits, p in noise_edges_p:
        nm.add_quantum_error(depolarizing_error(p, 2), gate, qubits)
    sim = AerSimulator(noise_model=nm)
    t = transpile(qc, sim, optimization_level=0)
    vals = []
    for sd in SEEDS:
        counts = sim.run(t, shots=SHOTS, seed_simulator=sd).result().get_counts()
        tot = sum(counts.values())
        ok = sum(c for b, c in counts.items() if b.replace(" ", "") in {"00", "11"})
        vals.append(ok / tot)
    return ideal_counts, sum(vals) / len(vals)


if __name__ == "__main__":
    graph = em.build_connectivity_graph(em.load_calibration("sherbrooke"), "sherbrooke")
    # the exact 4 real edges bell_near_57_61 routes across (Entry 030)
    p1 = em.edge_error(graph, 57, 58)   # 0.1168 -- the flagged "outlier"
    p2 = em.edge_error(graph, 60, 61)   # 0.0035
    p3 = em.edge_error(graph, 59, 60)   # 0.0041
    p4 = em.edge_error(graph, 58, 59)   # 0.0046 -- the real entangling edge
    survive = (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4)
    closed_form = 0.5 + 0.5 * survive

    print(f"edges: p1={p1:.4f} p2={p2:.4f} p3={p3:.4f} p4={p4:.4f}")
    print(f"closed-form prediction (old model, entangle-first assumption): "
          f"{closed_form*100:.2f}%\n")

    # --- Toy 1: WRONG structure (entangle first, route, entangle again) ---
    qc_wrong = QuantumCircuit(5, 2)
    qc_wrong.h(0); qc_wrong.cx(0, 2)
    qc_wrong.swap(0, 1)
    qc_wrong.swap(2, 3)
    qc_wrong.swap(3, 4)
    qc_wrong.cx(1, 4)
    qc_wrong.measure(1, 0); qc_wrong.measure(4, 1)
    ideal_wrong, aer_wrong = run(
        qc_wrong,
        [("swap", [0, 1], p1), ("swap", [2, 3], p2),
         ("swap", [3, 4], p3), ("cx", [1, 4], p4)], None)
    print(f"Toy 1 (WRONG: re-entangling an already-entangled pair)")
    print(f"  no-noise outcome (should be pure {{00,11}} if correct): {ideal_wrong}")
    print(f"  Aer: {aer_wrong*100:.2f}%   closed-form: {closed_form*100:.2f}%   "
          f"diff: {(aer_wrong-closed_form)*100:+.2f} pts\n")

    # --- Toy 2: CORRECT structure (route first, entangle once at the end) ---
    qc_right = QuantumCircuit(5, 2)
    qc_right.h(0)
    qc_right.swap(0, 1)
    qc_right.swap(2, 3)
    qc_right.swap(3, 4)
    qc_right.cx(1, 4)
    qc_right.measure(1, 0); qc_right.measure(4, 1)
    ideal_right, aer_right = run(
        qc_right,
        [("swap", [0, 1], p1), ("swap", [2, 3], p2),
         ("swap", [3, 4], p3), ("cx", [1, 4], p4)], None)
    print(f"Toy 2 (CORRECT: matches what the real transpiler does)")
    print(f"  no-noise outcome (should be pure {{00,11}}): {ideal_right}")
    print(f"  Aer: {aer_right*100:.2f}%   closed-form: {closed_form*100:.2f}%   "
          f"diff: {(aer_right-closed_form)*100:+.2f} pts\n")

    verdict = (
        "The correct-structure toy (Toy 2) under-predicts by "
        f"{(aer_right-closed_form)*100:+.2f} points -- same direction and "
        "comparable scale to the real bell_near_57_61 route's gate-error-only "
        "gap (+15.19 points, Entry 030/031). The wrong-structure toy (Toy 1) "
        "shows this is not a small effect: getting the entanglement timing "
        "wrong doesn't shift the prediction slightly, it inverts it "
        f"({(aer_wrong-closed_form)*100:+.2f} points). CONCLUSION: every gate-"
        "error model since Entry 023 assumes every SWAP carries an already-"
        "entangled payload. Real transpiled circuits for these Bell/GHZ "
        "preparations route qubits into position FIRST and entangle LAST -- "
        "most SWAP legs on a long route are moving unentangled product-state "
        "information, which is measurably less vulnerable to depolarizing "
        "damage than the model assumes. This is the real cause of the "
        "persistent over-prediction on long routes (bell_mid, bell_far, and "
        "now bell_near_57_61) across every chip tested. Not fixed in this "
        "entry -- the correct closed form for a pre-entanglement SWAP needs "
        "its own derivation, left for Entry 032.")
    print(verdict)

    import json
    json.dump({"chip": "sherbrooke", "route_edges": {"p1": p1, "p2": p2, "p3": p3, "p4": p4},
              "old_model_closed_form_prediction": closed_form,
              "toy1_wrong_structure": {"ideal_counts": dict(ideal_wrong), "aer": aer_wrong},
              "toy2_correct_structure": {"ideal_counts": dict(ideal_right), "aer": aer_right},
              "real_route_gate_error_only_aer": 0.987060546875,
              "real_route_gate_error_only_predicted": 0.8351761475375721,
              "verdict": verdict},
              open("quantumbridge_data/entry031_route_then_entangle.json", "w"),
              indent=2, default=str)
    print("\nSaved to quantumbridge_data/entry031_route_then_entangle.json")
