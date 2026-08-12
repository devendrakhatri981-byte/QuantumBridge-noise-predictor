"""
QuantumBridge — Entry 024: exact SWAP dwell-time routing.

WHY THIS EXISTS
---------------
v4 (Entry 023) added a decoherence term but had to GUESS how long the quantum
state dwells on each physical qubit during a SWAP chain -- it assumed v3's
BFS path and charged each hop a flat "3 gates worth" of time. That heuristic
fixed the local circuits (readout term did the real work there) but got
WORSE on long routes: bell_far_0_126 went from a 9.39-point gap to 34.60,
because the guessed dwell time had no relationship to the route Qiskit's real
router (SABRE) actually chose.

THE FIX
-------
Transpile with 'swap' kept as an explicit, undecomposed gate (normally
BasisTranslator expands every SWAP into 3 cx/ecr gates, which is exactly what
made Entries 013-022's routing invisible to inspection). Walking the
resulting circuit gate-by-gate gives the REAL sequence of physical qubits the
state visited and the REAL duration of each leg -- no guessing.

VERIFIED EXAMPLE (bell_near_77_82, logical qubits 77 and 82):
    swap [77, 78]
    swap [82, 81]
    swap [80, 81]
    swap [80, 79]
    ecr  [79, 78]      <- the real entangling gate, on a route v3's BFS
                           never predicted
Four SWAP legs, each with its own real duration, converging on a physical
edge (79,78) that is NOT the shortest hop-count path -- SABRE optimized for
something BFS does not model at all.
"""

import collections

from qiskit import transpile

import emulator_v4 as v4
from emulator_v3_routing import edge_error, is_calibrated, variable_forgiveness_ratio


def route_with_explicit_swaps(circuit, backend, seed=1):
    """Transpile keeping 'swap' as its own gate rather than letting
    BasisTranslator decompose it into cx/ecr. Once decomposed, a SWAP's three
    gates are indistinguishable from a real entangling gate in the output
    stream -- this is what made the actual routing decision invisible in
    Entries 013-022."""
    basis = list(backend.operation_names) + ["swap"]
    return transpile(circuit, backend=backend, basis_gates=basis,
                     initial_layout=list(range(backend.num_qubits)),
                     optimization_level=1, seed_transpiler=seed)


def exact_dwell_cost(logical_circuit, backend, graph, coh, pairs, measured_qubits):
    """Predict success probability using the REAL swap sequence instead of a
    BFS-path guess.

    logical_circuit : the untranspiled circuit (used only to count
        single-qubit gates for the SQ_GATE_COST term -- basis-translation
        artifacts like extra rz/sx from the router should not be charged).
    pairs           : ordered list of (logical_a, logical_b) pairs that must
        become physically adjacent, in the order the circuit issues them.
    measured_qubits : logical qubit indices measured at the end -- readout
        error is charged on wherever that qubit PHYSICALLY ends up.
    """
    t = route_with_explicit_swaps(logical_circuit, backend)

    loc = {q: q for pair in pairs for q in pair}  # logical -> current physical
    pending = list(pairs)
    success = 1.0
    notes = []
    swap_legs = 0
    dwell_log = []  # (physical_qubit, duration, dephasing) for the writeup

    for inst in t.data:
        op = inst.operation
        if op.num_qubits != 2:
            continue
        p, q = [t.find_bit(x).index for x in inst.qubits]
        d = v4.gate_duration(coh, p, q)

        if op.name == "swap":
            # BUG FOUND DURING BUILD, FIXED HERE: a physical SWAP is not one
            # gate, it is v4.GATES_PER_SWAP (3) real two-qubit pulses back to
            # back. The first version of this function charged only ONE
            # gate's worth of decoherence and NO gate error at all for each
            # SWAP -- silently dropping the error term v3 and v4-BFS both
            # already applied on every hop. That made bell_near_77_82 predict
            # 83.42% against an actual 48.74%, i.e. worse than the BFS
            # heuristic it was meant to replace. Caught by comparing against
            # the Entry 022 reference before trusting the result.
            raw = edge_error(graph, p, q)
            gate_err = raw * variable_forgiveness_ratio(raw)
            success *= (1 - gate_err) ** v4.GATES_PER_SWAP

            swap_duration = v4.GATES_PER_SWAP * d
            deco_p = v4.dephasing(coh, p, swap_duration)
            deco_q = v4.dephasing(coh, q, swap_duration)
            success *= (1 - deco_p) * (1 - deco_q)
            dwell_log.append((p, q, swap_duration, gate_err, max(deco_p, deco_q)))

            if not is_calibrated(p, q):
                notes.append(f"uncalibrated swap edge ({p},{q})")

            for lg, ph in list(loc.items()):
                if ph == p:
                    loc[lg] = q
                elif ph == q:
                    loc[lg] = p
            swap_legs += 1
            continue

        # Real interaction gate: only meaningful once a pending pair's two
        # endpoints are physically adjacent at (p, q).
        for pair in list(pending):
            a, b = pair
            if {loc.get(a), loc.get(b)} == {p, q}:
                raw = edge_error(graph, p, q)
                success *= (1 - raw * variable_forgiveness_ratio(raw))
                success *= (1 - v4.dephasing(coh, p, d)) * (1 - v4.dephasing(coh, q, d))
                if not is_calibrated(p, q):
                    notes.append(f"uncalibrated entangling edge ({p},{q})")
                pending.remove(pair)
                break

    if pending:
        notes.append(f"WARNING: {len(pending)} pair(s) never became adjacent: {pending}")

    sq_gates = sum(1 for inst in logical_circuit.data
                  if inst.operation.num_qubits == 1
                  and inst.operation.name not in ("measure", "barrier"))
    success *= (1 - v4.SQ_GATE_COST) ** sq_gates

    for lq in measured_qubits:
        phys = loc.get(lq, lq)
        ro = coh["readout"].get(phys)
        if ro:
            success *= (1 - ro)

    notes.append(f"{swap_legs} real swap-legs, {sq_gates} single-qubit gates")
    return success, notes, dwell_log


if __name__ == "__main__":
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime.fake_provider import FakeKyiv
    import emulator_v3_routing as em

    backend = FakeKyiv()
    NQ = backend.num_qubits
    graph = em.build_connectivity_graph(em.load_calibration("kyiv"), "kyiv")
    coh = v4.load_coherence("kyiv")

    # Aer/MPS reference, 4 seeds x 4096 shots (Entry 022)
    REF = {
        "bell_adjacent_77_78": (0.9607, [(77, 78)], [77, 78]),
        "bell_near_77_82":     (0.4874, [(77, 82)], [77, 82]),
        "bell_mid_77_100":     (0.8369, [(77, 100)], [77, 100]),
        "bell_far_0_126":      (0.6990, [(0, 126)], [0, 126]),
        "ghz_local_77_78_79":  (0.8853, [(77, 78), (78, 79)], [77, 78, 79]),
    }

    def build(pairs):
        qc = QuantumCircuit(NQ, len(pairs) + 1)
        qc.h(pairs[0][0])
        for a, b in pairs:
            qc.cx(a, b)
        return qc

    print(f"{'Circuit':<22}{'Aer':>8}{'v3':>9}{'v4(BFS)':>9}{'v4.1(exact)':>13}"
          f"{'gap v3':>8}{'gap v4':>8}{'gap v4.1':>9}")
    print("=" * 90)

    gaps3, gaps4, gaps41 = [], [], []
    for name, (ref, pairs, meas) in REF.items():
        qc = build(pairs)

        p3 = 1.0
        for inst in qc.data:
            if inst.operation.num_qubits == 2:
                i = [qc.find_bit(x).index for x in inst.qubits]
                p3 *= (1 - em.cnot_error_for_pair(graph, i[0], i[1])[0])
            elif inst.operation.num_qubits == 1:
                p3 *= (1 - v4.SQ_GATE_COST)

        p4 = v4.predict(qc, graph, coh, measured_qubits=meas)
        p41, notes, _ = exact_dwell_cost(qc, backend, graph, coh, pairs, meas)

        g3, g4, g41 = abs(p3 - ref), abs(p4 - ref), abs(p41 - ref)
        gaps3.append(g3); gaps4.append(g4); gaps41.append(g41)

        print(f"{name:<22}{ref*100:7.2f}%{p3*100:8.2f}%{p4*100:8.2f}%{p41*100:12.2f}%"
              f"{g3*100:7.2f}{g4*100:8.2f}{g41*100:9.2f}")

    print("=" * 90)
    print(f"{'MEAN ABSOLUTE GAP':<22}{'':8}{'':9}{np.mean(gaps4)*100:8.2f}"
          f"{'':4}{np.mean(gaps3)*100:9.2f}{np.mean(gaps4)*100:8.2f}{np.mean(gaps41)*100:9.2f}")
    print(f"{'MEDIAN':<39}{np.median(gaps3)*100:16.2f}{np.median(gaps4)*100:8.2f}"
          f"{np.median(gaps41)*100:9.2f}")
    print(f"{'WORST':<39}{max(gaps3)*100:16.2f}{max(gaps4)*100:8.2f}{max(gaps41)*100:9.2f}")
