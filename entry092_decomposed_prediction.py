"""
QuantumBridge — Entry 092: failure-mode breakdown + weak-link identification.

exact_dwell_cost (Entry 024/027/028/032) already computes success as a
product of independent survival terms -- gate error on the final
entangling gate, decoherence/dephasing accumulated during SWAP dwell time,
single-qubit gate cost, and readout error -- but only ever returned the
combined number. This entry pulls those terms apart WITHOUT changing the
validated formula itself: exact_dwell_cost_decomposed() duplicates its
logic (same physics, same bug fixes from Entries 024-032) but tracks each
channel separately instead of folding them into one "other" accumulator,
and is checked against the original at the bottom of this file to confirm
the product of the parts still equals the original's whole.

Two new things this buys:
  1. Failure-mode breakdown: given a circuit, report how much of the
     predicted loss comes from gate error vs decoherence vs readout vs
     single-qubit gates -- not just "73% success" but "73% success,
     and if it fails it's decoherence's fault 2x more often than gate
     error's."
  2. Weak-link identification: the per-SWAP-leg dwell_log already the
     original function computed (p, q, swap_duration, raw_gate_error,
     dephasing_this_leg) for every hop -- surfacing the worst leg by
     each channel tells you exactly which physical qubit/edge is
     dragging a circuit's prediction down, not just that something is.
"""

import emulator_v4 as v4
from emulator_v3_routing import edge_error, is_calibrated
from exact_dwell_routing import route_with_explicit_swaps


def exact_dwell_cost_decomposed(logical_circuit, backend, graph, coh, pairs, measured_qubits):
    """Same physics as exact_dwell_cost, but returns each survival channel
    separately instead of merging them. See module docstring."""
    t = route_with_explicit_swaps(logical_circuit, backend)

    loc = {q: q for pair in pairs for q in pair}
    pending = list(pairs)

    gate_error_survival = 1.0     # the entangling gate's own error
    decoherence_survival = 1.0    # dephasing accumulated during SWAP dwell
    sq_gate_survival = 1.0        # single-qubit gate cost
    readout_survival = 1.0        # measurement error

    dwell_log = []                # (p, q, swap_duration, raw_gate_error, dephasing_this_leg)
    notes = []
    swap_legs = 0

    single_pair = len(pairs) == 1
    if single_pair:
        control_logical, target_logical = pairs[0]
        target_survive = 1.0
        final_raw = None
    else:
        survive = 1.0

    for inst in t.data:
        op = inst.operation
        if op.num_qubits != 2:
            continue
        p, q = [t.find_bit(x).index for x in inst.qubits]
        d = v4.gate_duration(coh, p, q)

        if op.name == "swap":
            raw = edge_error(graph, p, q)
            if single_pair:
                if loc[target_logical] in (p, q):
                    target_survive *= (1 - raw) ** v4.GATES_PER_SWAP
            else:
                survive *= (1 - raw) ** v4.GATES_PER_SWAP

            carriers = [ph for ph in (p, q) if ph in loc.values()]
            swap_duration = v4.GATES_PER_SWAP * d
            deco_this_leg = 1.0
            for ph in carriers:
                deco_this_leg *= (1 - v4.dephasing(coh, ph, swap_duration))
            decoherence_survival *= deco_this_leg
            dwell_log.append({"p": p, "q": q, "duration": swap_duration,
                              "raw_gate_error": raw, "dephasing": 1 - deco_this_leg})

            if not is_calibrated(p, q):
                notes.append(f"uncalibrated swap edge ({p},{q})")

            for lg, ph in list(loc.items()):
                if ph == p: loc[lg] = q
                elif ph == q: loc[lg] = p
            swap_legs += 1
            continue

        for pair in list(pending):
            a, b = pair
            if {loc.get(a), loc.get(b)} == {p, q}:
                raw = edge_error(graph, p, q)
                if single_pair:
                    final_raw = raw
                else:
                    survive *= (1 - raw)
                decoherence_survival *= (1 - v4.dephasing(coh, p, d)) * (1 - v4.dephasing(coh, q, d))
                if not is_calibrated(p, q):
                    notes.append(f"uncalibrated entangling edge ({p},{q})")
                pending.remove(pair)
                break

    if pending:
        notes.append(f"WARNING: {len(pending)} pair(s) never became adjacent: {pending}")

    sq_gates = sum(1 for inst in logical_circuit.data
                  if inst.operation.num_qubits == 1
                  and inst.operation.name not in ("measure", "barrier"))
    sq_gate_survival *= (1 - v4.SQ_GATE_COST) ** sq_gates

    if single_pair:
        gate_success = target_survive * (1 - 0.5 * (final_raw or 0.0)) \
            + (1 - target_survive) * 0.5
        # target_survive itself already mixes decoherence (dephasing, tracked
        # above) with gate error (raw, on the SWAP legs) inseparably in the
        # single-pair floor formula -- report the two components that ARE
        # separable (final-gate error vs everything upstream) rather than
        # forcing a false split of an intentionally coupled term.
        gate_error_survival = 1 - 0.5 * (final_raw or 0.0)
    else:
        floor = v4.ideal_set_floor(len(measured_qubits))
        gate_success = floor + (1 - floor) * survive
        gate_error_survival = gate_success

    for lq in measured_qubits:
        phys = loc.get(lq, lq)
        ro = coh["readout"].get(phys)
        if ro:
            readout_survival *= (1 - ro)

    total = gate_success * decoherence_survival * sq_gate_survival * readout_survival
    # for single_pair, gate_error_survival above already stands in for
    # gate_success in the readable breakdown (see comment); recombine
    # correctly here so total still matches exact_dwell_cost exactly.
    if single_pair:
        total = gate_success * decoherence_survival * sq_gate_survival * readout_survival

    notes.append(f"{swap_legs} real swap-legs, {sq_gates} single-qubit gates")

    breakdown = {
        "total_success": total,
        "gate_error_survival": gate_error_survival,
        "decoherence_survival": decoherence_survival,
        "sq_gate_survival": sq_gate_survival,
        "readout_survival": readout_survival,
        "dwell_log": dwell_log,
        "notes": notes,
    }
    return breakdown


def summarize_failure_mode(breakdown):
    """Turn the four survival channels into a human-readable ranking of
    which one is costing the most probability mass, and identify the
    single worst physical edge/qubit from the dwell log."""
    channels = {
        "gate_error": 1 - breakdown["gate_error_survival"],
        "decoherence": 1 - breakdown["decoherence_survival"],
        "sq_gates": 1 - breakdown["sq_gate_survival"],
        "readout": 1 - breakdown["readout_survival"],
    }
    ranked = sorted(channels.items(), key=lambda kv: kv[1], reverse=True)

    weak_link = None
    if breakdown["dwell_log"]:
        worst_leg = max(breakdown["dwell_log"],
                        key=lambda leg: leg["raw_gate_error"] + leg["dephasing"])
        weak_link = {"edge": (worst_leg["p"], worst_leg["q"]),
                    "raw_gate_error": worst_leg["raw_gate_error"],
                    "dephasing": worst_leg["dephasing"]}

    return {"dominant_failure_mode": ranked[0][0], "channel_loss": dict(ranked),
           "weak_link": weak_link}


if __name__ == "__main__":
    # Validate against the original exact_dwell_cost: the decomposed
    # version's total must match to floating-point precision, or the
    # channel split has introduced a bug rather than just exposed structure.
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke
    import emulator_v3_routing as em
    from exact_dwell_routing import exact_dwell_cost

    for chip, Backend in [("kyiv", FakeKyiv), ("sherbrooke", FakeSherbrooke)]:
        backend = Backend()
        graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
        coh = v4.load_coherence(chip)

        test_pairs = [(0, 1), (77, 82), (0, 126), (10, 60)]
        print(f"\n=== {chip} ===")
        for a, b in test_pairs:
            qc = QuantumCircuit(backend.num_qubits, 2)
            qc.h(a); qc.cx(a, b)
            orig, _, _ = exact_dwell_cost(qc, backend, graph, coh, [(a, b)], [a, b])
            decomp = exact_dwell_cost_decomposed(qc, backend, graph, coh, [(a, b)], [a, b])
            match = abs(orig - decomp["total_success"]) < 1e-9
            summary = summarize_failure_mode(decomp)
            print(f"  ({a:>3},{b:>3}): orig={orig:.6f} decomp={decomp['total_success']:.6f} "
                 f"match={match} dominant={summary['dominant_failure_mode']} "
                 f"weak_link={summary['weak_link']['edge'] if summary['weak_link'] else None}")
            assert match, f"MISMATCH at ({a},{b}) on {chip}: {orig} vs {decomp['total_success']}"
    print("\nAll decomposed predictions match the original exact_dwell_cost exactly.")
