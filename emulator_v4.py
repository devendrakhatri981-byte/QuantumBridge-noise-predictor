"""
QuantumBridge Emulator v4 — gate error + time-integrated decoherence + readout.

WHAT CHANGED FROM v3 (Entry 022 -> 023)
---------------------------------------
v3 modelled two-qubit gate error alone. Entry 022 showed that leaves two
independent terms unmodelled, and that their absence produces errors in
OPPOSITE directions, so no single fudge factor could ever fix both:

  * DECOHERENCE. A route through one low-T2 qubit is catastrophic regardless of
    gate error. On Kyiv, bell_near_77_82 crosses qubit 80 (T2 = 8.51 us) and
    v3 over-predicted success by 44 points.

  * READOUT. Even a single adjacent-pair Bell state lands ~3 points below v3's
    prediction, which gate error cannot explain. Measurement is lossy.

THE KEY DESIGN POINT
--------------------
Entry 022 tested the naive decoherence correction -- charge gate_duration / T2
per gate to both of its qubits -- and rejected it: it fixed the short bad route
(44.4 -> 10.1 pts) while destroying the long good one (5.8 -> 39.2 pts).

The reason is that in a SWAP chain THE STATE MOVES. Each qubit holds it only
briefly. Charging every gate the full T2 cost of both its qubits double-counts
across 76 gates spread over 26 healthy qubits, while being roughly right for 6
gates parked on one dead qubit.

v4 therefore integrates decoherence ALONG THE CARRIER PATH: for each qubit the
state actually occupies, (time spent there) / (that qubit's T2). A long route
over healthy qubits accumulates little; a short route parked on a dead qubit
accumulates a lot. Same formula, both regimes.
"""

import json

from emulator_v3_routing import (edge_error, is_calibrated, shortest_path,
                                 uncalibrated_on_path)

SQ_GATE_COST = 0.00224   # Entry 010, deconfounded single-qubit gate cost
GATES_PER_SWAP = 3       # a SWAP decomposes into 3 two-qubit gates


def ideal_set_floor(num_measured_qubits, ideal_set_size=2):
    """The population floor a maximally-mixed state contributes to the ideal
    outcome set once the depolarizing channel has fired at least once.

    Entry 025: for a 2-of-4 Bell circuit this is 0.5. Entry 026 confirmed no
    residual chip physics survives subtracting this baseline from measured
    decay curves -- Entry 027 retires the fitted variable_forgiveness_ratio()
    power law and uses this closed form directly instead. Generalizes to any
    ideal-outcome-set size over any number of measured qubits (default 2:
    the |0..0> / |1..1> pair every circuit in this project targets)."""
    if not num_measured_qubits:
        return 0.5
    return ideal_set_size / (2 ** num_measured_qubits)


def load_coherence(chip):
    """T1/T2/readout per qubit and per-edge gate durations (export_coherence.py)."""
    with open(f"quantumbridge_data/coherence_{chip}.json") as f:
        raw = json.load(f)
    return {
        "T1": {int(q): v["T1"] for q, v in raw["qubits"].items()},
        "T2": {int(q): v["T2"] for q, v in raw["qubits"].items()},
        "readout": {int(q): v["readout_error"] for q, v in raw["qubits"].items()},
        "duration": {tuple(int(x) for x in k.split(",")): v
                     for k, v in raw["gate_durations"].items()},
        "median_duration": raw["median_gate_duration"],
    }


def gate_duration(coh, a, b):
    return coh["duration"].get((min(a, b), max(a, b)), coh["median_duration"])


def dephasing(coh, qubit, seconds):
    """Probability the state on `qubit` has dephased after `seconds`.

    Uses 1 - exp(-t/T2). T2 rather than T1 because the states this project
    measures (Bell, GHZ) carry their information in the phase, so dephasing is
    the channel that destroys the measured outcome."""
    t2 = coh["T2"].get(qubit)
    if not t2:
        return 0.0
    import math
    return 1 - math.exp(-seconds / t2)


def route_cost(graph, coh, q1, q2):
    """Cost of one logical two-qubit gate between q1 and q2, including the
    SWAP chain needed to bring them together.

    Entry 027: gate error and decoherence are no longer merged into one
    "success probability" here. Gate error is a depolarizing channel with an
    absorbing final state (Entry 025) -- it has to be accumulated as a
    survival product across the WHOLE circuit and converted to a success
    probability exactly once, at the end, via the ideal_set_floor() closed
    form. Decoherence has no such absorbing structure, so it's still an
    ordinary multiplicative probability.

    Returns (survive_factor, deco_factor, notes) where survive_factor is the
    probability this gate's real edge error(s) did NOT fire the depolarizing
    channel, and deco_factor is the ordinary (1 - dephasing) probability."""
    notes = []

    # Adjacent: one gate, no routing.
    for neighbor, err in graph.get(q1, []):
        if neighbor == q2:
            d = gate_duration(coh, q1, q2)
            deco = 1 - dephasing(coh, q1, d)
            if not is_calibrated(q1, q2):
                notes.append(f"UNCALIBRATED edge ({q1},{q2})")
            return (1 - err), deco, notes or ["direct"]

    path = shortest_path(graph, q1, q2)
    if path is None or len(path) < 2:
        return 0.95, 1.0, ["no_path_fallback"]

    bad = uncalibrated_on_path(path)
    if bad:
        notes.append(f"{len(bad)} uncalibrated edge(s) {bad}")

    num_swaps = len(path) - 2
    survive = 1.0
    deco_factor = 1.0

    # Gate-error term: 3 real gates per SWAP, plus the final entangling gate.
    # Raw calibrated error, no forgiveness-ratio correction (Entry 027) --
    # the absorbing-state closed form applied at the circuit level already
    # accounts for the "forgiveness" Entries 017-021 were fitting empirically.
    for i in range(num_swaps):
        raw = edge_error(graph, path[i], path[i + 1])
        survive *= (1 - raw) ** GATES_PER_SWAP
    raw_final = edge_error(graph, path[-2], path[-1])
    survive *= (1 - raw_final)

    # Decoherence term, integrated along the CARRIER path only. The state
    # occupies path[j] for the duration of SWAP j, then moves on.
    worst = (0.0, None)
    for j in range(num_swaps):
        held = GATES_PER_SWAP * gate_duration(coh, path[j], path[j + 1])
        d = dephasing(coh, path[j], held)
        deco_factor *= (1 - d)
        if d > worst[0]:
            worst = (d, path[j])
    carrier = path[-2] if num_swaps else path[0]
    deco_factor *= (1 - dephasing(coh, carrier, gate_duration(coh, path[-2], path[-1])))

    if worst[1] is not None and worst[0] > 0.05:
        notes.append(f"dephasing dominated by q{worst[1]} "
                     f"(T2={coh['T2'][worst[1]] * 1e6:.1f}us, {worst[0] * 100:.1f}% loss)")
    notes.append(f"routed via {len(path) - 1} hops, {num_swaps} SWAPs")
    return survive, deco_factor, notes


def predict(circuit, graph, coh, measured_qubits=None, include_readout=True):
    """Predicted success probability for a logical circuit.

    Entry 027: gate error across all two-qubit gates in the circuit is
    accumulated as a single survival product and converted to a success
    probability exactly once via the absorbing-state closed form (Entry 025),
    rather than being "forgiveness-ratio" corrected gate-by-gate. Decoherence,
    single-qubit gate cost, and readout remain ordinary per-event probabilities."""
    survive = 1.0
    other = 1.0
    for inst in circuit.data:
        gate = inst.operation
        if gate.name in ("measure", "barrier"):
            continue
        idx = [circuit.find_bit(q).index for q in inst.qubits]
        if gate.num_qubits == 2:
            sp, deco, _ = route_cost(graph, coh, idx[0], idx[1])
            survive *= sp
            other *= deco
        elif gate.num_qubits == 1:
            other *= (1 - SQ_GATE_COST)

    floor = ideal_set_floor(len(measured_qubits) if measured_qubits else 2)
    p = (floor + (1 - floor) * survive) * other

    if include_readout and measured_qubits:
        for q in measured_qubits:
            ro = coh["readout"].get(q)
            if ro:
                p *= (1 - ro)
    return p
