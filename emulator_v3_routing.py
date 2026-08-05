"""
QuantumBridge Emulator v3 — With SWAP Routing for Non-Adjacent Qubits

Real hardware doesn't have direct connections between every qubit pair.
When a circuit needs a CNOT between two non-adjacent qubits, the real
transpiler inserts SWAP gates to route the interaction. This version
models that routing cost instead of guessing a flat number.
"""

import json
from collections import deque

def load_forgiveness_ratio():
    """Load the empirically fitted forgiveness ratio from the gate-count
    decay curve experiment (see quantumbridge_data/gate_count_decay_curve.json
    and fit_decay_curve.py). Falls back to 1.0 (no correction) if the fitted
    model file isn't present."""
    try:
        with open("quantumbridge_data/fitted_decay_model.json") as f:
            return json.load(f)["forgiveness_ratio"]
    except FileNotFoundError:
        return 1.0

FORGIVENESS_RATIO = load_forgiveness_ratio()

def load_calibration():
    with open("quantumbridge_data/offline_calibration_cairo_full.json") as f:
        return json.load(f)


def build_connectivity_graph(calibration):
    """Build an adjacency map: qubit -> list of (neighbor, cnot_error).
    Topology (which edges exist) comes from the real chip's coupling map.
    Error rates come from calibration data where available, with a
    fallback estimate for edges the calibration export didn't capture."""
    with open("quantumbridge_data/real_topology_cairo.json") as f:
        real_edges = json.load(f)

    error_lookup = {}
    for g in calibration["two_qubit_gate_errors"]:
        q1, q2 = g["qubits"]
        err = g["gate_error"] or 0.01
        error_lookup[tuple(sorted((q1, q2)))] = err

    FALLBACK_ERR = 0.01  # used when calibration didn't capture this edge

    graph = {}
    for a, b in real_edges:
        err = error_lookup.get((a, b), FALLBACK_ERR)
        graph.setdefault(a, []).append((b, err))
        graph.setdefault(b, []).append((a, err))

    return graph


def shortest_path(graph, start, end):
    """BFS shortest path between two qubits on the real chip topology."""
    if start == end:
        return [start]
    visited = {start}
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        for neighbor, _ in graph.get(node, []):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None  # no path found (shouldn't happen on a connected chip)

def edge_error(graph, a, b):
    """Look up the real per-edge error rate from the graph. Returns a
    conservative fallback if the lookup fails (shouldn't normally trigger
    for a valid path returned by shortest_path())."""
    for neighbor, err in graph.get(a, []):
        if neighbor == b:
            return err
    return 0.01  # fallback
def cnot_error_for_pair(graph, q1, q2):
    """Real direct error if adjacent, or estimated SWAP-routing cost if not.
    Error is scaled per-edge by variable_forgiveness_ratio() — an
    empirically fitted, error-magnitude-aware correction (Entry 019)
    replacing the earlier flat FORGIVENESS_RATIO constant (Entry 017),
    since high-error edges were found to be forgiven less than low-error
    edges."""
    for neighbor, err in graph.get(q1, []):
        if neighbor == q2:
            return err * variable_forgiveness_ratio(err), ["direct"]

    path = shortest_path(graph, q1, q2)
    if path is None or len(path) < 2:
        return 0.05, ["no_path_fallback"]

    num_swaps = len(path) - 2

    success_prob = 1.0
    for i in range(num_swaps):
        raw_hop_err = edge_error(graph, path[i], path[i + 1])
        hop_err = raw_hop_err * variable_forgiveness_ratio(raw_hop_err)
        success_prob *= (1 - hop_err) ** 3

    raw_final_err = edge_error(graph, path[-2], path[-1])
    final_hop_err = raw_final_err * variable_forgiveness_ratio(raw_final_err)
    success_prob *= (1 - final_hop_err)

    total_error = round(1 - success_prob, 5)
    return total_error, [f"routed via {path} ({num_swaps} SWAPs, variable-forgiveness-corrected)"]

def success_prob_from_gate_count(cx_count, graph):
    """Predict success probability from a real gate count. Each edge's
    error is scaled by its own error-magnitude-aware forgiveness ratio
    (Entry 019) before averaging, rather than applying one flat ratio to
    the chip-wide average error."""
    corrected_errors = [err * variable_forgiveness_ratio(err)
                         for neighbors in graph.values()
                         for _, err in neighbors]
    avg_corrected_err = sum(corrected_errors) / len(corrected_errors)

    success_prob = 1.0
    for _ in range(cx_count):
        success_prob *= (1 - avg_corrected_err)

    return success_prob

def load_forgiveness_law():
    """Load the fitted power law relating raw edge error to forgiveness ratio.

    Entry 021 replaced the original two-point Cairo fit. Those two points
    (Entry 017: 0.53 at 0.60%; Entry 019: 0.334 at 3.13%) defined a power law
    exactly, with zero residual, so the fit could never be tested — and the
    high-error anchor edge (19,22) was subsequently found to have T2 = 6.9 us
    against a 697 ns gate, meaning its decay was ~3 parts dephasing to 1 part
    gate error. It was not measuring forgiveness at all.

    The current law is fitted on four FakeKyiv edges that pass the Entry 021
    decoherence screen D = gate_error / (gate_duration / T2_min) >= 1.5,
    giving R^2 = 0.9648 across a 4.4x error span.

    Kept in JSON rather than hardcoded so per-chip calibration can replace it
    without touching this module."""
    try:
        with open("quantumbridge_data/forgiveness_law.json") as f:
            law = json.load(f)
        return law["coefficient"], law["exponent"], law.get("error_range_fitted")
    except (FileNotFoundError, KeyError):
        # Kyiv Entry 021 fit, inlined as fallback
        return 0.0898, -0.3873, [0.00986, 0.043229]


FORGIVENESS_COEFFICIENT, FORGIVENESS_EXPONENT, FORGIVENESS_FIT_RANGE = load_forgiveness_law()


def variable_forgiveness_ratio(raw_edge_error):
    """Forgiveness ratio for an edge, from its raw calibrated gate error.

    High-error edges are forgiven less than low-error edges. See Entry 021 for
    the fit and its validity conditions.

    CAVEAT (Entry 020): absolute forgiveness constants are chip-specific —
    Sherbrooke measured ~26% above Cairo at both error magnitudes. This law is
    fitted on Kyiv and is an interim default, not a universal constant. Applying
    it to another chip carries a systematic offset of that order.

    CAVEAT (Entry 021): the fit covers 0.99%-4.32% raw edge error. Below that
    range it extrapolates, and the two Kyiv edges below 1% were exactly the ones
    where decoherence rather than gate error drove the decay. Predictions on
    very-low-error edges should be treated as least trustworthy."""
    if raw_edge_error <= 0:
        return 1.0
    ratio = FORGIVENESS_COEFFICIENT * (raw_edge_error ** FORGIVENESS_EXPONENT)
    return max(0.1, min(1.0, ratio))  # clamp to a sane range


