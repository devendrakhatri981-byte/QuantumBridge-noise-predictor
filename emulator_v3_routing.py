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
    Error is scaled by FORGIVENESS_RATIO — an empirically fitted correction
    (Entry 016/017) accounting for the fact that not every gate error
    actually causes circuit-level failure, since success is measured
    against a multi-outcome ideal set rather than a single exact bitstring."""
    for neighbor, err in graph.get(q1, []):
        if neighbor == q2:
            return err * FORGIVENESS_RATIO, ["direct"]

    path = shortest_path(graph, q1, q2)
    if path is None or len(path) < 2:
        return 0.05, ["no_path_fallback"]

    num_swaps = len(path) - 2

    success_prob = 1.0
    for i in range(num_swaps):
        hop_err = edge_error(graph, path[i], path[i + 1]) * FORGIVENESS_RATIO
        success_prob *= (1 - hop_err) ** 3

    final_hop_err = edge_error(graph, path[-2], path[-1]) * FORGIVENESS_RATIO
    success_prob *= (1 - final_hop_err)

    total_error = round(1 - success_prob, 5)
    return total_error, [f"routed via {path} ({num_swaps} SWAPs, forgiveness-corrected)"]

def success_prob_from_gate_count(cx_count, graph):
    """Predict success probability from a real gate count, using the
    average per-edge error scaled by the empirically fitted forgiveness
    ratio (Entry 016/017)."""
    all_errors = [err for neighbors in graph.values() for _, err in neighbors]
    avg_edge_err = sum(all_errors) / len(all_errors) * FORGIVENESS_RATIO

    success_prob = 1.0
    for _ in range(cx_count):
        success_prob *= (1 - avg_edge_err)

    return success_prob