"""
QuantumBridge Emulator v3 — With SWAP Routing for Non-Adjacent Qubits

Real hardware doesn't have direct connections between every qubit pair.
When a circuit needs a CNOT between two non-adjacent qubits, the real
transpiler inserts SWAP gates to route the interaction. This version
models that routing cost instead of guessing a flat number.
"""

import json
from collections import deque


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


def cnot_error_for_pair(graph, q1, q2):
    """Real direct error if adjacent, or estimated SWAP-routing cost if not."""
    # Check direct connection first
    for neighbor, err in graph.get(q1, []):
        if neighbor == q2:
            return err, ["direct"]

    # Not adjacent — find shortest path and estimate SWAP cost
    path = shortest_path(graph, q1, q2)
    if path is None or len(path) < 2:
        return 0.05, ["no_path_fallback"]

    num_swaps = len(path) - 2  # hops needed to bring q1 adjacent to q2
    swap_error_equivalent = 0.01  # approx error of 1 SWAP (~3 CNOTs)

    # Combine: each swap degrades fidelity, plus the final real CNOT
    success_prob = 1.0
    for _ in range(num_swaps):
        success_prob *= (1 - swap_error_equivalent) ** 3  # 1 SWAP = 3 CNOTs

    final_hop_err = None
    for neighbor, err in graph.get(path[-2], []):
        if neighbor == path[-1]:
            final_hop_err = err
    if final_hop_err is None:
        final_hop_err = 0.01
    success_prob *= (1 - final_hop_err)

    total_error = round(1 - success_prob, 5)
    return total_error, [f"routed via {path} ({num_swaps} SWAPs)"]


if __name__ == "__main__":
    calibration = load_calibration()
    graph = build_connectivity_graph(calibration)

    print(f"Chip: {calibration['chip']} — real connectivity graph built")
    total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"Total direct connections: {total_edges}\n")
    print("="*65)

    test_pairs = [(24, 25), (22, 19), (0, 1), (0, 26), (5, 22)]

    for q1, q2 in test_pairs:
        err, notes = cnot_error_for_pair(graph, q1, q2)
        print(f"\nQubits ({q1}, {q2}):")
        print(f"  Estimated CNOT-equivalent error: {round(err*100, 2)}%")
        print(f"  {notes[0]}")
