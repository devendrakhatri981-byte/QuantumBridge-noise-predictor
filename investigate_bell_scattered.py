"""
QuantumBridge — Investigating the bell_scattered_0_26 Outlier (Entry 019)

Two hypotheses to test:
(a) Its real route crosses unusually high-error edges that the chip-wide
    average (used in success_prob_from_gate_count) doesn't capture.
(b) Sabre's routing is inconsistent run-to-run for this specific pair,
    at optimization_level=1.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
from qiskit.transpiler import CouplingMap
import json

from emulator_v3_routing import load_calibration, build_connectivity_graph, shortest_path, edge_error

backend = FakeCairoV2()
calibration = load_calibration()
graph = build_connectivity_graph(calibration)

with open("quantumbridge_data/real_topology_cairo.json") as f:
    _real_edges = json.load(f)
FIXED_COUPLING_MAP = CouplingMap(couplinglist=_real_edges + [[b, a] for a, b in _real_edges])

Q1, Q2 = 0, 26

print("=" * 60)
print("HYPOTHESIS (a): Per-edge error breakdown along the real route")
print("=" * 60)

path = shortest_path(graph, Q1, Q2)
print(f"BFS path: {path}")
print(f"Hops: {len(path) - 1}\n")

all_errors = [err for neighbors in graph.values() for _, err in neighbors]
avg_edge_err = sum(all_errors) / len(all_errors)
print(f"Chip-wide average edge error (used in gate-count model): {avg_edge_err*100:.4f}%\n")

print(f"{'Hop':<12} {'Edge error':>12} {'vs avg':>10}")
print("-" * 36)
path_errors = []
for i in range(len(path) - 1):
    err = edge_error(graph, path[i], path[i + 1])
    path_errors.append(err)
    diff_vs_avg = err / avg_edge_err
    flag = "  <-- HIGH" if diff_vs_avg > 1.5 else ""
    print(f"({path[i]},{path[i+1]:<3}) {err*100:>11.4f}% {diff_vs_avg:>9.2f}x{flag}")

print(f"\nMax edge error on path: {max(path_errors)*100:.4f}%")
print(f"Path average edge error: {sum(path_errors)/len(path_errors)*100:.4f}%")
print(f"Ratio (path avg / chip avg): {(sum(path_errors)/len(path_errors))/avg_edge_err:.2f}x")

print("\n" + "=" * 60)
print("HYPOTHESIS (b): Sabre routing consistency across repeated runs")
print("=" * 60)

qc = QuantumCircuit(27, 2)
qc.h(Q1)
qc.cx(Q1, Q2)
qc.measure([Q1, Q2], [0, 1])

cx_counts = []
for run in range(10):
    transpiled = transpile(
        qc,
        coupling_map=FIXED_COUPLING_MAP,
        basis_gates=backend.operation_names,
        initial_layout=list(range(qc.num_qubits)),
        optimization_level=1,
        seed_transpiler=None,  # explicitly unseeded, to catch real run-to-run variance
    )
    cx_count = transpiled.count_ops().get("cx", 0)
    cx_counts.append(cx_count)
    print(f"Run {run+1}: {cx_count} CX gates")

print(f"\nMin: {min(cx_counts)}, Max: {max(cx_counts)}, Range: {max(cx_counts)-min(cx_counts)}")
if max(cx_counts) - min(cx_counts) > 2:
    print(">>> Routing IS inconsistent across runs — this may be the real cause.")
else:
    print(">>> Routing is stable across runs — hypothesis (b) unlikely to be the cause.")
