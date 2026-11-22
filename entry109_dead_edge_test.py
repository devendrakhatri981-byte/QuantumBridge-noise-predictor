"""Entry 109: does making v4.1 dead-coupler-aware fix the Fez failure? Same 30 Aer-labelled bell pairs as Entry 106.
usage: python3 entry109_dead_edge_test.py"""
import json, numpy as np
from qiskit import QuantumCircuit
import emulator_v3_routing as em
import entry089_grow_multibell as g
from exact_dwell_routing import exact_dwell_cost, route_with_explicit_swaps

chip = "fez"; rows = json.load(open("quantumbridge_data/entry106_fez_v41_check.json"))
backend, _, coh, _ = g.get_ctx(chip); nq = backend.num_qubits
cal = json.load(open(f"quantumbridge_data/offline_calibration_{chip}_full.json"))["two_qubit_gate_errors"]
dead = {tuple(sorted(x["qubits"])) for x in cal if (x["gate_error"] or 0) >= 0.5}
res = {}
for aware in (False, True):
    em.DEAD_EDGE_AWARE = aware; graph = em.build_connectivity_graph(em.load_calibration(chip), chip); out = []
    for r in rows:
        a, b = r["a"], r["b"]; qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b)
        t = route_with_explicit_swaps(qc, backend)
        hit = any(tuple(sorted(t.find_bit(q).index for q in i.qubits)) in dead for i in t.data if i.operation.num_qubits == 2)
        out.append((exact_dwell_cost(qc, backend, graph, coh, [(a, b)], [a, b])[0], hit, r["truth"]))
    res[aware] = out
for name, sel in (("ALL", lambda h: True), ("crosses dead edge", lambda h: h), ("clean routes", lambda h: not h)):
    for aware in (False, True):
        s = [(p, y) for p, h, y in res[aware] if sel(h)]; p = np.array([x[0] for x in s]); y = np.array([x[1] for x in s])
        print(f"{name:18s} aware={aware!s:5s} n={len(s):2d} MAE {np.abs(p-y).mean()*100:6.2f} bias {(p-y).mean()*100:+6.2f} corr {np.corrcoef(p,y)[0,1]:.3f}")
