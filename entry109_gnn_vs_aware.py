"""Entry 109c: how much of the GNN's gain over v4.1 is just dead-coupler routes?
Uses the 5-chip lookup (entry097: v41, gnn, Aer truth where simulated). NOTE: GNN is trained on these circuits (in-sample).
usage: python3 entry109_gnn_vs_aware.py CHIP"""
import json, sys, time, numpy as np
from qiskit import QuantumCircuit
import emulator_v3_routing as em
import emulator_v4 as v4
import entry089_grow_multibell as g
from exact_dwell_routing import exact_dwell_cost, route_with_explicit_swaps
chip = sys.argv[1]; t0 = time.time()
rows = [r for r in json.load(open("quantumbridge_data/entry097_lookup_5chip.json"))[chip] if r[8] is not None]
backend = g.BACKENDS[chip](); nq = backend.num_qubits; coh = v4.load_coherence(chip)
cal = json.load(open(f"quantumbridge_data/offline_calibration_{chip}_full.json"))["two_qubit_gate_errors"]
dead = {tuple(sorted(x["qubits"])) for x in cal if (x["gate_error"] or 0) >= 0.5}
em.DEAD_EDGE_AWARE = True; graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
out = []
for r in rows:
    if time.time() - t0 > 130: break
    a, b = r[0], r[1]; qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b)
    t = route_with_explicit_swaps(qc, backend)
    hit = any(tuple(sorted(t.find_bit(q).index for q in i.qubits)) in dead for i in t.data if i.operation.num_qubits == 2)
    aware = exact_dwell_cost(qc, backend, graph, coh, [(a, b)], [a, b])[0]
    out.append((r[8], r[5], r[6], aware, hit))
y, v, gn, aw, h = (np.array([x[i] for x in out], dtype=float) for i in range(5)); h = h.astype(bool)
print(f"{chip}: n={len(y)} (Aer-labelled pairs) | crossing a dead edge: {int(h.sum())} ({h.mean()*100:.0f}%) | {time.time()-t0:.0f}s")
for nm, m in (("ALL", np.ones_like(h)), ("crossing", h), ("clean", ~h)):
    if m.sum():
        e = lambda p: np.abs(p[m] - y[m]).mean() * 100
        print(f"  {nm:9s} n={int(m.sum()):4d} MAE  v4.1 {e(v):5.2f} | v4.1+dead-aware {e(aw):5.2f} | GNN (in-sample) {e(gn):5.2f}")
