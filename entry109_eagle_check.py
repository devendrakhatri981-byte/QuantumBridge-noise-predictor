"""Entry 109b: effect of dead-edge awareness on an Eagle chip with existing Aer labels. usage: python3 entry109_eagle_check.py CHIP FILE START N"""
import json, sys, time, numpy as np
from qiskit import QuantumCircuit
import emulator_v3_routing as em
import emulator_v4 as v4
import entry089_grow_multibell as g
from exact_dwell_routing import exact_dwell_cost, route_with_explicit_swaps
chip, fn, start, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); t0 = time.time()
recs = [r for r in json.load(open(fn)) if r["kind"] == "bell"][start:start + N]
backend = g.BACKENDS[chip](); nq = backend.num_qubits; coh = v4.load_coherence(chip)
cal = json.load(open(f"quantumbridge_data/offline_calibration_{chip}_full.json"))["two_qubit_gate_errors"]
dead = {tuple(sorted(x["qubits"])) for x in cal if (x["gate_error"] or 0) >= 0.5}
graphs = {}
for aware in (False, True):
    em.DEAD_EDGE_AWARE = aware; graphs[aware] = em.build_connectivity_graph(em.load_calibration(chip), chip)
rows = []
for r in recs:
    if time.time() - t0 > 140: break
    a, b = r["logical_pairs"][0]; qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b)
    t = route_with_explicit_swaps(qc, backend)
    hit = any(tuple(sorted(t.find_bit(q).index for q in i.qubits)) in dead for i in t.data if i.operation.num_qubits == 2)
    p0 = exact_dwell_cost(qc, backend, graphs[False], coh, [(a, b)], [a, b])[0]
    p1 = exact_dwell_cost(qc, backend, graphs[True], coh, [(a, b)], [a, b])[0]
    rows.append((r["aer_ground_truth"], p0, p1, hit))
y, p0, p1, h = (np.array([x[i] for x in rows]) for i in range(4))
print(f"{chip}: n={len(y)} dead edges={len(dead)} | circuits crossing a dead edge: {int(h.sum())} ({h.mean()*100:.1f}%) | {time.time()-t0:.0f}s")
for nm, m in (("ALL", np.ones_like(h, bool)), ("crossing", h), ("clean", ~h)):
    if m.sum():
        print(f"  {nm:9s} n={int(m.sum()):3d} MAE old {np.abs(p0[m]-y[m]).mean()*100:5.2f} -> aware {np.abs(p1[m]-y[m]).mean()*100:5.2f}")
