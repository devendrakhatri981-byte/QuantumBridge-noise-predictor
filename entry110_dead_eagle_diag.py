"""Entry 110: where does a dead coupler sit in the route, and what does Aer do? (Eagle chip, Aer-labelled pairs)
Classes: dead edge used by (T) a swap on the TARGET's path, (C) a swap on the CONTROL's path only, (F) the final entangling gate.
usage: python3 entry110_dead_eagle_diag.py CHIP"""
import json, sys, time, collections, numpy as np
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
by = collections.defaultdict(list); allrows = []
for r in rows:
    if time.time() - t0 > 120: break
    a, b = r[0], r[1]; qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b)
    t = route_with_explicit_swaps(qc, backend); loc = {a: a, b: b}; cls = set()
    for i in t.data:
        if i.operation.num_qubits != 2: continue
        x, y = [t.find_bit(q).index for q in i.qubits]; e = tuple(sorted((x, y)))
        if i.operation.name == "swap":
            if e in dead: cls.add("T" if b in [k for k, v in loc.items() if v in (x, y)] else "C")
            for lg, ph in list(loc.items()): loc[lg] = y if ph == x else x if ph == y else ph
        elif e in dead: cls.add("F")
    key = "".join(sorted(cls)) or "clean"
    aware = exact_dwell_cost(qc, backend, graph, coh, [(a, b)], [a, b])[0]
    by[key].append((r[8], r[5], aware, r[6])); allrows.append((key, r[8], r[5], r[6]))
print(f"{chip} | {time.time()-t0:.0f}s | class: n, truth mean, v4.1(old) MAE, v4.1(aware) MAE, GNN MAE")
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    a = np.array(v)
    print(f"  {k:6s} n={len(a):4d} truth {a[:,0].mean():.3f} | old {np.abs(a[:,1]-a[:,0]).mean()*100:5.2f} | aware {np.abs(a[:,2]-a[:,0]).mean()*100:5.2f} (mean pred {a[:,2].mean():.3f}) | GNN {np.abs(a[:,3]-a[:,0]).mean()*100:5.2f}")

json.dump(allrows, open(f"quantumbridge_data/entry110_{chip}_deadclass.json", "w"))
