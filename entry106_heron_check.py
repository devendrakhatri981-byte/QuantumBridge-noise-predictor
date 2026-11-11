"""Entry 106: does v4.1 (closed-form physics) transfer to Heron/CZ chips? Single bell pairs, hops 1-6,
Aer MPS ground truth vs exact_dwell_cost. usage: python3 entry106_heron_check.py CHIP N"""
import json, random, sys, time
import numpy as np
import entry089_grow_multibell as g
from qiskit import QuantumCircuit, transpile
import emulator_v3_routing as em
from exact_dwell_routing import exact_dwell_cost

chip, N = sys.argv[1], int(sys.argv[2]); t0 = time.time()
backend, graph, coh, sim = g.get_ctx(chip); nq = backend.num_qubits
rng = random.Random(106); rows = []
while len(rows) < N and time.time() - t0 < 140:
    a, b = rng.sample(range(nq), 2); p = em.shortest_path(graph, a, b)
    if p is None or len(p) - 1 > 6: continue
    qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b); qc.measure(a, 0); qc.measure(b, 1)
    t = transpile(qc, backend=backend, initial_layout=list(range(nq)), optimization_level=3, seed_transpiler=1)
    cn = sim.run(t, shots=8192, seed_simulator=1).result().get_counts(); tot = sum(cn.values())
    truth = (cn.get("00", 0) + cn.get("11", 0)) / tot
    qb = QuantumCircuit(nq, 2); qb.h(a); qb.cx(a, b)
    pred, _, _ = exact_dwell_cost(qb, backend, graph, coh, [(a, b)], [a, b])
    rows.append({"a": a, "b": b, "hops": len(p) - 1, "truth": truth, "v41": pred})
y = np.array([r["truth"] for r in rows]); v = np.array([r["v41"] for r in rows])
print(f"{chip}: n={len(rows)} truth mean {y.mean():.3f} | v4.1 MAE {np.abs(v-y).mean()*100:.2f} pts, bias {(v-y).mean()*100:+.2f}, corr {np.corrcoef(v,y)[0,1]:.3f} | {time.time()-t0:.0f}s")
for h in sorted({r['hops'] for r in rows}):
    s = [r for r in rows if r['hops'] == h]
    print(f"  hop {h}: n={len(s)} truth {np.mean([r['truth'] for r in s]):.3f} v4.1 {np.mean([r['v41'] for r in s]):.3f}")
json.dump(rows, open(f"quantumbridge_data/entry106_{chip}_v41_check.json", "w"))
