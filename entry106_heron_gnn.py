"""Entry 106b: zero-shot GNN (entry085, Eagle-trained) on a Heron/CZ chip with OWN-stats normalization
(feature stats from unlabeled random pairs on the new chip, no labels). Compares to Aer truth from the
v4.1 check file. usage: python3 entry106_heron_gnn.py CHIP NSTATS"""
import json, random, sys, time
import numpy as np
sys.path.insert(0, ".")
import api.main as M
import entry089_grow_multibell as g89
import emulator_v3_routing as em
from qiskit import QuantumCircuit
from exact_dwell_routing import exact_dwell_cost

chip, NS = sys.argv[1], int(sys.argv[2]); t0 = time.time()
M.BACKENDS.update({k: v for k, v in g89.BACKENDS.items() if k not in M.BACKENDS})
backend, graph, coh = M.get_backend_ctx(chip); nq = backend.num_qubits


def graph_of(a, b):
    p = em.shortest_path(graph, a, b); d = len(p) - 1
    qb = QuantumCircuit(nq, 2); qb.h(a); qb.cx(a, b)
    v41, _, _ = exact_dwell_cost(qb, backend, graph, coh, [(a, b)], [a, b])
    rec = {"chip": chip, "kind": "bell", "pairs_flat": [a, b], "logical_pairs": [[a, b]], "bfs_hop_distance": d}
    gr = M.graph_for_record(backend, graph, coh, rec)
    gr.update({"chip": chip, "kind": "bell", "a": a, "b": b, "bfs_hop_distance": d, "v4_1_prediction": v41})
    return gr


rows = json.load(open(f"quantumbridge_data/entry106_{chip}_v41_check.json"))
need = {(r["a"], r["b"]) for r in rows}; rng = random.Random(5); graphs = {}
while len(graphs) < NS + len(need) and time.time() - t0 < 150:
    a, b = rng.sample(range(nq), 2) if len(graphs) >= len(need) else sorted(need)[len(graphs)]
    if (a, b) in graphs: continue
    p = em.shortest_path(graph, a, b)
    if p is None or len(p) - 1 > 8: continue
    graphs[(a, b)] = graph_of(a, b)
nf = np.concatenate([np.array(x["nodes"]) for x in graphs.values()]); ef = np.concatenate([np.array(x["edges"])[:, 2:] for x in graphs.values()])
M.NODE_STATS[chip] = (nf.mean(0), nf.std(0) + 1e-6); M.EDGE_STATS[chip] = (ef.mean(0), ef.std(0) + 1e-6)
out = []
for r in rows:
    gm, gs = M.predict_mc(graphs[(r["a"], r["b"])], chip, 1); out.append(gm)
y = np.array([r["truth"] for r in rows]); v = np.array([r["v41"] for r in rows]); gp = np.array(out)
print(f"{chip}: n={len(y)} stats from {len(graphs)} unlabeled pairs | {time.time()-t0:.0f}s")
for nm, p in (("v4.1", v), ("GNN own-stats", gp)):
    print(f"  {nm:14s} MAE {np.abs(p-y).mean()*100:5.2f} pts | bias {(p-y).mean()*100:+5.2f} | corr {np.corrcoef(p,y)[0,1]:.3f}")
json.dump({"truth": y.tolist(), "v41": v.tolist(), "gnn": gp.tolist(), "hops": [r["hops"] for r in rows]},
          open(f"quantumbridge_data/entry106_{chip}_gnn_zero_shot.json", "w"))
