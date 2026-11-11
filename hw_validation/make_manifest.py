"""
Hardware-validation kit, step 1: build a PRE-REGISTERED manifest.
Samples ~40 circuits (20 single bell pairs at 1-5 hops, 21 multi-bell with 2-4 pairs),
computes v4.1 + GNN predictions NOW (before any hardware run), writes manifest.json.
usage (from repo root): python3 hw_validation/make_manifest.py CHIP [SEED]
"""
import json, random, sys, time
sys.path.insert(0, ".")
import api.main as M
import emulator_v3_routing as em

chip = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rng = random.Random(seed)
backend, graph, coh = M.get_backend_ctx(chip); nq = backend.num_qubits


def hops(a, b):
    p = em.shortest_path(graph, a, b)
    return None if p is None else len(p) - 1


circuits = []
for h in range(1, 6):                       # singles: 4 per hop distance
    got = 0
    while got < 4:
        a, b = rng.sample(range(nq), 2)
        if hops(a, b) == h:
            r = M.live_predict(chip, a, b)
            circuits.append({"id": f"single_{len(circuits):02d}", "kind": "bell", "pairs": [[a, b]],
                             "hops": [h], "v41": r["v4_1_prediction"], "gnn": r["gnn_prediction"],
                             "gnn_unc": r["gnn_uncertainty"]})
            got += 1
for n in (2, 3, 4):                         # multi: 7 each, every pair <=3 hops
    got = 0
    while got < 7:
        qs = rng.sample(range(nq), 2 * n); prs = [[qs[2 * i], qs[2 * i + 1]] for i in range(n)]
        hs = [hops(a, b) for a, b in prs]
        if None in hs or max(hs) > 3:
            continue
        try:
            r = M.predict_multi(M.MultiRequest(chip=chip, pairs=prs))
        except Exception:
            continue
        circuits.append({"id": f"multi{n}_{len(circuits):02d}", "kind": "multibell", "pairs": prs,
                         "hops": hs, "v41": r.independence_baseline if hasattr(r, "independence_baseline")
                         else r["independence_baseline"], "gnn": r["gnn_prediction"],
                         "gnn_unc": r["gnn_uncertainty"]})
        got += 1
json.dump({"chip_model": chip, "num_qubits": nq, "seed": seed, "created": time.strftime("%Y-%m-%d"),
           "shots": 8192, "note": "predictions frozen before hardware run (pre-registered)",
           "circuits": circuits}, open(f"hw_validation/manifest_{chip}.json", "w"), indent=1)
print(len(circuits), "circuits ->", f"hw_validation/manifest_{chip}.json")
