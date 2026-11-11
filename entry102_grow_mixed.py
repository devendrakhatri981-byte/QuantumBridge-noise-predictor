"""
QuantumBridge — Entry 102: new circuit template, MIXED parallel components.
Each circuit runs 2-3 independent components on DISJOINT qubits at once; a
component is either a bell pair (H, CX a->b) or a 3-qubit GHZ (H, CX a->b,
CX a->c). Success = EVERY component measured fully correlated in the same
shot. This is the first template mixing component TYPES (entry089 was
bell-only), a step toward arbitrary circuits: the model must now handle
components with different numbers of entangling gates per qubit group.

Baseline (product_of_marginals_v41): product over components of the v4.1
closed-form prediction for that component run alone (bell: one pair; GHZ:
pairs [(a,b),(a,c)], the existing exact_dwell_cost GHZ path).
Reuses entry089's Aer/MPS context. Usage: python3 entry102_grow_mixed.py N CHIP
"""
import json, multiprocessing as mp, os, random, sys, time

import entry089_grow_multibell as g
from qiskit import QuantumCircuit, transpile
import emulator_v3_routing as em
from exact_dwell_routing import exact_dwell_cost

OUT_PATH = "quantumbridge_data/entry102_mixed.json"
LEG_MAX = 4      # max hop distance for any entangling leg (cost control)


def comp_pairs(comp):
    return [(comp[0], q) for q in comp[1:]]


def run_mixed(task):
    chip, comps, hops = task
    backend, graph, coh, sim = g.get_ctx(chip)
    nq = backend.num_qubits
    ncl = sum(len(c) for c in comps)
    qc = QuantumCircuit(nq, ncl)
    for c in comps:
        qc.h(c[0])
        for q in c[1:]:
            qc.cx(c[0], q)
    k = 0; offsets = []
    for c in comps:
        offsets.append(k)
        for q in c:
            qc.measure(q, k); k += 1
    t = transpile(qc, backend=backend, initial_layout=list(range(nq)),
                  optimization_level=3, seed_transpiler=1)
    counts = sim.run(t, shots=8192, seed_simulator=g.SEEDS[0]).result().get_counts()
    tot = sum(counts.values())

    def comp_ok(bstr, i):
        clean = bstr.replace(" ", "")
        bits = [clean[-(offsets[i] + j + 1)] for j in range(len(comps[i]))]
        return len(set(bits)) == 1

    joint = sum(c for b, c in counts.items() if all(comp_ok(b, i) for i in range(len(comps))))
    marg = [sum(c for b, c in counts.items() if comp_ok(b, i)) / tot for i in range(len(comps))]

    prod = 1.0; per = []
    for c in comps:
        qs = QuantumCircuit(nq, len(c))
        qs.h(c[0])
        for q in c[1:]:
            qs.cx(c[0], q)
        p, _, _ = exact_dwell_cost(qs, backend, graph, coh, comp_pairs(c), list(c))
        per.append(p); prod *= p
    return {"chip": chip, "kind": "mixed", "components": [list(c) for c in comps],
            "comp_kinds": ["bell" if len(c) == 2 else "ghz3" for c in comps],
            "n_comps": len(comps), "bfs_hop_distances": hops,
            "aer_ground_truth": joint / tot, "aer_marginal_per_comp": marg,
            "product_of_marginals_v41": prod, "per_comp_v41": per}


def sample_task(chip, nq, graph, rng):
    n = rng.choice((2, 3))
    kinds = [rng.choice(("bell", "ghz3")) for _ in range(n)]
    if "ghz3" not in kinds:
        kinds[rng.randrange(n)] = "ghz3"
    comps, hops, used = [], [], set()
    for kd in kinds:
        size = 2 if kd == "bell" else 3
        ok = False
        for _ in range(200):
            qs = rng.sample(range(nq), size)
            if any(q in used for q in qs):
                continue
            ds = []
            for q in qs[1:]:
                p = em.shortest_path(graph, qs[0], q)
                if p is None:
                    ds = None; break
                ds.append(len(p) - 1)
            if ds is None or max(ds) > LEG_MAX:
                continue
            comps.append(tuple(qs)); hops.append(ds); used.update(qs); ok = True
            break
        if not ok:
            return None
    return (chip, comps, hops)


def main(n_circuits, chip, seed=102, n_workers=4):
    records = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
    mine = [r for r in records if r["chip"] == chip]
    rng = random.Random(seed + hash(chip) % 10000 + len(mine))
    sigs = {(r["chip"], tuple(sorted(tuple(c) for c in r["components"]))) for r in records}
    backend = g.BACKENDS[chip]()
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    tasks, tries = [], 0
    while len(tasks) < n_circuits and tries < n_circuits * 100:
        tries += 1
        t = sample_task(chip, backend.num_qubits, graph, rng)
        if t is None:
            continue
        sig = (chip, tuple(sorted(tuple(c) for c in t[1])))
        if sig in sigs:
            continue
        sigs.add(sig); tasks.append(t)
    print(f"resuming with {len(mine)} for {chip}; queued {len(tasks)}")
    t0 = time.time(); done = 0
    budget = float(os.environ.get("GROW_BUDGET", "110"))
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(n_workers)
    it = pool.imap_unordered(run_mixed, tasks, chunksize=1)
    try:
        while done < len(tasks):
            left = budget - (time.time() - t0)
            if left <= 0:
                break
            records.append(it.next(timeout=left)); done += 1
            if done % 5 == 0:
                g._atomic_save(records, OUT_PATH)
    except mp.TimeoutError:
        pass
    finally:
        pool.terminate()
    g._atomic_save(records, OUT_PATH)
    print(f"added {done} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40, sys.argv[2] if len(sys.argv) > 2 else "kyiv")
