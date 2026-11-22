"""
Entry 111: grow a Heron/CZ simulated bell-pair dataset (Aer MPS noise from the fake-backend snapshot).
Routes crossing dead couplers (err>=0.5) or dead qubits are EXCLUDED (Entry 109/110 regime; live ibm_fez has 1 dead coupler).
usage: GROW_BUDGET=120 python3 entry111_grow_heron.py CHIP N     (resumable; appends to quantumbridge_data/entry111_heron_<chip>.json)
"""
import json, os, random, sys, time
from multiprocessing.pool import ThreadPool
from qiskit import QuantumCircuit, transpile
import emulator_v3_routing as em
import entry089_grow_multibell as g
from exact_dwell_routing import exact_dwell_cost, route_with_explicit_swaps

chip, N = sys.argv[1], int(sys.argv[2]); OUT = f"quantumbridge_data/entry111_heron_{chip}.json"
backend, graph, coh, sim = g.get_ctx(chip); nq = backend.num_qubits
cal = json.load(open(f"quantumbridge_data/offline_calibration_{chip}_full.json"))["two_qubit_gate_errors"]
dead = {tuple(sorted(x["qubits"])) for x in cal if (x["gate_error"] or 0) >= 0.5}
recs = json.load(open(OUT)) if os.path.exists(OUT) else []
seen = {(r["a"], r["b"]) for r in recs}; rng = random.Random(111 + len(recs) + hash(chip) % 1000)


def clean(a, b):
    qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b); t = route_with_explicit_swaps(qc, backend)
    return not any(tuple(sorted(t.find_bit(q).index for q in i.qubits)) in dead for i in t.data if i.operation.num_qubits == 2)


tasks = []
while len(tasks) < N:
    a, b = rng.sample(range(nq), 2); p = em.shortest_path(graph, a, b)
    if p is None or len(p) - 1 > 6 or (a, b) in seen or not clean(a, b): continue
    seen.add((a, b)); tasks.append((a, b, len(p) - 1))


def run(t):
    a, b, h = t
    qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b); qc.measure(a, 0); qc.measure(b, 1)
    tc = transpile(qc, backend=backend, initial_layout=list(range(nq)), optimization_level=3, seed_transpiler=1)
    cn = sim.run(tc, shots=4096, seed_simulator=1).result().get_counts(); tot = sum(cn.values())
    qb = QuantumCircuit(nq, 2); qb.h(a); qb.cx(a, b)
    v = exact_dwell_cost(qb, backend, graph, coh, [(a, b)], [a, b])[0]
    return {"chip": chip, "kind": "bell", "a": a, "b": b, "logical_pairs": [[a, b]], "pairs_flat": [a, b],
            "bfs_hop_distance": h, "aer_ground_truth": (cn.get("00", 0) + cn.get("11", 0)) / tot, "v4_1_prediction": v}


t0 = time.time(); budget = float(os.environ.get("GROW_BUDGET", "110")); done = 0
for t in tasks:                       # sequential: threads/pools hung the sandbox (Entry 111 notes)
    if time.time() - t0 > budget: break
    recs.append(run(t)); done += 1
tmp = OUT + ".tmp"; json.dump(recs, open(tmp, "w")); os.replace(tmp, OUT)
print(f"{chip}: +{done} in {time.time()-t0:.0f}s -> total {len(recs)}", flush=True)
os._exit(0)   # kill any lingering sim threads
