"""
Hardware manifest 2: do REAL dead couplers behave like Aer's ~0.5 floor? (Entry 110 claim)
Routes on live ibm_fez that cross its dead coupler(s), classified T (swap on target path), C (swap on control path only),
F (final entangling gate on the dead edge). Frozen predictions: v4.1 old, v4.1 dead-aware, pre-registered dead-route rule.
usage: python3 hw_validation/make_manifest_dead.py fez_live
"""
import json, random, sys, time
import numpy as np
sys.path.insert(0, ".")
import entry089_grow_multibell as g89
import api.main as M
import emulator_v3_routing as em
from qiskit import QuantumCircuit
from exact_dwell_routing import exact_dwell_cost, route_with_explicit_swaps

chip = sys.argv[1]
M.BACKENDS.update({k: v for k, v in g89.BACKENDS.items() if k not in M.BACKENDS})
backend, _, coh = M.get_backend_ctx(chip); nq = backend.num_qubits
cal = json.load(open(f"quantumbridge_data/offline_calibration_{chip}_full.json"))["two_qubit_gate_errors"]
dead = {tuple(sorted(x["qubits"])) for x in cal if (x["gate_error"] or 0) >= 0.5}
dead_q = set(json.load(open(f"quantumbridge_data/coherence_{chip}.json")).get("missing_qubits_filled_with_median", []))
graphs = {}
for aware in (False, True):
    em.DEAD_EDGE_AWARE = aware; graphs[aware] = em.build_connectivity_graph(em.load_calibration(chip), chip)
em.DEAD_EDGE_AWARE = False
graph = graphs[False]


def classify(a, b):
    qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b); t = route_with_explicit_swaps(qc, backend); loc = {a: a, b: b}; cls = set(); used = set()
    for i in t.data:
        if i.operation.num_qubits != 2: continue
        x, y = [t.find_bit(q).index for q in i.qubits]; used |= {x, y}; e = tuple(sorted((x, y)))
        if i.operation.name == "swap":
            if e in dead: cls.add("T" if loc[b] in (x, y) else "C")
            for lg, ph in list(loc.items()): loc[lg] = y if ph == x else x if ph == y else ph
        elif e in dead: cls.add("F")
    return "".join(sorted(cls)), used


rng = random.Random(77); want = {"F": 4, "T": 4, "C": 4}; got = {k: [] for k in want}
cands = [(a, b) for e in dead for (a, b) in (e, e[::-1])]          # adjacent pairs across the dead edge itself (class F)
tries = 0
while any(len(got[k]) < want[k] for k in want) and tries < 200000:
    tries += 1
    a, b = cands.pop() if cands else rng.sample(range(nq), 2)
    p = em.shortest_path(graph, a, b)
    if p is None or len(p) - 1 > 6 or a in dead_q or b in dead_q: continue
    cls, used = classify(a, b)
    if cls in want and len(got[cls]) < want[cls] and not (used & dead_q) and (a, b) not in got[cls]:
        got[cls].append((a, b))

circuits = []
for cls, prs in got.items():
    for a, b in prs:
        qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b)
        p_old = exact_dwell_cost(qc, backend, graphs[False], coh, [(a, b)], [a, b])[0]
        p_aw = exact_dwell_cost(qc, backend, graphs[True], coh, [(a, b)], [a, b])[0]
        rule = 0.505 + 0.074 * p_old if cls in ("T", "F") else 0.66       # pre-registered from Eagle Aer fits (Entry 110)
        circuits.append({"id": f"dead{cls}_{len(circuits):02d}", "kind": "bell", "dead_class": cls, "pairs": [[a, b]],
                         "hops": [len(em.shortest_path(graph, a, b)) - 1], "v41": p_old, "v41_dead_aware": p_aw, "rule": rule,
                         "v41_stale_snapshot": p_old, "gnn": None})
json.dump({"chip_model": chip, "sim_chip": chip.replace("_live", ""), "num_qubits": nq, "created": time.strftime("%Y-%m-%d %H:%M"),
           "shots": 8192, "dead_edges_targeted": sorted(dead), "dead_qubits_excluded": sorted(dead_q),
           "note": "Entry 113: predictions frozen before hardware run. rule = 0.505+0.074*v41 for T/F, 0.66 for C (Eagle Aer fits).",
           "circuits": circuits}, open(f"hw_validation/manifest_dead_{chip}.json", "w"), indent=1)
print({k: len(v) for k, v in got.items()}, "| dead edges:", sorted(dead), "|", len(circuits), "circuits")
for c in circuits: print(f"  {c['id']:10s} {c['dead_class']} pair {c['pairs'][0]} hops {c['hops'][0]} v4.1 {c['v41']:.3f} aware {c['v41_dead_aware']:.3f} rule {c['rule']:.3f}")
