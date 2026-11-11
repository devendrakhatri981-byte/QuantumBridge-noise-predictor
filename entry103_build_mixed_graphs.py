"""
QuantumBridge — Entry 103: graph builder for the MIXED bell/GHZ3 parallel-
component template (Entry 102). Same 7-node / 5-edge feature schema as the
multi-bell builder (entry090) so a multi-bell-trained model can be applied
directly as a template-transfer test:
  node: T1, T2, readout, T1/T2, is_control, is_target, n_components_touched
  edge: err, dur, pos, is_final, component_id_norm
A component's first qubit is its control (gets H and drives every CX); the
rest are targets. Component index plays the role pair index played in
entry090 (a GHZ3 component has two entangling gates sharing one control).
"""
import json
import os

from qiskit import QuantumCircuit

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import route_with_explicit_swaps
import entry089_grow_multibell as g89

IN_PATH = "quantumbridge_data/entry102_mixed.json"
OUT_PATH = "quantumbridge_data/entry103_mixed_graph_dataset.json"
BACKENDS = g89.BACKENDS


def build_circuit(nq, comps):
    qc = QuantumCircuit(nq, sum(len(c) for c in comps))
    for c in comps:
        qc.h(c[0])
        for q in c[1:]:
            qc.cx(c[0], q)
    return qc


def graph_for_record(backend, graph, coh, r):
    comps = [tuple(c) for c in r["components"]]
    n = len(comps)
    t = route_with_explicit_swaps(build_circuit(backend.num_qubits, comps), backend)

    loc, role_at, comp_of = {}, {}, {}
    for i, c in enumerate(comps):
        for j, q in enumerate(c):
            loc[q] = q
            role_at[q] = "control" if j == 0 else "target"
            comp_of[q] = i

    physical_nodes = set(loc.values())
    ever_role, ever_comp, edges = {}, {}, []
    for lg, ph in loc.items():
        ever_role.setdefault(ph, set()).add(role_at[lg])
        ever_comp.setdefault(ph, set()).add(comp_of[lg])

    n_2q = sum(1 for inst in t.data if inst.operation.num_qubits == 2)
    step = 0
    for inst in t.data:
        op = inst.operation
        if op.num_qubits != 2:
            continue
        p, q = [t.find_bit(x).index for x in inst.qubits]
        physical_nodes.add(p); physical_nodes.add(q)
        err = em.edge_error(graph, p, q)
        dur = v4.gate_duration(coh, p, q) * 1e6
        pos = step / max(1, n_2q - 1)
        is_final = 1.0 if op.name != "swap" else 0.0
        cid = 0
        if is_final:
            inv = {v: k for k, v in loc.items()}
            for ph in (p, q):
                lg = inv.get(ph)
                if lg is not None and lg in comp_of:
                    cid = comp_of[lg]; break
        edges.append((p, q, err, dur, pos, is_final, cid / max(1, n - 1)))
        step += 1
        if op.name == "swap":
            new_loc = dict(loc)
            for lg, ph in list(loc.items()):
                if ph == p: new_loc[lg] = q
                elif ph == q: new_loc[lg] = p
            loc = new_loc
            for lg, ph in loc.items():
                ever_role.setdefault(ph, set()).add(role_at[lg])
                ever_comp.setdefault(ph, set()).add(comp_of[lg])

    nodes = sorted(physical_nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}
    feats = []
    for nd in nodes:
        t1 = coh["T1"].get(nd, 1e-4) * 1e6
        t2 = coh["T2"].get(nd, 1e-4) * 1e6
        ro = coh["readout"].get(nd, 0.05)
        roles = ever_role.get(nd, set())
        feats.append([t1, t2, ro, t1 / max(t2, 1e-6),
                      1.0 if "control" in roles else 0.0, 1.0 if "target" in roles else 0.0,
                      float(len(ever_comp.get(nd, set())))])
    elist = [[idx[u], idx[v], e, d, ps, f, c] for u, v, e, d, ps, f, c in edges]
    return {"nodes": feats, "edges": elist, "n_nodes": len(nodes)}


def main():
    records = json.load(open(IN_PATH))
    out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
    cache = {}
    for i, r in enumerate(records):
        if i < len(out):
            continue
        chip = r["chip"]
        if chip not in cache:
            cache[chip] = (BACKENDS[chip](), em.build_connectivity_graph(em.load_calibration(chip), chip),
                           v4.load_coherence(chip))
        b, gr, coh = cache[chip]
        gg = graph_for_record(b, gr, coh, r)
        gg.update({"chip": chip, "kind": "mixed", "n_pairs": r["n_comps"],
                   "n_ghz": r["comp_kinds"].count("ghz3"),
                   "aer_ground_truth": r["aer_ground_truth"],
                   "product_of_marginals_v41": r["product_of_marginals_v41"],
                   "bfs_hop_distances": r["bfs_hop_distances"]})
        out.append(gg)
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            json.dump(out, open(OUT_PATH + ".tmp", "w")); os.replace(OUT_PATH + ".tmp", OUT_PATH)
            print(f"  {i+1}/{len(records)}", flush=True)
    json.dump(out, open(OUT_PATH + ".tmp", "w")); os.replace(OUT_PATH + ".tmp", OUT_PATH)
    print("max nodes", max(x["n_nodes"] for x in out), "max edges", max(len(x["edges"]) for x in out))


if __name__ == "__main__":
    main()
