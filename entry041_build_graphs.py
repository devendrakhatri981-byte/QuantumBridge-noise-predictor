"""
QuantumBridge — Entry 041a: building the graph dataset for a GNN.

WHY THIS EXISTS
---------------
Entries 034-040 built a 1,376-circuit dataset, but every record only
stores SCALAR summary features (hop distance, worst edge error on the BFS
path, worst T1, etc.) -- exactly the kind of hand-engineered features a
tabular model uses, and exactly what a GNN is supposed to make
unnecessary. This script re-derives the REAL per-circuit graph structure
(the actual physical qubits and edges the transpiler used, per Entry
031/032's route-then-entangle finding) for every record already in the
combined dataset, using route_with_explicit_swaps (a transpile call, no
Aer simulation needed since the ground-truth label is already known) --
node features are each physical qubit's own T1/T2/readout, edge features
are each real edge's gate error and duration. This is the input a GNN
needs; the tabular features were a compression of it.
"""

import json

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import route_with_explicit_swaps

BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke}
IN_PATH = "quantumbridge_data/entry040_combined_dataset.json"
OUT_PATH = "quantumbridge_data/entry041_graph_dataset.json"


def build_circuit(chip_nq, r):
    pairs = r["logical_pairs"]
    qc = QuantumCircuit(chip_nq, len(r["pairs_flat"]))
    qc.h(pairs[0][0])
    for a, b in pairs:
        qc.cx(a, b)
    return qc, pairs


def graph_for_record(backend, graph, coh, r):
    qc, pairs = build_circuit(backend.num_qubits, r)
    t = route_with_explicit_swaps(qc, backend)

    loc = {q: q for pair in pairs for q in pair}
    physical_nodes = set(loc.values())
    edges = []  # (u, v, gate_error, duration_us)

    for inst in t.data:
        op = inst.operation
        if op.num_qubits != 2:
            continue
        p, q = [t.find_bit(x).index for x in inst.qubits]
        physical_nodes.add(p); physical_nodes.add(q)
        err = em.edge_error(graph, p, q)
        dur = v4.gate_duration(coh, p, q) * 1e6
        edges.append((p, q, err, dur))
        if op.name == "swap":
            for lg, ph in list(loc.items()):
                if ph == p: loc[lg] = q
                elif ph == q: loc[lg] = p

    nodes = sorted(physical_nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    node_feats = []
    for n in nodes:
        t1 = coh["T1"].get(n, 1e-4) * 1e6
        t2 = coh["T2"].get(n, 1e-4) * 1e6
        ro = coh["readout"].get(n, 0.05)
        node_feats.append([t1, t2, ro])
    edge_list = [[node_index[u], node_index[v], err, dur] for u, v, err, dur in edges]

    return {"nodes": node_feats, "edges": edge_list, "n_nodes": len(nodes)}


def main():
    records = json.load(open(IN_PATH))
    out = json.load(open(OUT_PATH)) if __import__("os").path.exists(OUT_PATH) else []
    done = len(out)
    print(f"resuming with {done}/{len(records)} graphs already built")

    cache = {}
    for i, r in enumerate(records):
        if i < done:
            continue
        chip = r["chip"]
        if chip not in cache:
            backend = BACKENDS[chip]()
            graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
            coh = v4.load_coherence(chip)
            cache[chip] = (backend, graph, coh)
        backend, graph, coh = cache[chip]

        g = graph_for_record(backend, graph, coh, r)
        g.update({"chip": chip, "kind": r["kind"],
                  "aer_ground_truth": r["aer_ground_truth"],
                  "v4_1_prediction": r["v4_1_prediction"],
                  "bfs_hop_distance": r["bfs_hop_distance"]})
        out.append(g)

        if (i + 1) % 100 == 0 or i + 1 == len(records):
            json.dump(out, open(OUT_PATH, "w"))
            print(f"  {i+1}/{len(records)} graphs built, saved checkpoint")

    json.dump(out, open(OUT_PATH, "w"))
    print(f"DONE: {len(out)} graphs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
