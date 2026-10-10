"""
QuantumBridge — Entry 090: graph feature schema for parallel multi-bell-pair
circuits. Extends entry044_build_graphs.py's approach (control/target role
labels tracked through the real SWAP-preserving transpile) to handle N
co-equal entangling events instead of one, per the plan agreed after
Entry 089's sanity check confirmed a real crosstalk/joint-effect signal
worth modeling (mean gap of 8.2pts between real joint success and the
naive independence assumption, across 20 test circuits).

Two schema changes from the single-pair/GHZ/star/chain templates:
  1. Node features gain a 7th dimension: how many DIFFERENT logical pairs'
     routing ever touched this physical qubit. A qubit shared across two
     pairs' SWAP paths is a direct physical channel for the crosstalk this
     template exists to capture; the single-pair schema had no way to
     represent this because it never happens with one pair.
  2. Edge features gain a 5th dimension: which of the N pairs this edge's
     gate belongs to (normalized 0..1). Every prior template had at most
     one "the gate that matters" so never needed to disambiguate.
  is_final keeps its old meaning unchanged (real entangling gate vs SWAP)
  since that logic already generalizes correctly to multiple non-swap
  gates without modification -- confirmed by inspecting entry044's loop,
  which marks is_final via op.name != "swap", not via a single-gate count.

This is a SEPARATE builder/feature space from entry044_build_graphs.py's
6-node/4-edge schema, not a drop-in replacement -- multi-bell circuits
need their own model, not a fold-in to the existing chip-generalist
pipeline, since the feature dimensions themselves differ.
"""

import json
import os

from qiskit import QuantumCircuit

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import route_with_explicit_swaps

IN_PATH = "quantumbridge_data/entry089_multibell_sanity.json"
OUT_PATH = "quantumbridge_data/entry090_multibell_graph_dataset.json"

from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke, FakeBrisbane, FakeOsaka, FakeQuebec
BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke, "brisbane": FakeBrisbane,
            "osaka": FakeOsaka, "quebec": FakeQuebec}


def build_circuit(chip_nq, pairs):
    n = len(pairs)
    qc = QuantumCircuit(chip_nq, 2 * n)
    for (a, b) in pairs:
        qc.h(a); qc.cx(a, b)
    return qc


def graph_for_multibell_record(backend, graph, coh, r):
    pairs = [tuple(p) for p in r["logical_pairs"]]
    n = len(pairs)
    qc = build_circuit(backend.num_qubits, pairs)
    t = route_with_explicit_swaps(qc, backend)

    # role_at: every logical qubit's role (control = gets H, target = gets
    # CX'd into) AND which pair index it belongs to.
    loc = {}
    role_at = {}
    pair_of_logical = {}
    for i, (a, b) in enumerate(pairs):
        loc[a] = a; loc[b] = b
        role_at[a] = "control"; role_at[b] = "target"
        pair_of_logical[a] = i; pair_of_logical[b] = i

    physical_nodes = set(loc.values())
    edges = []
    ever_role = {}          # physical qubit -> set of roles ever carried
    ever_pairs = {}         # physical qubit -> set of pair indices ever carried
    for lg, ph in loc.items():
        ever_role.setdefault(ph, set()).add(role_at[lg])
        ever_pairs.setdefault(ph, set()).add(pair_of_logical[lg])

    step = 0
    n_2q_total = sum(1 for inst in t.data if inst.operation.num_qubits == 2)
    for inst in t.data:
        op = inst.operation
        if op.num_qubits != 2:
            continue
        p, q = [t.find_bit(x).index for x in inst.qubits]
        physical_nodes.add(p); physical_nodes.add(q)
        err = em.edge_error(graph, p, q)
        dur = v4.gate_duration(coh, p, q) * 1e6
        pos = step / max(1, n_2q_total - 1)
        is_final = 1.0 if (op.name != "swap") else 0.0

        # which pair does this edge belong to? for a genuine entangling
        # gate, (p, q) are exactly the two physical qubits currently
        # holding one pair's control+target logical qubits -- look up via
        # the inverse of loc.
        pair_id = 0
        if is_final:
            inv = {v: k for k, v in loc.items()}
            lg_p, lg_q = inv.get(p), inv.get(q)
            if lg_p is not None and lg_p in pair_of_logical:
                pair_id = pair_of_logical[lg_p]
            elif lg_q is not None and lg_q in pair_of_logical:
                pair_id = pair_of_logical[lg_q]
        pair_id_norm = pair_id / max(1, n - 1)

        edges.append((p, q, err, dur, pos, is_final, pair_id_norm))
        step += 1
        if op.name == "swap":
            new_loc = dict(loc)
            for lg, ph in list(loc.items()):
                if ph == p: new_loc[lg] = q
                elif ph == q: new_loc[lg] = p
            loc = new_loc
            for lg, ph in loc.items():
                ever_role.setdefault(ph, set()).add(role_at[lg])
                ever_pairs.setdefault(ph, set()).add(pair_of_logical[lg])

    nodes = sorted(physical_nodes)
    node_index = {nd: i for i, nd in enumerate(nodes)}
    node_feats = []
    for nd in nodes:
        t1 = coh["T1"].get(nd, 1e-4) * 1e6
        t2 = coh["T2"].get(nd, 1e-4) * 1e6
        ro = coh["readout"].get(nd, 0.05)
        t1_t2_ratio = t1 / max(t2, 1e-6)
        roles = ever_role.get(nd, set())
        is_control = 1.0 if "control" in roles else 0.0
        is_target = 1.0 if "target" in roles else 0.0
        n_pairs_touched = float(len(ever_pairs.get(nd, set())))
        node_feats.append([t1, t2, ro, t1_t2_ratio, is_control, is_target, n_pairs_touched])

    edge_list = [[node_index[u], node_index[v], err, dur, pos, is_final, pid]
                for u, v, err, dur, pos, is_final, pid in edges]

    return {"nodes": node_feats, "edges": edge_list, "n_nodes": len(nodes)}


def main():
    records = json.load(open(IN_PATH))
    out = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
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

        g = graph_for_multibell_record(backend, graph, coh, r)
        g.update({"chip": chip, "kind": "multibell", "n_pairs": r["n_pairs"],
                  "aer_ground_truth": r["aer_ground_truth"],
                  "product_of_marginals_v41": r["product_of_marginals_v41"],
                  "bfs_hop_distances": r["bfs_hop_distances"]})
        out.append(g)

        if (i + 1) % 50 == 0 or i + 1 == len(records):
            tmp = OUT_PATH + ".tmp"
            json.dump(out, open(tmp, "w"))
            os.replace(tmp, OUT_PATH)
            print(f"  {i+1}/{len(records)} graphs built, checkpoint saved")

    tmp = OUT_PATH + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, OUT_PATH)
    sizes = [g["n_nodes"] for g in out]
    print(f"DONE: {len(out)} graphs -> {OUT_PATH}")
    print(f"max n_nodes={max(sizes)}, max n_edges={max(len(g['edges']) for g in out)}")


if __name__ == "__main__":
    main()
