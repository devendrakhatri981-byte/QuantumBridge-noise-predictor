"""
QuantumBridge — Entry 044a: feature-engineered graph dataset.

WHY THIS EXISTS
---------------
Entry 043 showed a small growth increment (+4%) was too noisy to prove
volume alone keeps helping. Rather than keep grinding for more of the
SAME features, this adds features the GNN currently cannot see at all:

  1. node role (control / target / relay / measured) -- Entry 032 proved
     the CONTROL qubit's routing survival has zero effect on final
     success while the TARGET's is everything. Entry 041/042's graph
     only gave the GNN raw T1/T2/readout per node with no idea which
     node is playing which logical role -- it had to try to infer this
     from graph position alone. Making it explicit is the single most
     direct way to hand the model a finding this project already proved.
  2. T1/T2 ratio per node -- Entry 034/036's flagged anomaly (qubit 5's
     unusually large T1/T2 split) is a ratio, not visible from the two
     raw numbers without the network first learning to compute a divide.
  3. normalized edge position in the route -- lets the model distinguish
     "early swap, far from the entangling gate" from "the entangling
     gate itself" without inferring it from local structure alone.

Re-derives every graph from scratch (chip + qubit pairs are already
saved per record) at 6 node features / 5 edge features instead of
Entry 041's 3 node / 2 edge features. Same circuits, same labels --
only the representation given to the model changes.
"""

import json
import os

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke, FakeBrisbane

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import route_with_explicit_swaps

BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke, "brisbane": FakeBrisbane}
IN_PATH = "quantumbridge_data/entry043_combined_dataset.json"
OUT_PATH = "quantumbridge_data/entry044_graph_dataset.json"


def build_circuit(chip_nq, r):
    pairs = r["logical_pairs"]
    qc = QuantumCircuit(chip_nq, len(r["pairs_flat"]))
    qc.h(pairs[0][0])
    for a, b in pairs:
        qc.cx(a, b)
    return qc, pairs


def role_labels(pairs):
    """control_logical = whichever qubit gets H (pairs[0][0]); every other
    qubit appearing is a target (Entry 032's terminology, generalized to
    GHZ where there are several target legs off one control)."""
    control = pairs[0][0]
    targets = sorted(set(q for pr in pairs for q in pr) - {control})
    return control, targets


def graph_for_record(backend, graph, coh, r):
    qc, pairs = build_circuit(backend.num_qubits, r)
    t = route_with_explicit_swaps(qc, backend)
    control_logical, target_logicals = role_labels(pairs)

    loc = {q: q for pair in pairs for q in pair}
    role_at = {q: ("control" if q == control_logical else "target") for q in loc}
    physical_nodes = set(loc.values())
    edges = []
    ever_role = {}  # physical qubit -> set of roles it ever carried
    for lg, ph in loc.items():
        ever_role.setdefault(ph, set()).add(role_at[lg])

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
        edges.append((p, q, err, dur, pos, is_final))
        step += 1
        if op.name == "swap":
            new_loc = dict(loc)
            for lg, ph in list(loc.items()):
                if ph == p: new_loc[lg] = q
                elif ph == q: new_loc[lg] = p
            loc = new_loc
            for lg, ph in loc.items():
                ever_role.setdefault(ph, set()).add(role_at[lg])

    nodes = sorted(physical_nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    node_feats = []
    for n in nodes:
        t1 = coh["T1"].get(n, 1e-4) * 1e6
        t2 = coh["T2"].get(n, 1e-4) * 1e6
        ro = coh["readout"].get(n, 0.05)
        t1_t2_ratio = t1 / max(t2, 1e-6)
        roles = ever_role.get(n, set())
        is_control = 1.0 if "control" in roles else 0.0
        is_target = 1.0 if "target" in roles else 0.0
        node_feats.append([t1, t2, ro, t1_t2_ratio, is_control, is_target])

    edge_list = [[node_index[u], node_index[v], err, dur, pos, is_final]
                for u, v, err, dur, pos, is_final in edges]

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
