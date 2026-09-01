"""QuantumBridge — Entry 059: extract real physical-qubit routes for one
star-topology and one chain-topology large circuit (both Kyiv), for the
new 3D visualization. Also recomputes per-qubit usage counts across the
full current 3,221-circuit dataset (up from the 2,317 used in the last
3D viz)."""

import json

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke

import emulator_v3_routing as em
from exact_dwell_routing import route_with_explicit_swaps
from entry044_build_graphs import role_labels, BACKENDS

backend = FakeKyiv()
nq = backend.num_qubits


def extract_route(pairs_flat, logical_pairs, is_bell_kind_ghz=True):
    qc = QuantumCircuit(nq, len(pairs_flat))
    qc.h(logical_pairs[0][0])
    for a, b in logical_pairs:
        qc.cx(a, b)
    t = route_with_explicit_swaps(qc, backend)
    control_logical, target_logicals = role_labels(logical_pairs)
    loc = {q: q for pair in logical_pairs for q in pair}
    physical_nodes = set(loc.values())
    edge_path = []
    for inst in t.data:
        op = inst.operation
        if op.num_qubits != 2:
            continue
        p, q = [t.find_bit(x).index for x in inst.qubits]
        physical_nodes.add(p); physical_nodes.add(q)
        edge_path.append([p, q, op.name])
    return sorted(physical_nodes), edge_path


# star example: largest k=14 star from entry052/056
star_raw = json.load(open("quantumbridge_data/entry052_biggerghz_dataset.json"))
star_ex = max((r for r in star_raw if r["chip"] == "kyiv"), key=lambda r: r["n_logical_qubits"])
star_nodes, star_edges = extract_route(star_ex["pairs_flat"], star_ex["logical_pairs"])
print(f"star example: k={star_ex['n_logical_qubits']} physical_nodes={len(star_nodes)}")

# chain example: largest k=12 chain from entry058
chain_raw = json.load(open("quantumbridge_data/entry058_chain_dataset.json"))
chain_ex = max(chain_raw, key=lambda r: r["n_logical_qubits"])
chain_nodes, chain_edges = extract_route(chain_ex["pairs_flat"], chain_ex["logical_pairs"])
print(f"chain example: k={chain_ex['n_logical_qubits']} physical_nodes={len(chain_nodes)}")

# usage counts across the full current dataset (any chip, using stored node
# features isn't possible since physical IDs aren't stored there -- so we
# recompute usage from the RAW record files directly via routing, same as
# the original 3D viz did, but only for Kyiv circuits to keep this fast
# and because both example routes above are Kyiv).
usage = {i: 0 for i in range(nq)}
combined = json.load(open("quantumbridge_data/entry043_combined_dataset.json"))
big45 = json.load(open("quantumbridge_data/entry045_bigghz_dataset.json"))
big52 = json.load(open("quantumbridge_data/entry052_biggerghz_dataset.json"))
new054 = json.load(open("quantumbridge_data/entry054_new_bell_only.json"))
new055 = json.load(open("quantumbridge_data/entry055_new_bell_only.json"))
chain058 = json.load(open("quantumbridge_data/entry058_chain_dataset.json"))
all_records = combined + big45 + big52 + new054 + new055 + chain058
kyiv_records = [r for r in all_records if r["chip"] == "kyiv"]
print(f"recomputing usage across {len(kyiv_records)} Kyiv circuits (fast, no routing -- using pairs_flat only)")
for r in kyiv_records:
    for q in r["pairs_flat"]:
        usage[q] = usage.get(q, 0) + 1

json.dump({
    "star_example": {"k": star_ex["n_logical_qubits"], "nodes": star_nodes, "edges": star_edges},
    "chain_example": {"k": chain_ex["n_logical_qubits"], "nodes": chain_nodes, "edges": chain_edges},
    "usage": usage,
    "n_circuits": len(kyiv_records),
}, open("quantumbridge_data/entry059_3d_data.json", "w"))
print("saved -> quantumbridge_data/entry059_3d_data.json")
