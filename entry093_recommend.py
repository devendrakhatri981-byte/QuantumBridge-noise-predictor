"""
QuantumBridge — Entry 093: best chip/route recommender.

Turns the model from "grade this specific circuit" into "help me pick
where to run it." Built entirely on top of data we already have --
entry081_lookup_with_uncertainty.json already holds GNN mean+uncertainty
and v4.1 predictions for every possible pair on all 5 chips (kyiv,
sherbrooke, brisbane, osaka trained; quebec zero-shot) -- so this is a
ranking/filtering layer, not a new model.

Each recommendation is also annotated with a failure-mode breakdown
(Entry 092) computed live for the winning candidate, so the answer isn't
just "chip X, qubits A-B, 94% predicted" but also "and here's why: this
route's biggest loss is decoherence at qubit 42."

recommend() usage:
    recommend(min_success=0.85, max_hop=5, chips=["kyiv","sherbrooke"])
    recommend(top_k=5)  # best pairs across all 5 chips, no filters
"""

import json

import emulator_v3_routing as em
import emulator_v4 as v4
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke, FakeBrisbane, FakeOsaka, FakeQuebec
from entry092_decomposed_prediction import exact_dwell_cost_decomposed, summarize_failure_mode

LOOKUP_PATH = "quantumbridge_data/entry081_lookup_with_uncertainty.json"
BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke, "brisbane": FakeBrisbane,
           "osaka": FakeOsaka, "quebec": FakeQuebec}
TRAINED_CHIPS = {"kyiv", "sherbrooke", "brisbane", "osaka"}  # quebec is zero-shot, flagged separately

_ctx_cache = {}


def _get_ctx(chip):
    if chip not in _ctx_cache:
        backend = BACKENDS[chip]()
        graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
        coh = v4.load_coherence(chip)
        _ctx_cache[chip] = (backend, graph, coh)
    return _ctx_cache[chip]


def recommend(min_success=0.0, max_hop=None, max_uncertainty=None, chips=None, top_k=5,
             lookup_path=LOOKUP_PATH, explain_top=True):
    """Rank (chip, qubit_a, qubit_b) bell-pair candidates by predicted GNN
    success, applying optional filters. Returns top_k candidates, each with
    a live-computed failure-mode breakdown for the winner if explain_top."""
    lookup = json.load(open(lookup_path))
    chips = chips or list(lookup.keys())

    candidates = []
    for chip in chips:
        if chip not in lookup:
            continue
        for row in lookup[chip]:
            a, b, hop, n_nodes, n_edges, v41, gnn_mean, gnn_std, real = row
            if gnn_mean < min_success:
                continue
            if max_hop is not None and hop > max_hop:
                continue
            if max_uncertainty is not None and gnn_std is not None and gnn_std > max_uncertainty:
                continue
            candidates.append({
                "chip": chip, "qubit_a": a, "qubit_b": b, "hop_distance": hop,
                "predicted_success": gnn_mean, "uncertainty": gnn_std,
                "v41_prediction": v41, "real_ground_truth": real,
                "is_zero_shot": chip not in TRAINED_CHIPS,
            })

    candidates.sort(key=lambda c: (-c["predicted_success"], c["uncertainty"] or 1.0))
    top = candidates[:top_k]

    if explain_top and top:
        best = top[0]
        backend, graph, coh = _get_ctx(best["chip"])
        a, b = best["qubit_a"], best["qubit_b"]
        qc = QuantumCircuit(backend.num_qubits, 2)
        qc.h(a); qc.cx(a, b)
        breakdown = exact_dwell_cost_decomposed(qc, backend, graph, coh, [(a, b)], [a, b])
        best["explanation"] = summarize_failure_mode(breakdown)

    return top


if __name__ == "__main__":
    print("=== Best 5 pairs overall, no filters ===")
    for c in recommend(top_k=5, explain_top=False):
        flag = " [ZERO-SHOT]" if c["is_zero_shot"] else ""
        print(f"  {c['chip']:>10}{flag} ({c['qubit_a']:>3},{c['qubit_b']:>3}) hop={c['hop_distance']:>2} "
             f"pred={c['predicted_success']*100:.2f}% +/-{(c['uncertainty'] or 0)*100:.2f}")

    print("\n=== Best pair, trained chips only, hop<=5, with explanation ===")
    top = recommend(max_hop=5, chips=["kyiv", "sherbrooke", "brisbane", "osaka"], top_k=1)
    if top:
        c = top[0]
        print(f"  Recommended: {c['chip']} qubits ({c['qubit_a']},{c['qubit_b']}), "
             f"hop={c['hop_distance']}, predicted={c['predicted_success']*100:.2f}% "
             f"+/-{(c['uncertainty'] or 0)*100:.2f}")
        exp = c["explanation"]
        print(f"  Dominant failure mode: {exp['dominant_failure_mode']}")
        print(f"  Channel loss breakdown: {exp['channel_loss']}")
        print(f"  Weak link: {exp['weak_link']}")

    print("\n=== Realistic ask: 'success >= 80%, chip I haven't tried yet (Quebec)' ===")
    top = recommend(min_success=0.80, chips=["quebec"], top_k=3, explain_top=False)
    for c in top:
        print(f"  quebec [ZERO-SHOT] ({c['qubit_a']},{c['qubit_b']}) hop={c['hop_distance']} "
             f"pred={c['predicted_success']*100:.2f}% +/-{(c['uncertainty'] or 0)*100:.2f}")
