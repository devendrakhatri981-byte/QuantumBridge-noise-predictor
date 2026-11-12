"""Entry 107: per-channel ablation of v4.1 vs Aer on a Heron/CZ chip (same method as Entry 028).
For long-route single bell pairs, run Aer with ONE noise channel at a time and compare to v4.1's
decomposed channel survivals. usage: python3 entry107_cz_ablation.py CHIP N"""
import json, random, sys, time
import numpy as np
import entry089_grow_multibell as g
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import emulator_v3_routing as em
from entry092_decomposed_prediction import exact_dwell_cost_decomposed

chip, N = sys.argv[1], int(sys.argv[2]); t0 = time.time()
backend, graph, coh, _ = g.get_ctx(chip); nq = backend.num_qubits
cfg = {"all": dict(gate_error=True, readout_error=True, thermal_relaxation=True),
       "gate": dict(gate_error=True, readout_error=False, thermal_relaxation=False),
       "deco": dict(gate_error=False, readout_error=False, thermal_relaxation=True),
       "read": dict(gate_error=False, readout_error=True, thermal_relaxation=False)}
sims = {k: AerSimulator(noise_model=NoiseModel.from_backend(backend, **v), method="matrix_product_state") for k, v in cfg.items()}
rng = random.Random(107); rows = []
while len(rows) < N and time.time() - t0 < 150:
    a, b = rng.sample(range(nq), 2); p = em.shortest_path(graph, a, b)
    if p is None or not (4 <= len(p) - 1 <= 6): continue
    qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b); qc.measure(a, 0); qc.measure(b, 1)
    t = transpile(qc, backend=backend, initial_layout=list(range(nq)), optimization_level=3, seed_transpiler=1)
    r = {"hops": len(p) - 1}
    for k, s in sims.items():
        cn = s.run(t, shots=8192, seed_simulator=1).result().get_counts()
        r[k] = (cn.get("00", 0) + cn.get("11", 0)) / sum(cn.values())
    qb = QuantumCircuit(nq, 2); qb.h(a); qb.cx(a, b)
    d = exact_dwell_cost_decomposed(qb, backend, graph, coh, [(a, b)], [a, b])
    r.update(p_total=d["total_success"], p_gate=d["gate_error_survival"], p_deco=d["decoherence_survival"],
             p_sq=d["sq_gate_survival"], p_read=d["readout_survival"]); rows.append(r)
m = lambda k: np.mean([r[k] for r in rows])
print(f"{chip}: n={len(rows)} hops 4-6 | {time.time()-t0:.0f}s")
print(f"  total : Aer {m('all'):.3f} vs v4.1 {m('p_total'):.3f}")
print(f"  2q gate error only : Aer {m('gate'):.3f} vs v4.1 gate {m('p_gate'):.3f}")
print(f"  thermal only       : Aer {m('deco'):.3f} vs v4.1 deco {m('p_deco'):.3f}")
print(f"  readout only       : Aer {m('read'):.3f} vs v4.1 read {m('p_read'):.3f}")
json.dump(rows, open(f"quantumbridge_data/entry107_{chip}_ablation.json", "w"))
