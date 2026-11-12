"""Entry 107b: which gate-error accounting matches Aer (gate-error-only) on a CZ chip?
A = current v4.1 (target-path swaps only); B = all swap legs charged; C = B without the 0.5 floor mix.
usage: python3 entry107_gate_variants.py CHIP N [hop_lo hop_hi]"""
import json, random, sys, time
import numpy as np
import entry089_grow_multibell as g
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import route_with_explicit_swaps

chip, N = sys.argv[1], int(sys.argv[2]); lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2, 6)
t0 = time.time(); backend, graph, coh, _ = g.get_ctx(chip); nq = backend.num_qubits
sim = AerSimulator(noise_model=NoiseModel.from_backend(backend, gate_error=True, readout_error=False, thermal_relaxation=False),
                   method="matrix_product_state")
rng = random.Random(77); rows = []
while len(rows) < N and time.time() - t0 < 150:
    a, b = rng.sample(range(nq), 2); p = em.shortest_path(graph, a, b)
    if p is None or not (lo <= len(p) - 1 <= hi): continue
    qc = QuantumCircuit(nq, 2); qc.h(a); qc.cx(a, b)
    t = route_with_explicit_swaps(qc, backend)
    loc = {a: a, b: b}; St = Sall = 1.0; final = 0.0
    for inst in t.data:
        if inst.operation.num_qubits != 2: continue
        x, y = [t.find_bit(q).index for q in inst.qubits]; raw = em.edge_error(graph, x, y)
        if inst.operation.name == "swap":
            f = (1 - raw) ** 3; Sall *= f
            if loc[b] in (x, y): St *= f
            for lg, ph in list(loc.items()):
                loc[lg] = y if ph == x else x if ph == y else ph
        else:
            final = raw
    qm = QuantumCircuit(nq, 2); qm.h(a); qm.cx(a, b); qm.measure(a, 0); qm.measure(b, 1)
    tm = transpile(qm, backend=backend, initial_layout=list(range(nq)), optimization_level=3, seed_transpiler=1)
    cn = sim.run(tm, shots=8192, seed_simulator=1).result().get_counts()
    truth = (cn.get("00", 0) + cn.get("11", 0)) / sum(cn.values())
    A = St * (1 - final / 2) + (1 - St) * .5
    B = Sall * (1 - final / 2) + (1 - Sall) * .5
    C = Sall * (1 - final)
    D = 1.0; n2 = 0
    for inst in tm.data:
        if inst.operation.num_qubits == 2:
            x, y = [tm.find_bit(q).index for q in inst.qubits]; D *= (1 - em.edge_error(graph, x, y)); n2 += 1
    rows.append((len(p) - 1, truth, A, B, C, D, n2, 0.5 + 0.5 * D))
r = np.array(rows)
for x in sorted(rows, key=lambda z: z[0]): print("  hops %d truth %.3f A %.3f D %.3f n2q %d" % (x[0], x[1], x[2], x[5], x[6]))
print(f"{chip} n={len(r)} hops {lo}-{hi} gate-error-only | {time.time()-t0:.0f}s | truth mean {r[:,1].mean():.3f}")
for nm, i in (("A current", 2), ("B all-swaps+floor", 3), ("C all-swaps no floor", 4), ("D prod over TRUE sim circuit", 5), ("E 0.5+0.5*D", 7)):
    print(f"  {nm:28s} mean {r[:,i].mean():.3f} MAE {np.abs(r[:,i]-r[:,1]).mean()*100:5.2f} pts corr {np.corrcoef(r[:,i], r[:,1])[0,1]:.3f}")

print("  mean #2q gates in sim circuit", r[:,6].mean(), "| corr(n2q, truth)", round(np.corrcoef(r[:,6], r[:,1])[0,1],3))
