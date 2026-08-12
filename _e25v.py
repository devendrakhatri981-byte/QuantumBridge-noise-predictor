"""Quick cross-check: does success = 0.5 + 0.5*(1-p)^n hold at a very
different p, and for a 3-outcome GHZ ideal set (should give a different
absorbing-state constant, testing the derivation generalizes)."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeKyiv
import emulator_v3_routing as em

backend = FakeKyiv(); NQ = backend.num_qubits
graph = em.build_connectivity_graph(em.load_calibration('kyiv'), 'kyiv')

def iso_nm(edge, p):
    nm = NoiseModel()
    err = depolarizing_error(p, 2)
    for g in ("cx", "ecr"):
        if g in backend.operation_names:
            nm.add_quantum_error(err, g, list(edge))
    return nm

# high-error edge, Bell pair (2-of-4 ideal outcomes)
edge = (23, 24)
p = em.edge_error(graph, *edge)
sim = AerSimulator(noise_model=iso_nm(edge, p))
print(f"edge {edge}  p={p*100:.4f}%  (high-error cross-check)")
for n in [5, 21, 51]:
    extra = (n-1)//2
    qc = QuantumCircuit(NQ, 2); qc.h(edge[0]); qc.cx(*edge)
    for _ in range(extra): qc.cx(*edge); qc.cx(*edge)
    qc.measure(list(edge), [0,1])
    t = transpile(qc, backend=backend, initial_layout=list(range(NQ)), optimization_level=0)
    vals=[]
    for sd in (1,2,3,4):
        c = sim.run(t, shots=8192, seed_simulator=sd).result().get_counts()
        tot = sum(c.values())
        vals.append(sum(v for b,v in c.items() if b.replace(" ","") in {"00","11"})/tot)
    aer = np.mean(vals)
    pred = 0.5 + 0.5*(1-p)**n
    print(f"  n={n:>3}  Aer={aer*100:7.3f}%  predicted={pred*100:7.3f}%  diff={100*(aer-pred):+.3f}")

# GHZ-3 ideal set is {000,111} -- 2 of 8 outcomes, so a fully depolarized
# state gives P(ideal) = 2/8 = 0.25, not 0.5. Formula becomes:
# success = 0.25 + 0.75*(1-p)^n  (survive with prob (1-p)^n, else 1/4 chance)
edge2 = (77, 78); p2 = em.edge_error(graph, *edge2)
sim2 = AerSimulator(noise_model=iso_nm(edge2, p2))
print(f"\nedge {edge2}  p={p2*100:.4f}%  (GHZ-3, ideal set size 2/8)")
for n in [3, 15, 33]:
    qc = QuantumCircuit(NQ, 3)
    qc.h(0); qc.cx(0, edge2[0])
    extra = (n-2)  # roughly distribute extra cancelling pairs on the noisy edge only
    qc.cx(*edge2)
    for _ in range(max(0,extra)//2): qc.cx(*edge2); qc.cx(*edge2)
    qc.measure([0, edge2[0], edge2[1]], [0,1,2])
    t = transpile(qc, backend=backend, initial_layout=list(range(NQ)), optimization_level=0)
    vals=[]
    for sd in (1,2,3,4):
        c = sim2.run(t, shots=8192, seed_simulator=sd).result().get_counts()
        tot = sum(c.values())
        vals.append(sum(v for b,v in c.items() if b.replace(" ","") in {"000","111"})/tot)
    aer = np.mean(vals)
    real_n = 1 + 2*(max(0,extra)//2)
    pred = 0.25 + 0.75*(1-p2)**real_n
    print(f"  n~{real_n:>3}  Aer={aer*100:7.3f}%  predicted={pred*100:7.3f}%  diff={100*(aer-pred):+.3f}")
