"""
QuantumBridge — Entry 022 diagnostic: the FakeCairoV2 basis-gate mismatch.

FINDING
-------
The v3 validation harness has been measuring against a reference that silently
applies no noise to most two-qubit gates in routed circuits.

FakeCairoV2 is a MIXED-BASIS calibration snapshot: 12 of its directed edges are
calibrated as `cx`, the other 14 as `ecr`. Qiskit's NoiseModel.from_backend
keys its quantum errors by (gate name, qubit pair), so it carries a `cx` error
for the first group and an `ecr` error for the second.

validate_v3_circuits.py transpiles with

    basis_gates = backend.operation_names

which hands the transpiler the union {cx, ecr, ...}. The transpiler picks ONE
two-qubit gate for the whole circuit -- `cx` -- and emits it on every edge,
including the 14 that are only calibrated as `ecr`. For those gates the noise
model finds no matching (name, pair) entry and applies nothing at all.

The gate executes perfectly. Silently. For free.

CONSEQUENCE
-----------
On bell_scattered_0_26, 21 of 34 two-qubit gates (62%) run noiselessly. The
"Aer reference" of ~96% is therefore not a noisy-hardware reference at all, and
v3's apparent 14-point overshoot on that circuit is largely the reference being
wrong rather than the model.

This affects every circuit whose route leaves the cx-calibrated subgraph. It
does NOT affect FakeSherbrooke or FakeKyiv, which are uniformly `ecr` with
noise-model coverage on all 144 edges.

Run this script to reproduce the audit.
"""

import collections
import json

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeCairoV2, FakeKyiv, FakeSherbrooke

TOPOLOGY = "quantumbridge_data/real_topology_cairo.json"


def noise_coverage(backend):
    """Map gate name -> set of directed qubit pairs the noise model covers."""
    nm = NoiseModel.from_backend(backend)
    cov = collections.defaultdict(set)
    for err in nm.to_dict()["errors"]:
        ops = err.get("operations", [])
        if ops and len(err["gate_qubits"][0]) == 2:
            cov[ops[0]].add(tuple(err["gate_qubits"][0]))
    return cov


def calibrated_edges_by_gate(backend):
    """Map gate name -> set of undirected pairs with a plausible gate_error."""
    out = collections.defaultdict(set)
    for g in backend.properties().gates:
        if len(g.qubits) == 2:
            e = next((p.value for p in g.parameters if p.name == "gate_error"), None)
            if e is not None and e < 0.5:
                out[g.gate].add(tuple(sorted(g.qubits)))
    return out


def audit_circuit(name, qc, backend, coupling_map, cov):
    t = transpile(qc, coupling_map=coupling_map,
                  basis_gates=backend.operation_names,
                  initial_layout=list(range(backend.num_qubits)),
                  optimization_level=3, seed_transpiler=1)
    counts = collections.Counter()
    for inst in t.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(t.find_bit(q).index for q in inst.qubits)
            gate = inst.operation.name
            hit = pair in cov[gate] or pair[::-1] in cov[gate]
            counts["noisy" if hit else "silent"] += 1
    total = sum(counts.values())
    pct = 100 * counts["silent"] / total if total else 0
    print(f"{name:<26}{total:>10}{counts['noisy']:>8}{counts['silent']:>9}{pct:>8.0f}%")
    return total, counts["silent"]


def build_validation_circuits(n=27):
    qc1 = QuantumCircuit(n, 3); qc1.h(0); qc1.cx(0, 1); qc1.cx(1, 2)
    qc1.measure([0, 1, 2], [0, 1, 2])
    qc2 = QuantumCircuit(n, 3); qc2.h(0); qc2.cx(0, 26); qc2.cx(26, 22)
    qc2.measure([0, 26, 22], [0, 1, 2])
    qc3 = QuantumCircuit(n, 2); qc3.h(24); qc3.cx(24, 25)
    qc3.measure([24, 25], [0, 1])
    qc4 = QuantumCircuit(n, 2); qc4.h(0); qc4.cx(0, 26)
    qc4.measure([0, 26], [0, 1])
    return {"ghz_local_0_1_2": qc1, "ghz_scattered_0_26_22": qc2,
            "bell_adjacent_24_25": qc3, "bell_scattered_0_26": qc4}


if __name__ == "__main__":
    print("=" * 62)
    print("PART 1 — basis composition of each fake backend")
    print("=" * 62)
    for B, label in [(FakeCairoV2, "FakeCairoV2"), (FakeSherbrooke, "FakeSherbrooke"),
                     (FakeKyiv, "FakeKyiv")]:
        b = B()
        cal = calibrated_edges_by_gate(b)
        cov = noise_coverage(b)
        cm = {tuple(sorted(e)) for e in b.coupling_map.get_edges()}
        mixed = len(cal) > 1
        print(f"\n  {label} ({b.num_qubits} qubits, {len(cm)} coupling-map edges)")
        print(f"    calibrated by gate : { {k: len(v) for k, v in cal.items()} }")
        print(f"    noise model by gate: { {k: len(v) for k, v in cov.items()} }")
        print(f"    MIXED BASIS: {mixed}" + ("   <-- transpiler can emit the wrong gate"
                                             if mixed else ""))

    print("\n" + "=" * 62)
    print("PART 2 — silent-gate audit of the v3 validation suite on Cairo")
    print("=" * 62)
    backend = FakeCairoV2()
    edges = json.load(open(TOPOLOGY))
    cm = CouplingMap(couplinglist=edges + [[b, a] for a, b in edges])
    cov = noise_coverage(backend)

    print(f"\n{'Circuit':<26}{'2q gates':>10}{'noisy':>8}{'SILENT':>9}{'% free':>9}")
    print("-" * 62)
    tot_g = tot_s = 0
    for name, qc in build_validation_circuits(backend.num_qubits).items():
        g, s = audit_circuit(name, qc, backend, cm, cov)
        tot_g += g; tot_s += s
    print("-" * 62)
    print(f"{'TOTAL':<26}{tot_g:>10}{tot_g - tot_s:>8}{tot_s:>9}{100 * tot_s / tot_g:>8.0f}%")

    print("\nEvery gate in the SILENT column executed with no error applied.")
    print("Any accuracy figure measured against this reference is not trustworthy.")
