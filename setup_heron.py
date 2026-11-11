"""
QuantumBridge — Entry 106 setup: Heron-class chips (156q, native CZ) from fake-backend snapshots.
Writes the same three files the Eagle chips have: real_topology_<chip>.json,
offline_calibration_<chip>_full.json, coherence_<chip>.json.
usage: python3 setup_heron.py fez marrakesh kingston
"""
import json, statistics, sys
from qiskit_ibm_runtime import fake_provider

FAKE = {"fez": "FakeFez", "marrakesh": "FakeMarrakesh", "kingston": "FakeKingston"}

for chip in sys.argv[1:]:
    b = getattr(fake_provider, FAKE[chip])(); props = b.properties()
    edges = sorted({(min(a, c), max(a, c)) for a, c in set(map(tuple, b.coupling_map.get_edges()))})
    json.dump(edges, open(f"quantumbridge_data/real_topology_{chip}.json", "w"), indent=2)
    errs, durs = [], {}
    for g in props.gates:
        if len(g.qubits) == 2 and g.gate == "cz":
            e = next((p.value for p in g.parameters if p.name == "gate_error"), None)
            if e is not None:
                errs.append({"qubits": list(g.qubits), "gate_error": e})
            d = next((p.value for p in g.parameters if p.name == "gate_length"), None)
            if d:
                x, y = sorted(g.qubits); durs[f"{x},{y}"] = d * 1e-9
    json.dump({"chip": chip, "two_qubit_gate_errors": errs},
              open(f"quantumbridge_data/offline_calibration_{chip}_full.json", "w"), indent=2)
    qubits = {}
    for q in range(b.num_qubits):
        try:
            qubits[str(q)] = {"T1": props.t1(q), "T2": props.t2(q), "readout_error": props.readout_error(q)}
        except Exception:
            qubits[str(q)] = {"T1": None, "T2": None, "readout_error": None}
    med = statistics.median(durs.values()) if durs else 5.6e-7
    json.dump({"chip": chip, "qubits": qubits, "gate_durations": durs, "median_gate_duration": med},
              open(f"quantumbridge_data/coherence_{chip}.json", "w"), indent=2)
    print(chip, b.num_qubits, "q |", len(edges), "edges |", len(errs), "cal entries |",
          "median CZ", round(med * 1e9), "ns | basis", sorted(b.operation_names)[:6])
