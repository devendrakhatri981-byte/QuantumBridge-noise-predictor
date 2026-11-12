"""
QuantumBridge — Entry 107: pull the LIVE calibration of a real IBM Heron device (metadata only,
no QPU time) and write the same three chip files the project uses, under the name <chip>_live.
usage (your Mac, python3.12, saved IBM account):  python3.12 pull_live_heron.py ibm_fez
Writes quantumbridge_data/{real_topology,offline_calibration,coherence}_<chip>_live[_full].json
"""
import json, statistics, sys
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

name = sys.argv[1] if len(sys.argv) > 1 else "ibm_fez"
chip = name.replace("ibm_", "") + "_live"
b = QiskitRuntimeService().backend(name)
T = b.target
cz = {k: v for k, v in T["cz"].items() if v is not None}

edges = sorted({(min(a, c), max(a, c)) for a, c in cz})
errs = [{"qubits": list(k), "gate_error": v.error} for k, v in cz.items() if v.error is not None]
durs = {}
for (a, c), v in cz.items():
    if v.duration:
        durs[f"{min(a, c)},{max(a, c)}"] = v.duration          # already seconds
qubits = {}
for q in range(b.num_qubits):
    qp = T.qubit_properties[q]
    ro = T["measure"][(q,)].error if ("measure" in T.operation_names and (q,) in T["measure"]) else None
    qubits[str(q)] = {"T1": qp.t1, "T2": qp.t2, "readout_error": ro}
med = statistics.median(durs.values()) if durs else 8e-8

json.dump(edges, open(f"quantumbridge_data/real_topology_{chip}.json", "w"), indent=2)
json.dump({"chip": chip, "pulled_at": datetime.now().isoformat(), "source": name,
           "two_qubit_gate_errors": errs}, open(f"quantumbridge_data/offline_calibration_{chip}_full.json", "w"), indent=2)
json.dump({"chip": chip, "qubits": qubits, "gate_durations": durs, "median_gate_duration": med},
          open(f"quantumbridge_data/coherence_{chip}.json", "w"), indent=2)
dead = sorted({tuple(sorted(e["qubits"])) for e in errs if (e["gate_error"] or 0) >= 0.5})
print(f"{name}: {b.num_qubits}q, {len(edges)} edges, median CZ {med*1e9:.0f} ns, "
      f"{len(dead)} dead edges (err>=0.5): {dead}")
print(f"saved as chip '{chip}' at {datetime.now().isoformat(timespec='seconds')}")
