"""
QuantumBridge — Entry 023: export per-qubit coherence data for the v4 model.

v3 modelled two-qubit gate error only. Entry 022 showed that leaves it blind to
the dominant failure mode on real routes: a single low-T2 qubit on the path.
v4 needs T1, T2 and gate durations, none of which the existing
offline_calibration_*_full.json exports carry.

Writes quantumbridge_data/coherence_<chip>.json:

    {"chip": ..., "qubits": {"<q>": {"T1": s, "T2": s, "readout_error": p}, ...},
     "gate_durations": {"<a>,<b>": seconds, ...},
     "median_gate_duration": seconds}

Usage:  python export_coherence.py kyiv sherbrooke cairo
"""

import json
import statistics
import sys

FAKE = {
    "kyiv": "FakeKyiv",
    "sherbrooke": "FakeSherbrooke",
    "cairo": "FakeCairoV2",
    "brisbane": "FakeBrisbane",
}


def export(chip):
    from qiskit_ibm_runtime import fake_provider
    backend = getattr(fake_provider, FAKE[chip])()
    props = backend.properties()

    qubits = {}
    for q in range(backend.num_qubits):
        try:
            qubits[str(q)] = {"T1": props.t1(q), "T2": props.t2(q),
                              "readout_error": props.readout_error(q)}
        except Exception:
            qubits[str(q)] = {"T1": None, "T2": None, "readout_error": None}

    durations = {}
    for g in props.gates:
        if len(g.qubits) == 2:
            d = next((p.value for p in g.parameters if p.name == "gate_length"), None)
            if d:
                a, b = sorted(g.qubits)
                durations[f"{a},{b}"] = d * 1e-9  # ns -> s

    med = statistics.median(durations.values()) if durations else 5.6e-7
    out = {"chip": chip, "qubits": qubits, "gate_durations": durations,
           "median_gate_duration": med}

    path = f"quantumbridge_data/coherence_{chip}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    t2s = sorted((v["T2"], int(k)) for k, v in qubits.items() if v["T2"])
    print(f"\n[{chip}] {backend.num_qubits} qubits, {len(durations)} gate durations")
    print(f"  median gate duration : {med * 1e9:.1f} ns")
    print(f"  T2 median            : {statistics.median(t[0] for t in t2s) * 1e6:.1f} us")
    print(f"  T2 worst 5           : " +
          ", ".join(f"q{q}={v * 1e6:.1f}us" for v, q in t2s[:5]))
    # how many qubits are bad enough that decoherence beats a typical gate error?
    danger = [q for v, q in t2s if med / v > 0.01]
    print(f"  qubits where decoherence/gate > 1%: {len(danger)}"
          + (f" -> {danger[:12]}" if danger else ""))
    print(f"  saved -> {path}")


if __name__ == "__main__":
    chips = sys.argv[1:] or ["kyiv"]
    for c in chips:
        if c not in FAKE:
            print(f"unknown chip {c!r}; known: {list(FAKE)}")
            continue
        export(c)
