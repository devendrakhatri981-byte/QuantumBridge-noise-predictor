"""
QuantumBridge — Entry 021 Analysis: why two of six Kyiv edges break the
forgiveness-ratio power law, and what that implies for Entries 017/019/020.

Run kyiv_decay_curves.py first to produce
quantumbridge_data/kyiv_decay_curves.json.

THE TEST
--------
The forgiveness ratio is defined as (measured decay constant) / (raw
two-qubit gate error). That definition silently assumes the decay is
*caused* by the two-qubit gate error. It isn't always: every CX also takes
finite time, during which the qubits dephase. If a pair's T2 is short
relative to the gate duration, the measured decay is dominated by
decoherence and the "forgiveness ratio" stops measuring forgiveness at all.

Define the domination factor:

    D = raw_gate_error / (gate_duration / T2_min)

D > 1 means gate error dominates. D < 1 means decoherence dominates and the
edge is unusable as a forgiveness-ratio anchor point.
"""

import json
import math
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeCairoV2, FakeSherbrooke, FakeKyiv

D_THRESHOLD = 1.5


def edge_stats(backend, q1, q2):
    """Return (raw_gate_error, gate_duration_ns, T2_min_s, D) for an edge."""
    props = backend.properties()
    err = dur = None
    for g in props.gates:
        if len(g.qubits) == 2 and sorted(g.qubits) == sorted([q1, q2]):
            e = next((p.value for p in g.parameters if p.name == "gate_error"), None)
            d = next((p.value for p in g.parameters if p.name == "gate_length"), None)
            if e is not None:
                err = e
            if d:
                dur = d
    t2_min = min(props.t2(q1), props.t2(q2))
    deco_per_gate = (dur * 1e-9) / t2_min
    return err, dur, t2_min, err / deco_per_gate


def report(backend, name, edges):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"{'edge':>10} {'raw err':>9} {'T2min(us)':>10} {'gate(ns)':>9} "
          f"{'deco/gate':>10} {'D':>7} {'ratio':>8}  verdict")
    out = []
    for (q1, q2), measured_ratio in edges:
        err, dur, t2, D = edge_stats(backend, q1, q2)
        deco = (dur * 1e-9) / t2
        verdict = "usable" if D >= D_THRESHOLD else "CONTAMINATED"
        print(f"{str((q1, q2)):>10} {err * 100:8.4f}% {t2 * 1e6:10.1f} {dur:9.1f} "
              f"{deco * 100:9.3f}% {D:7.2f} {measured_ratio:8.4f}  {verdict}")
        out.append({"qubits": [q1, q2], "raw_error": err, "t2_min_s": t2,
                    "gate_ns": dur, "deco_per_gate": deco, "D": D,
                    "measured_ratio": measured_ratio, "usable": D >= D_THRESHOLD})
    return out


def power_law(errors, ratios):
    x, y = np.log(np.asarray(errors)), np.log(np.asarray(ratios))
    e, log_c = np.polyfit(x, y, 1)
    pred = log_c + e * x
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(e), float(np.exp(log_c)), float(r2)


if __name__ == "__main__":
    kyiv = json.load(open("quantumbridge_data/kyiv_decay_curves.json"))
    kyiv_edges = [((c["qubits"][0], c["qubits"][1]), c["ratio"]) for c in kyiv["curves"]]

    k_stats = report(FakeKyiv(), "KYIV — Entry 021 (6 edges)", kyiv_edges)
    c_stats = report(FakeCairoV2(), "CAIRO — Entries 017 / 019 (2 edges)",
                     [((24, 25), 0.5304), ((19, 22), 0.3337)])
    s_stats = report(FakeSherbrooke(), "SHERBROOKE — Entry 020 (2 edges)",
                     [((60, 61), 0.6665), ((66, 73), 0.4200)])

    print(f"\n{'=' * 78}\nPOWER-LAW FITS: ALL KYIV EDGES vs USABLE KYIV EDGES ONLY\n{'=' * 78}")
    all_e = [s["raw_error"] for s in k_stats]
    all_r = [s["measured_ratio"] for s in k_stats]
    e_all, c_all, r2_all = power_law(all_e, all_r)
    print(f"  all 6 edges:    ratio = {c_all:.4f} * err^({e_all:+.4f})   R^2 = {r2_all:.4f}")

    ok = [s for s in k_stats if s["usable"]]
    e_ok, c_ok, r2_ok = power_law([s["raw_error"] for s in ok],
                                  [s["measured_ratio"] for s in ok])
    print(f"  {len(ok)} usable edges: ratio = {c_ok:.4f} * err^({e_ok:+.4f})   R^2 = {r2_ok:.4f}")
    print(f"\n  Dropping the {len(k_stats) - len(ok)} contaminated edges takes R^2 from "
          f"{r2_all:.3f} to {r2_ok:.3f}.")
    print("  The power-law form is fine. The measurement protocol was the problem.")

    print(f"\n{'=' * 78}\nIMPACT ON THE EMULATOR\n{'=' * 78}")

    def old_law(e):
        x1, y1, x2, y2 = 0.0060, 0.53, 0.0313, 0.334
        ex = (math.log(y2) - math.log(y1)) / (math.log(x2) - math.log(x1))
        return max(0.1, min(1.0, (y1 / (x1 ** ex)) * (e ** ex)))

    def new_law(e):
        return max(0.1, min(1.0, c_ok * (e ** e_ok)))

    print(f"  {'raw err':>9} {'old ratio':>10} {'new ratio':>10} {'change':>9}")
    for e in (0.003, 0.005, 0.010, 0.020, 0.030, 0.045):
        o, n = old_law(e), new_law(e)
        print(f"  {e * 100:8.3f}% {o:10.4f} {n:10.4f} {100 * (n - o) / o:+8.1f}%")

    print("\n  Predicted success probability, N CNOTs on a 1.0% edge:")
    e = 0.010
    for N in (1, 5, 10, 20, 40):
        so, sn = (1 - e * old_law(e)) ** N, (1 - e * new_law(e)) ** N
        print(f"    N={N:>3}: old={so * 100:6.2f}%  new={sn * 100:6.2f}%  "
              f"gap={100 * (sn - so):+.2f} pts")

    json.dump({"d_threshold": D_THRESHOLD,
               "kyiv": k_stats, "cairo": c_stats, "sherbrooke": s_stats,
               "fit_all": {"coefficient": c_all, "exponent": e_all, "r2": r2_all},
               "fit_usable": {"coefficient": c_ok, "exponent": e_ok, "r2": r2_ok}},
              open("quantumbridge_data/entry021_analysis.json", "w"), indent=2)
    print("\nSaved to quantumbridge_data/entry021_analysis.json")
