"""
QuantumBridge — Entry 026: what's left after subtracting the closed-form
baseline from Entries 017-021's raw decay-curve data.

Entry 025 derived success = 0.5 + 0.5*(1-p)^n for a Bell-state (2-of-4 ideal
set) circuit under pure depolarizing error, and verified it against a
synthetic single-source-of-error noise model. Entries 017-021 fitted their
"forgiveness ratio" against REAL hardware noise models (FakeCairoV2,
FakeSherbrooke, FakeKyiv), which include decoherence and readout on top of
gate error. Their fitted ratios are a mixture of the fixed geometric baseline
and whatever real chip physics is underneath it.

This entry separates the two. For every Bell-circuit decay curve on record:

1. Fit k_measured the same way the original entries did (log-linear decay
   constant across the real measured points).
2. Fit k_baseline the SAME way, but on the closed-form formula evaluated at
   the identical gate counts -- this control's sampling matches the real
   data exactly, so the comparison is fair even though the baseline isn't a
   pure exponential.
3. residual_ratio = (k_measured / p) - (k_baseline / p)

If residual_ratio collapses onto a clean trend across chips and edges, that
trend is the real chip-physics signal Entries 017-021 were chasing. If it's
just noise, the geometric baseline was most of the story.

As a consistency check: Entry 021's D-screen (D = gate_error / (gate_duration
/ T2)) already flagged which edges were decoherence-contaminated. Since the
closed-form baseline has NO decoherence term at all, contaminated edges
should show up as large positive residuals here -- decoherence that the
baseline can't see. This is a prediction, not a tuned outcome.
"""

import json

import numpy as np

BELL_IDEAL_SET_SIZE = 2  # {00, 11} out of 4 possible outcomes
FLOOR = BELL_IDEAL_SET_SIZE / 4  # 0.5, the absorbing-state population


def baseline_success(n, p):
    return FLOOR + (1 - FLOOR) * (1 - p) ** n


def fit_k(n_arr, y_arr):
    n_arr = np.asarray(n_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    slope, _ = np.polyfit(n_arr, np.log(y_arr), 1)
    return -slope


def analyze(label, edge, p, results, D=None):
    ns = [r[0] for r in results]
    ys = [r[1] for r in results]
    k_measured = fit_k(ns, ys)
    k_baseline = fit_k(ns, [baseline_success(n, p) for n in ns])
    ratio_measured = k_measured / p
    ratio_baseline = k_baseline / p
    residual = ratio_measured - ratio_baseline
    return {"label": label, "edge": list(edge), "raw_error": p,
           "k_measured": k_measured, "k_baseline": k_baseline,
           "ratio_measured": ratio_measured, "ratio_baseline": ratio_baseline,
           "residual_ratio": residual, "D": D}


if __name__ == "__main__":
    rows = []

    # Entry 017: Cairo (24,25), low error
    g = json.load(open("quantumbridge_data/gate_count_decay_curve.json"))
    rows.append(analyze("cairo_017_low", (24, 25), 0.006, g, D=4.32))

    # Entry 019: Cairo (19,22), high error -- known decoherence-contaminated
    h = json.load(open("quantumbridge_data/high_error_decay_curve.json"))
    rows.append(analyze("cairo_019_high", tuple(h["edge"]), h["raw_edge_error"],
                        h["results"], D=0.31))

    # Entry 020: Sherbrooke, both edges -- both pass the D-screen
    sh = json.load(open("quantumbridge_data/sherbrooke_decay_curves.json"))
    rows.append(analyze("sherbrooke_020_low", tuple(sh["low_edge"]["qubits"]),
                        sh["low_edge"]["raw_error"], sh["low_edge"]["results"], D=2.34))
    rows.append(analyze("sherbrooke_020_high", tuple(sh["high_edge"]["qubits"]),
                        sh["high_edge"]["raw_error"], sh["high_edge"]["results"], D=9.78))

    # Entry 021: Kyiv, 6 edges with known D values from the retroactive screen
    ky = json.load(open("quantumbridge_data/kyiv_decay_curves.json"))
    d_lookup = {(119, 120): 1.12, (28, 29): 0.14, (77, 78): 3.73,
               (45, 46): 1.65, (66, 67): 4.03, (23, 24): 5.01}
    for c in ky["curves"]:
        edge = tuple(c["qubits"])
        rows.append(analyze(f"kyiv_021_{c['label'].replace(' ', '_')}", edge,
                            c["raw_error"], c["results"], D=d_lookup.get(edge)))

    print(f"{'edge':<24}{'raw err':>9}{'D':>7}{'ratio(orig)':>13}"
          f"{'ratio(baseline)':>17}{'residual':>11}  verdict")
    print("=" * 100)
    for r in rows:
        verdict = "CONTAMINATED (D<1.5)" if (r["D"] is not None and r["D"] < 1.5) else "usable"
        print(f"{r['label']:<24}{r['raw_error'] * 100:8.4f}%{r['D']:>7.2f}"
              f"{r['ratio_measured']:>13.4f}{r['ratio_baseline']:>17.4f}"
              f"{r['residual_ratio']:>11.4f}  {verdict}")

    usable = [r for r in rows if r["D"] is not None and r["D"] >= 1.5]
    contaminated = [r for r in rows if r["D"] is not None and r["D"] < 1.5]

    print(f"\n{'=' * 100}")
    print("USABLE EDGES -- does the residual follow a clean power law vs raw error?")
    print("=" * 100)
    xs = np.log([r["raw_error"] for r in usable])
    ys_pos = [r["residual_ratio"] for r in usable]
    if all(y > 0 for y in ys_pos):
        ys = np.log(ys_pos)
        exponent, log_c = np.polyfit(xs, ys, 1)
        pred = np.exp(log_c + exponent * xs)
        ss_res = np.sum((np.exp(ys) - pred) ** 2)
        ss_tot = np.sum((np.exp(ys) - np.mean(np.exp(ys))) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"residual = {np.exp(log_c):.4f} * err^({exponent:+.4f})   R^2 = {r2:.4f}")
        for r, p_ in zip(usable, pred):
            print(f"  {r['label']:<24} residual={r['residual_ratio']:.4f}  "
                  f"fitted={p_:.4f}  resid-of-resid={r['residual_ratio'] - p_:+.4f}")
    else:
        print("some residuals are negative or zero -- no clean power law possible "
              "in log-log space. residual values:")
        for r in usable:
            print(f"  {r['label']:<24} residual={r['residual_ratio']:+.4f}")

    print(f"\n{'=' * 100}")
    print("PREDICTION CHECK -- do D<1.5 (decoherence-contaminated) edges show")
    print("LARGER residuals than the usable-edge trend, since the baseline has")
    print("no decoherence term at all?")
    print("=" * 100)
    mean_usable_resid = float(np.mean([r["residual_ratio"] for r in usable]))
    for r in contaminated:
        print(f"  {r['label']:<24} D={r['D']:.2f}  residual={r['residual_ratio']:+.4f}"
              f"   (usable-edge mean: {mean_usable_resid:+.4f})")

    json.dump({"rows": rows, "usable_mean_residual": mean_usable_resid},
              open("quantumbridge_data/entry026_residual_analysis.json", "w"),
              indent=2, default=str)
    print("\nSaved to quantumbridge_data/entry026_residual_analysis.json")
