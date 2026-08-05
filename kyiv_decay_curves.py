"""
QuantumBridge — Entry 021: Forgiveness-ratio decay curves on FakeKyiv.

METHOD CHANGE vs Entries 017/019 (Cairo) and 020 (Sherbrooke)
------------------------------------------------------------
Those entries measured exactly TWO edges per chip (one low-error, one
high-error) and drew a power law through them. Two points define a power
law exactly, with zero residual — so there was no way to tell whether the
power-law form was actually right, or whether the two points just happened
to lie on some line in log-log space.

Entry 021 measures SIX edges on Kyiv, spanning 0.31% to 4.32% raw gate
error (roughly a 14x span). With six points the exponent is genuinely
fitted, and the residuals become a real goodness-of-fit test on the
power-law assumption itself.

This matters because the Cairo and Sherbrooke exponents disagree
(-0.280 vs -0.212) even though their high/low ratios are nearly identical
(0.629 vs 0.630). That near-identity is suspicious: the two chips were
probed over different error spans (5.2x vs 8.8x), so matching high/low
ratios actually imply DIFFERENT exponents. Kyiv, with a properly fitted
curve, is the tiebreaker.
"""

import json
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeKyiv

backend = FakeKyiv()
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator(noise_model=noise_model)

IDEAL_OUTCOMES = {"00", "11"}
SHOTS = 4096
SEED = 1729

# Six real, calibrated Kyiv edges chosen to spread evenly in log-error space.
# Raw errors read from quantumbridge_data/offline_calibration_kyiv_full.json.
EDGES = [
    ((119, 120), 0.003113, "very low"),
    ((28, 29), 0.006013, "low"),          # matches Cairo's low edge (0.600%)
    ((77, 78), 0.009860, "mid-low"),
    ((45, 46), 0.016181, "mid"),
    ((66, 67), 0.026181, "high"),         # near Cairo/Sherbrooke high edges
    ((23, 24), 0.043229, "very high"),
]


def build_circuit(q1, q2, extra_cx_pairs):
    """Bell pair, then extra_cx_pairs *pairs* of CX on the same edge.
    Pairs cancel logically, so the ideal outcome stays {00, 11} while the
    physical gate count grows — isolating accumulated two-qubit gate error."""
    qc = QuantumCircuit(backend.num_qubits, 2)
    qc.h(q1)
    qc.cx(q1, q2)
    for _ in range(extra_cx_pairs):
        qc.cx(q1, q2)
        qc.cx(q1, q2)
    qc.measure([q1, q2], [0, 1])
    return qc


def success_probability(qc):
    transpiled = transpile(qc, backend=backend, optimization_level=0, seed_transpiler=SEED)
    result = sim.run(transpiled, shots=SHOTS, seed_simulator=SEED).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return sum(c for b, c in counts.items()
               if b.replace(" ", "") in IDEAL_OUTCOMES) / total


def run_curve(q1, q2, raw_edge_err, label):
    print(f"\n{'=' * 56}")
    print(f"{label.upper()} EDGE ({q1},{q2}) — raw error {raw_edge_err * 100:.4f}%")
    print('=' * 56)

    extra_pair_counts = [0, 2, 4, 6, 8, 10, 13, 16, 20]
    results = []
    for extra in extra_pair_counts:
        total_cx = 1 + 2 * extra
        prob = success_probability(build_circuit(q1, q2, extra))
        results.append((total_cx, prob))
        print(f"  {total_cx:>3} CX gates: {prob * 100:6.2f}%")

    n_vals = np.array([r[0] for r in results], dtype=float)
    s_vals = np.array([r[1] for r in results], dtype=float)

    # Exponential decay fit: success = A * exp(-k * n)
    slope, log_A = np.polyfit(n_vals, np.log(s_vals), 1)
    k_fit = -slope

    # R^2 of the decay fit itself, so a bad curve can't silently feed the law
    pred = log_A + slope * n_vals
    ss_res = np.sum((np.log(s_vals) - pred) ** 2)
    ss_tot = np.sum((np.log(s_vals) - np.log(s_vals).mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ratio = k_fit / raw_edge_err
    print(f"\n  Fitted decay constant: {k_fit * 100:.4f}% per gate   (decay R^2 = {r2:.4f})")
    print(f"  Forgiveness ratio:     {ratio:.4f}")
    return results, k_fit, ratio, r2


def fit_power_law(errors, ratios):
    """Least-squares power law ratio = c * err^e, fitted in log-log space."""
    x, y = np.log(np.array(errors)), np.log(np.array(ratios))
    e, log_c = np.polyfit(x, y, 1)
    pred = log_c + e * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return e, float(np.exp(log_c)), r2, np.exp(pred)


if __name__ == "__main__":
    curves = []
    for (q1, q2), raw_err, label in EDGES:
        results, k_fit, ratio, decay_r2 = run_curve(q1, q2, raw_err, label)
        curves.append({
            "qubits": [q1, q2], "label": label, "raw_error": raw_err,
            "results": results, "k": k_fit, "ratio": ratio, "decay_r2": decay_r2,
        })

    errors = [c["raw_error"] for c in curves]
    ratios = [c["ratio"] for c in curves]
    exponent, coefficient, law_r2, fitted = fit_power_law(errors, ratios)

    print(f"\n{'=' * 56}")
    print("POWER-LAW FIT ACROSS 6 KYIV EDGES")
    print('=' * 56)
    print(f"  ratio = {coefficient:.4f} * err^({exponent:.4f})     R^2 = {law_r2:.4f}\n")
    print(f"  {'edge':>10}  {'raw err':>9}  {'measured':>9}  {'fitted':>8}  {'resid':>8}")
    for c, f in zip(curves, fitted):
        resid = c["ratio"] - f
        print(f"  {str(tuple(c['qubits'])):>10}  {c['raw_error'] * 100:8.4f}%  "
              f"{c['ratio']:9.4f}  {f:8.4f}  {resid:+8.4f}")

    print(f"\n{'=' * 56}")
    print("CROSS-CHIP COMPARISON")
    print('=' * 56)
    print(f"  Cairo      (2 pts): exponent = -0.2804   [untestable, zero residual]")
    print(f"  Sherbrooke (2 pts): exponent = -0.2119   [untestable, zero residual]")
    print(f"  Kyiv       (6 pts): exponent = {exponent:+.4f}   R^2 = {law_r2:.4f}")

    lo, hi = curves[0]["ratio"], curves[-1]["ratio"]
    span = curves[-1]["raw_error"] / curves[0]["raw_error"]
    print(f"\n  Kyiv high/low ratio = {hi / lo:.4f} over a {span:.1f}x error span")
    print(f"  (Cairo: 0.6292 over 5.2x;  Sherbrooke: 0.6302 over 8.8x)")
    print("\n  If the high/low ratio is chip-invariant it should hold here too.")
    print("  If instead the EXPONENT is the invariant, Kyiv's larger span")
    print("  should push its high/low ratio well below 0.63.")

    out = {
        "chip": "kyiv",
        "shots": SHOTS,
        "seed": SEED,
        "curves": curves,
        "power_law": {"coefficient": coefficient, "exponent": exponent, "r2": law_r2},
    }
    with open("quantumbridge_data/kyiv_decay_curves.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to quantumbridge_data/kyiv_decay_curves.json")
