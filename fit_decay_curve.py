"""
Fit the empirical gate-count decay curve to an exponential model:
    success(n) = A * exp(-k * n)
and compare the fitted per-gate decay constant `k` against the raw
calibrated per-edge error, to quantify the "ideal-outcome forgiveness"
effect.
"""

import json
import numpy as np

with open("quantumbridge_data/gate_count_decay_curve.json") as f:
    data = json.load(f)

n_vals = np.array([d[0] for d in data])
success_vals = np.array([d[1] for d in data])

# Fit ln(success) = ln(A) - k*n  →  linear regression
log_success = np.log(success_vals)
k_fit, log_A = np.polyfit(n_vals, log_success, 1)
k_fit = -k_fit
A_fit = np.exp(log_A)

print(f"Fitted model: success(n) = {A_fit:.4f} * exp(-{k_fit:.5f} * n)")
print(f"Effective per-gate decay constant: {k_fit*100:.4f}% per gate")

RAW_EDGE_ERROR = 0.006  # calibrated (24,25) direct error, Entry 013
forgiveness_ratio = k_fit / RAW_EDGE_ERROR
print(f"\nRaw calibrated per-edge error (24,25): {RAW_EDGE_ERROR*100:.2f}%")
print(f"Fitted effective decay:                 {k_fit*100:.2f}%")
print(f"Forgiveness ratio (fitted / raw):        {forgiveness_ratio:.2f}")

# Sanity check: how well does the fit predict the real data?
print(f"\n{'CX count':>10} {'Real':>10} {'Fitted':>10} {'Diff':>8}")
print("=" * 42)
for n, real in zip(n_vals, success_vals):
    predicted = A_fit * np.exp(-k_fit * n)
    print(f"{n:>10} {real*100:>9.2f}% {predicted*100:>9.2f}% {abs(real-predicted)*100:>7.2f}pts")

with open("quantumbridge_data/fitted_decay_model.json", "w") as f:
    json.dump({"A": A_fit, "k": k_fit, "raw_edge_error": RAW_EDGE_ERROR,
               "forgiveness_ratio": forgiveness_ratio}, f, indent=2)
print("\nSaved fitted model to quantumbridge_data/fitted_decay_model.json")
