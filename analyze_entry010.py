import json
import pandas as pd
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

records = []
for filepath in sorted(glob.glob("quantumbridge_data/gate_stress_*_v2/run_*.json")):
    with open(filepath) as f:
        data = json.load(f)
        counts = data["counts"]
        total_shots = data["shots"]
        error_count = counts.get('01', 0) + counts.get('10', 0)
        error_rate = round((error_count / total_shots) * 100, 2)

        records.append({
            "circuit_type": data["circuit_type"],
            "sq_gate_count": data["single_qubit_gates"],
            "backend": data["backend"],
            "error_rate": error_rate,
        })

df = pd.DataFrame(records)

print("This session's data — all same chip, same day:")
summary = df.groupby("sq_gate_count")["error_rate"].agg(["mean", "std", "min", "max"]).round(2)
print(summary)

x = df["sq_gate_count"].values
y = df["error_rate"].values
slope, intercept = np.polyfit(x, y, 1)
print(f"\nDeconfounded slope: {slope:.4f} percentage points per single-qubit gate")
print(f"Deconfounded intercept (0 gates): {intercept:.2f}%")
print(f"\nCompare to Entry 008's confounded estimate: ~0.10 pts/gate")
print(f"Compare to Entry 009's confounded 3-feature model estimate: 0.224 pts/gate")

df.to_csv("quantumbridge_data/entry010_dataset.csv", index=False)
print(f"\nSaved to quantumbridge_data/entry010_dataset.csv")
