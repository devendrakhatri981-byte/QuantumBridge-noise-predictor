import json
import pandas as pd
import glob

records = []

for filepath in sorted(glob.glob("quantumbridge_data/gate_stress_*/run_*.json")):
    with open(filepath) as f:
        data = json.load(f)
        counts = data["counts"]
        total_shots = data["shots"]

        # Same ideal outcome as any Bell state: only 00 or 11 expected
        error_count = counts.get('01', 0) + counts.get('10', 0)
        error_rate = round((error_count / total_shots) * 100, 2)

        records.append({
            "circuit_type": data["circuit_type"],
            "single_qubit_gates": data["single_qubit_gates"],
            "backend": data["backend"],
            "error_rate": error_rate,
        })

df = pd.DataFrame(records)

print("Backend check (should all be ibm_fez):")
print(df["backend"].unique())

print("\nError rate by single-qubit gate count:")
summary = df.groupby("single_qubit_gates")["error_rate"].agg(["mean", "std", "min", "max"]).round(2)
print(summary)

# Reference point: original bell_state data had 0 extra single-qubit gates, 3.80% mean
print("\nReference — original bell_state (0 extra gates, ibm_fez): 3.80% mean (from Entry 005)")

# Simple linear check: does error increase per single-qubit gate?
import numpy as np
x = df["single_qubit_gates"].values
y = df["error_rate"].values
slope, intercept = np.polyfit(x, y, 1)
print(f"\nRough slope: {slope:.4f} percentage points per extra single-qubit gate")
print(f"(For comparison, CNOT cost was ~1.6-1.8 points per gate)")

df.to_csv("quantumbridge_data/entry008_dataset.csv", index=False)
print("\nSaved to quantumbridge_data/entry008_dataset.csv")
