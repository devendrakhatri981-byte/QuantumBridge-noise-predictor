import json
import pandas as pd
import glob

records = []

for filepath in sorted(glob.glob("quantumbridge_data/*/run_*.json")):
    with open(filepath) as f:
        data = json.load(f)
        counts = data["counts"]
        total_shots = data["shots"]
        num_qubits = data["num_qubits"]

        # For 1-qubit circuits, "error" doesn't apply the same way — skip that calc
        if num_qubits == 1:
            error_rate = None
        else:
            # Count all outcomes that are NOT the "expected" all-0 or all-1 states
            expected_outcomes = {"0" * num_qubits, "1" * num_qubits}
            error_count = sum(v for k, v in counts.items() if k not in expected_outcomes)
            error_rate = round((error_count / total_shots) * 100, 2)

        records.append({
            "circuit_type": data["circuit_type"],
            "num_qubits": num_qubits,
            "run_number": data["run_number"],
            "backend": data["backend"],
            "shots": total_shots,
            "error_rate": error_rate,
            "raw_counts": json.dumps(counts)
        })

df = pd.DataFrame(records)

print(f"Master dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("Runs per circuit type:")
print(df["circuit_type"].value_counts())

print("\nAverage error rate per circuit type (excludes single-qubit circuits):")
print(df[df["error_rate"].notna()].groupby("circuit_type")["error_rate"].agg(["mean", "std", "min", "max"]).round(2))

df.to_csv("quantumbridge_data/master_dataset_v2.csv", index=False)
print("\nSaved to quantumbridge_data/master_dataset_v2.csv")
