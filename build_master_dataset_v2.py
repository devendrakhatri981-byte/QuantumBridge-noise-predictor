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
        circuit_type = data["circuit_type"]

        # Each circuit type has a DIFFERENT definition of "expected" outcomes
        if circuit_type == "single_superposition":
            # 1 qubit, H gate: both 0 and 1 are expected ~50/50, no "error" concept
            error_rate = None
        elif circuit_type == "independent_hadamards":
            # 2 unentangled qubits: ALL FOUR outcomes are expected ~25% each
            # "error" here would mean deviation from that even 25/25/25/25 split
            expected_frac = 0.25
            deviations = [abs((counts.get(k, 0)/total_shots) - expected_frac) for k in ["00","01","10","11"]]
            error_rate = round(sum(deviations) * 100 / 2, 2)  # total variation distance style metric
        else:
            # bell_state, ghz_state: entangled, only all-0 or all-1 expected
            expected_outcomes = {"0" * num_qubits, "1" * num_qubits}
            error_count = sum(v for k, v in counts.items() if k not in expected_outcomes)
            error_rate = round((error_count / total_shots) * 100, 2)

        records.append({
            "circuit_type": circuit_type,
            "num_qubits": num_qubits,
            "run_number": data["run_number"],
            "backend": data["backend"],
            "shots": total_shots,
            "error_rate": error_rate,
        })

df = pd.DataFrame(records)

print(f"Master dataset: {df.shape[0]} rows\n")
print("Average error rate per circuit type (corrected logic):")
print(df[df["error_rate"].notna()].groupby("circuit_type")["error_rate"].agg(["mean", "std", "min", "max"]).round(2))

print("\nEntangled circuits only — noise scaling with qubit count:")
entangled = df[df["circuit_type"].isin(["bell_state", "ghz_state"])]
print(entangled.groupby("num_qubits")["error_rate"].agg(["mean", "std"]).round(2))

df.to_csv("quantumbridge_data/master_dataset_final.csv", index=False)
print("\nSaved to quantumbridge_data/master_dataset_final.csv")
