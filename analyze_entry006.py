import json
import pandas as pd
import glob

records = []

for filepath in sorted(glob.glob("quantumbridge_data/ghz_state_*/run_*.json")):
    with open(filepath) as f:
        data = json.load(f)
        counts = data["counts"]
        total_shots = data["shots"]
        num_qubits = data["num_qubits"]

        expected_outcomes = {"0" * num_qubits, "1" * num_qubits}
        error_count = sum(v for k, v in counts.items() if k not in expected_outcomes)
        error_rate = round((error_count / total_shots) * 100, 2)

        records.append({
            "circuit_type": data["circuit_type"],
            "num_qubits": num_qubits,
            "cnot_count": data["cnot_count"],
            "backend": data["backend"],
            "error_rate": error_rate,
        })

df = pd.DataFrame(records)

print("Backend used per circuit type:")
print(df.groupby("circuit_type")["backend"].unique())

print("\nError rate summary (Entry 006 — new data only):")
print(df.groupby(["circuit_type", "num_qubits", "cnot_count"])["error_rate"].agg(["mean", "std", "min", "max"]).round(2))

df.to_csv("quantumbridge_data/entry006_dataset.csv", index=False)
print("\nSaved to quantumbridge_data/entry006_dataset.csv")

print("\n--- Hypothesis check (predicted ~1.7% per CNOT, from Entry 005) ---")
print("Entry 005 baseline: 0 CNOT=2.16%, 1 CNOT=3.80%, 2 CNOT=5.64% (all on ibm_fez)")
print("If pattern held on the SAME chip: 3 CNOT≈7.3%, 4 CNOT≈9.0%")
print("But new data is on a DIFFERENT chip (ibm_kingston) — direct comparison is not valid without caveats")
