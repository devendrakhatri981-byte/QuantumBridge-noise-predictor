import json
import pandas as pd
import glob

# CNOT count per circuit type (from how we built each circuit)
CNOT_LOOKUP = {
    "bell_state": 1,
    "ghz_state": 2,
    "ghz_state_4q": 3,
    "ghz_state_5q": 4,
}

records = []

for circuit_folder in CNOT_LOOKUP.keys():
    for filepath in sorted(glob.glob(f"quantumbridge_data/{circuit_folder}/run_*.json")):
        with open(filepath) as f:
            data = json.load(f)
            counts = data["counts"]
            total_shots = data["shots"]
            num_qubits = data["num_qubits"]

            expected_outcomes = {"0" * num_qubits, "1" * num_qubits}
            error_count = sum(v for k, v in counts.items() if k not in expected_outcomes)
            error_rate = round((error_count / total_shots) * 100, 2)

            records.append({
                "circuit_type": circuit_folder,
                "num_qubits": num_qubits,
                "cnot_count": CNOT_LOOKUP[circuit_folder],
                "backend": data["backend"],
                "error_rate": error_rate,
            })

df = pd.DataFrame(records)
print(f"Unified dataset: {df.shape[0]} samples\n")
print("Breakdown:")
print(df.groupby(["circuit_type", "cnot_count", "backend"]).size())

df.to_csv("quantumbridge_data/entry007_unified.csv", index=False)
print(f"\nSaved to quantumbridge_data/entry007_unified.csv")
