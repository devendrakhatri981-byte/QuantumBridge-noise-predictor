import json
import pandas as pd
import glob

records = []
for filepath in sorted(glob.glob("quantumbridge_data/run_*.json")):
    with open(filepath) as f:
        data = json.load(f)
        counts = data["counts"]
        total_shots = data["shots"]
        
        error_count = counts.get('01', 0) + counts.get('10', 0)
        error_rate = (error_count / total_shots) * 100
        
        records.append({
            "run_number": data["run_number"],
            "backend": data["backend"],
            "shots": total_shots,
            "count_00": counts.get('00', 0),
            "count_11": counts.get('11', 0),
            "count_01": counts.get('01', 0),
            "count_10": counts.get('10', 0),
            "error_rate": round(error_rate, 2)
        })

df = pd.DataFrame(records)

print("Your dataset as a table:")
print(df.to_string(index=False))
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

df.to_csv("quantumbridge_data/master_dataset.csv", index=False)
print("\nSaved to quantumbridge_data/master_dataset.csv")
