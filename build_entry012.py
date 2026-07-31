import json
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

CIRCUIT_INFO = {
    "bell_state":        {"cnot": 1, "sq_gates": 1},
    "ghz_state":         {"cnot": 2, "sq_gates": 1},
    "ghz_state_4q":      {"cnot": 3, "sq_gates": 1},
    "ghz_state_5q":      {"cnot": 4, "sq_gates": 1},
    "gate_stress_0_v2":  {"cnot": 1, "sq_gates": 1},
    "gate_stress_4_v2":  {"cnot": 1, "sq_gates": 5},
    "gate_stress_8_v2":  {"cnot": 1, "sq_gates": 9},
    "gate_stress_12_v2": {"cnot": 1, "sq_gates": 13},
    "cnot_sweep_1_v2":   {"cnot": 1, "sq_gates": 1},
    "cnot_sweep_2_v2":   {"cnot": 2, "sq_gates": 1},
    "cnot_sweep_3_v2":   {"cnot": 3, "sq_gates": 1},
    # cnot_sweep_4_v2 excluded — only 2 incomplete runs, not usable
}

records = []
for circuit_folder, info in CIRCUIT_INFO.items():
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
                "cnot_count": info["cnot"],
                "sq_gate_count": info["sq_gates"],
                "backend": data["backend"],
                "error_rate": error_rate,
            })

df = pd.DataFrame(records)
df["is_kingston"] = (df["backend"] == "ibm_kingston").astype(int)

print(f"Entry 012 dataset: {df.shape[0]} samples\n")

# Check: does cnot_count now vary independently of backend?
print("CNOT count vs backend crosstab (looking for BOTH chips at each cnot level):")
print(pd.crosstab(df["cnot_count"], df["backend"]))

X = df[["cnot_count", "sq_gate_count", "is_kingston"]]
y = df["error_rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(X_train)}, testing on {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

cnot_w, sq_w, kingston_w = model.coef_
intercept = model.intercept_

print(f"\nModel learned:")
print(f"  error_rate = {intercept:.3f} + ({cnot_w:.3f} x cnot_count) + ({sq_w:.3f} x sq_gate_count) + ({kingston_w:.3f} x is_kingston)")

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\nTest performance:")
print(f"  R2: {r2:.3f}")
print(f"  MAE: {mae:.2f} percentage points")

print(f"\n--- Full comparison ---")
print(f"  Entry 007: R2=0.747, MAE=0.63")
print(f"  Entry 009: R2=0.629, MAE=0.70")
print(f"  Entry 011: R2=0.553, MAE=0.92")
print(f"  Entry 012: R2={r2:.3f}, MAE={mae:.2f}")

df.to_csv("quantumbridge_data/entry012_unified.csv", index=False)
print(f"\nSaved to quantumbridge_data/entry012_unified.csv")
