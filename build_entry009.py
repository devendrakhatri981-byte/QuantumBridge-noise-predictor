import json
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Consistent gate counts per circuit type (including the baseline H gate every circuit has)
CIRCUIT_INFO = {
    "bell_state":      {"cnot": 1, "sq_gates": 1},   # H + CNOT
    "ghz_state":       {"cnot": 2, "sq_gates": 1},   # H + 2 CNOT
    "ghz_state_4q":    {"cnot": 3, "sq_gates": 1},   # H + 3 CNOT
    "ghz_state_5q":    {"cnot": 4, "sq_gates": 1},   # H + 4 CNOT
    "gate_stress_4":   {"cnot": 1, "sq_gates": 5},   # H + 4 extra X + CNOT
    "gate_stress_8":   {"cnot": 1, "sq_gates": 9},   # H + 8 extra X + CNOT
    "gate_stress_12":  {"cnot": 1, "sq_gates": 13},  # H + 12 extra X + CNOT
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

print(f"Unified Entry 009 dataset: {df.shape[0]} samples\n")
print(df.groupby(["circuit_type", "cnot_count", "sq_gate_count", "backend"]).size())

X = df[["cnot_count", "sq_gate_count", "is_kingston"]]
y = df["error_rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(X_train)}, testing on {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

cnot_w, sq_w, kingston_w = model.coef_
intercept = model.intercept_

print(f"\nThree-feature model learned:")
print(f"  error_rate = {intercept:.3f} + ({cnot_w:.3f} x cnot_count) + ({sq_w:.3f} x sq_gate_count) + ({kingston_w:.3f} x is_kingston)")

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\nTest performance:")
print(f"  R2: {r2:.3f}")
print(f"  MAE: {mae:.2f} percentage points")
print(f"\nCompare to Entry 007 two-feature model: R2=0.747, MAE=0.63")

df.to_csv("quantumbridge_data/entry009_unified.csv", index=False)
print(f"\nSaved to quantumbridge_data/entry009_unified.csv")
