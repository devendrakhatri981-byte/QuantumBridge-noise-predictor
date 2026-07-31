import json
import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── Load every sample across the whole project ──────────────
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
}

records = []
for folder, info in CIRCUIT_INFO.items():
    for filepath in sorted(glob.glob(f"quantumbridge_data/{folder}/run_*.json")):
        with open(filepath) as f:
            data = json.load(f)
            counts = data["counts"]
            total_shots = data["shots"]
            num_qubits = data["num_qubits"]
            expected = {"0"*num_qubits, "1"*num_qubits}
            error_count = sum(v for k, v in counts.items() if k not in expected)
            error_rate = round((error_count/total_shots)*100, 2)
            records.append({
                "circuit_type": folder, "cnot_count": info["cnot"],
                "sq_gate_count": info["sq_gates"], "backend": data["backend"],
                "error_rate": error_rate, "timestamp": data["timestamp"]
            })

df = pd.DataFrame(records)
df["is_kingston"] = (df["backend"]=="ibm_kingston").astype(int)
print(f"TOTAL DATASET: {len(df)} real quantum hardware measurements\n")

# ── 1. Overall statistics ──────────────────────────────────
print("="*60)
print("1. OVERALL PROJECT STATISTICS")
print("="*60)
print(f"Circuit types tested: {df['circuit_type'].nunique()}")
print(f"Backends used: {df['backend'].unique().tolist()}")
print(f"Error rate range: {df['error_rate'].min()}% to {df['error_rate'].max()}%")
print(f"Overall mean error rate: {df['error_rate'].mean():.2f}%")
print(f"Overall std dev: {df['error_rate'].std():.2f}%")

# ── 2. Best model so far (Entry 007 style, 2-feature) ──────
print("\n" + "="*60)
print("2. VALIDATED MODEL — cnot_count + backend")
print("="*60)
X = df[["cnot_count", "is_kingston"]]
y = df["error_rate"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"error_rate = {model.intercept_:.3f} + ({model.coef_[0]:.3f} x cnot_count) + ({model.coef_[1]:.3f} x is_kingston)")
print(f"R2 = {r2:.3f}, MAE = {mae:.2f} points  (on {len(df)}-sample full dataset)")

# ── 3. Residual analysis — where does the model struggle? ──
print("\n" + "="*60)
print("3. RESIDUAL ANALYSIS")
print("="*60)
df["predicted"] = model.predict(X)
df["residual"] = df["error_rate"] - df["predicted"]
print("Mean absolute residual by circuit type:")
print(df.groupby("circuit_type")["residual"].apply(lambda x: x.abs().mean()).round(2).sort_values(ascending=False))

# ── 4. Export charts for GitHub README ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)

# Error rate by circuit type
ax = axes[0,0]
means = df.groupby("circuit_type")["error_rate"].mean().sort_values()
ax.barh(means.index, means.values, color="#5C35A0")
ax.set_xlabel("Mean error rate (%)")
ax.set_title("Error Rate by Circuit Type", fontweight="bold")

# CNOT scaling
ax = axes[0,1]
cnot_means = df.groupby("cnot_count")["error_rate"].mean()
ax.plot(cnot_means.index, cnot_means.values, marker="o", color="#00695C", linewidth=2, markersize=8)
ax.set_xlabel("CNOT count")
ax.set_ylabel("Mean error rate (%)")
ax.set_title("Error Scales with CNOT Count", fontweight="bold")
ax.grid(alpha=0.3)

# Predicted vs actual
ax = axes[1,0]
ax.scatter(df["error_rate"], df["predicted"], alpha=0.5, color="#5C35A0")
lims = [df["error_rate"].min()-0.5, df["error_rate"].max()+0.5]
ax.plot(lims, lims, "--", color="#BF6000")
ax.set_xlabel("Actual error rate (%)")
ax.set_ylabel("Predicted error rate (%)")
ax.set_title(f"Model Fit (R2={r2:.2f})", fontweight="bold")

# Backend comparison
ax = axes[1,1]
df.boxplot(column="error_rate", by="backend", ax=ax)
ax.set_title("Error Rate by Backend", fontweight="bold")
ax.set_xlabel("")
plt.suptitle("")

plt.tight_layout()
plt.savefig("quantumbridge_data/project_summary_charts.png", dpi=150, facecolor="white")
print("\nCharts saved to quantumbridge_data/project_summary_charts.png")

df.to_csv("quantumbridge_data/full_project_dataset.csv", index=False)
print("Full dataset saved to quantumbridge_data/full_project_dataset.csv")
