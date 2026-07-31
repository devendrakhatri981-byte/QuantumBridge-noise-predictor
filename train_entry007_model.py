import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("quantumbridge_data/entry007_unified.csv")

# Convert backend (text) into a number the model can use: 0 or 1
# This is called "one-hot encoding" — a standard ML preprocessing step
df["is_kingston"] = (df["backend"] == "ibm_kingston").astype(int)

print("Data with encoded backend:")
print(df[["circuit_type", "cnot_count", "backend", "is_kingston", "error_rate"]].to_string(index=False))

# Two features now: cnot_count AND is_kingston
X = df[["cnot_count", "is_kingston"]]
y = df["error_rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples")

model = LinearRegression()
model.fit(X_train, y_train)

cnot_coef = model.coef_[0]
kingston_coef = model.coef_[1]
intercept = model.intercept_

print(f"\nModel learned:")
print(f"  error_rate = {intercept:.3f} + ({cnot_coef:.3f} x cnot_count) + ({kingston_coef:.3f} x is_kingston)")
print(f"\nInterpretation:")
print(f"  Base noise (ibm_fez, 0 CNOTs): {intercept:.2f}%")
print(f"  Cost per additional CNOT gate: {cnot_coef:.2f} percentage points")
print(f"  ibm_kingston adjustment vs ibm_fez: {kingston_coef:+.2f} percentage points")

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"\nTest set performance:")
print(f"  R² score: {r2:.3f}  (1.0 = perfect, 0 = no better than guessing average)")
print(f"  Mean Absolute Error: {mae:.2f} percentage points")

print(f"\nPredictions vs Reality on test set:")
for cnot, kingston, actual, pred in zip(X_test["cnot_count"], X_test["is_kingston"], y_test, predictions):
    chip = "kingston" if kingston == 1 else "fez"
    print(f"  {cnot} CNOTs, {chip}: actual={actual:.2f}%, predicted={pred:.2f}%, error={abs(actual-pred):.2f}")
