import pandas as pd
import numpy as np

df = pd.read_csv("quantumbridge_data/master_dataset.csv")

print("Data:")
print(df[["run_number", "count_01", "count_10"]].to_string(index=False))

# Pearson correlation coefficient
correlation = df["count_01"].corr(df["count_10"])
print(f"\nCorrelation between count_01 and count_10: {correlation:.3f}")

if correlation > 0.5:
    print("STRONG POSITIVE — both qubits' errors rise and fall together")
    print("This suggests a SHARED noise source (chip-wide drift, temperature, calibration)")
elif correlation > 0.2:
    print("WEAK POSITIVE — some shared tendency, but not strongly linked")
elif correlation > -0.2:
    print("NO CLEAR RELATIONSHIP — the two qubits appear to error independently")
else:
    print("NEGATIVE — when one qubit errors more, the other errors less (unusual)")

# Also check: does total error (count_01 + count_10) correlate with anything obvious?
df["total_errors"] = df["count_01"] + df["count_10"]
print(f"\nAverage errors from qubit path 01: {df['count_01'].mean():.1f}")
print(f"Average errors from qubit path 10: {df['count_10'].mean():.1f}")
print(f"Ratio (10-errors / 01-errors): {df['count_10'].mean() / df['count_01'].mean():.2f}x")
