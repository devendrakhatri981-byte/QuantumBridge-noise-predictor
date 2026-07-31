"""
QuantumBridge Emulator — Full Validation

Tests the emulator against REAL measured data across every CNOT level
you've collected real hardware results for, not just one circuit.
"""

from qiskit import QuantumCircuit
from quantumbridge_emulator import emulate, predict_error_rate

# Real measured means from your research log (Entries 004-006)
REAL_DATA = {
    1: {"backend": "ibm_fez", "real_mean": 3.80},
    2: {"backend": "ibm_fez", "real_mean": 5.64},
    3: {"backend": "ibm_kingston", "real_mean": 4.94},
    4: {"backend": "ibm_kingston", "real_mean": 6.39},
}

def build_ghz(num_qubits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc

print("QuantumBridge Emulator — Full Validation Report")
print("=" * 65)
print(f"{'CNOTs':<8}{'Backend':<16}{'Real (%)':<12}{'Emulated (%)':<14}{'Diff':<8}")
print("-" * 65)

results = []
for cnots, info in REAL_DATA.items():
    num_qubits = cnots + 1
    circuit = build_ghz(num_qubits)
    predicted = predict_error_rate(circuit, backend=info["backend"])
    real = info["real_mean"]
    diff = abs(predicted - real)
    results.append({"cnots": cnots, "backend": info["backend"], "real": real, "predicted": predicted, "diff": diff})
    print(f"{cnots:<8}{info['backend']:<16}{real:<12}{predicted:<14.2f}{diff:<8.2f}")

print("-" * 65)
avg_diff = sum(r["diff"] for r in results) / len(results)
print(f"\nAverage deviation from real hardware: {avg_diff:.2f} percentage points")
print("(For reference: real hardware itself varies ~1-2 points run-to-run)")

# Save validation chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cnots_list = [r["cnots"] for r in results]
real_list = [r["real"] for r in results]
pred_list = [r["predicted"] for r in results]

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(cnots_list, real_list, marker="o", markersize=10, linewidth=2, color="#5C35A0", label="Real IBM hardware")
ax.plot(cnots_list, pred_list, marker="s", markersize=9, linewidth=2, linestyle="--", color="#00695C", label="QuantumBridge emulator")
ax.set_xlabel("CNOT gate count")
ax.set_ylabel("Error rate (%)")
ax.set_title("QuantumBridge Emulator vs Real IBM Quantum Hardware", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(cnots_list)
plt.tight_layout()
plt.savefig("quantumbridge_data/emulator_validation.png", dpi=150, facecolor="white")
print("\nValidation chart saved to quantumbridge_data/emulator_validation.png")
