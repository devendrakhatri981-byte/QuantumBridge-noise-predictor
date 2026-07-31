from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import json
import os
import time
from datetime import datetime

RUNS_PER_CIRCUIT = 15
TOKEN = "whudFfVHj_V1izGAIXKVulrRhub49tQc-nrr8prWqVSA"

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=TOKEN,
    instance="open-instance"
)

backend = service.least_busy(operational=True, simulator=False)
print(f"Connected. Running on: {backend.name}\n")

# ── GHZ state builder — works for any qubit count ──────────
def build_ghz(num_qubits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc

circuit_types = {
    "ghz_state_4q": {"builder": lambda: build_ghz(4), "qubits": 4, "cnots": 3},
    "ghz_state_5q": {"builder": lambda: build_ghz(5), "qubits": 5, "cnots": 4},
}

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
sampler = Sampler(backend)

total_jobs = len(circuit_types) * RUNS_PER_CIRCUIT
completed = 0
failed = 0

print(f"Plan: {len(circuit_types)} circuit types x {RUNS_PER_CIRCUIT} runs = {total_jobs} total jobs\n")

for circuit_name, config in circuit_types.items():
    folder = f"quantumbridge_data/{circuit_name}"
    os.makedirs(folder, exist_ok=True)

    circuit = config["builder"]()
    isa_circuit = pm.run(circuit)
    num_qubits = config["qubits"]
    cnot_count = config["cnots"]

    print(f"=== {circuit_name} ({num_qubits} qubits, {cnot_count} CNOTs) ===")

    for run_num in range(1, RUNS_PER_CIRCUIT + 1):
        try:
            job = sampler.run([isa_circuit], shots=1024)
            result = job.result()
            pub_result = result[0]
            data = pub_result.data
            key = list(data.__dict__.keys())[0]
            bitarray = getattr(data, key)
            counts = bitarray.get_counts()

            entry = {
                "circuit_type": circuit_name,
                "num_qubits": num_qubits,
                "cnot_count": cnot_count,
                "run_number": run_num,
                "timestamp": datetime.now().isoformat(),
                "backend": backend.name,
                "job_id": job.job_id(),
                "shots": 1024,
                "counts": counts
            }

            filepath = f"{folder}/run_{run_num:02d}.json"
            with open(filepath, "w") as f:
                json.dump(entry, f, indent=2)

            completed += 1
            print(f"  Run {run_num}/{RUNS_PER_CIRCUIT} — saved — progress: {completed}/{total_jobs} total")

        except Exception as e:
            failed += 1
            print(f"  Run {run_num}/{RUNS_PER_CIRCUIT} — FAILED: {e}")
            print(f"  Waiting 10s before continuing...")
            time.sleep(10)
            continue

    print()

print(f"\n{'='*50}")
print(f"COLLECTION COMPLETE")
print(f"Successful runs: {completed}/{total_jobs}")
print(f"Failed runs: {failed}/{total_jobs}")
print(f"Data saved in: quantumbridge_data/<circuit_type>/")
print(f"{'='*50}")
