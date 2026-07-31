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

# Explicitly request ibm_fez to match Entry 004/005 baseline — avoids the
# backend confound discovered in Entry 006
try:
    backend = service.backend("ibm_fez")
    if not backend.status().operational:
        raise Exception("ibm_fez not operational")
    print(f"Connected to requested backend: {backend.name}\n")
except Exception as e:
    print(f"Could not get ibm_fez ({e}), falling back to least busy")
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Connected. Running on: {backend.name}\n")

# ── Gate-stress circuit builder ──────────────────────────
def build_gate_stress(extra_x_pairs):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    for _ in range(extra_x_pairs):
        qc.x(0)
        qc.x(0)   # X-X cancels to identity — ideal outcome unchanged
    qc.measure([0, 1], [0, 1])
    return qc

circuit_types = {
    "gate_stress_4":  {"builder": lambda: build_gate_stress(2), "single_qubit_gates": 4},
    "gate_stress_8":  {"builder": lambda: build_gate_stress(4), "single_qubit_gates": 8},
    "gate_stress_12": {"builder": lambda: build_gate_stress(6), "single_qubit_gates": 12},
}

pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
# NOTE: optimization_level=0 is critical here — a higher level would
# recognize X-X as identity and OPTIMIZE IT AWAY, defeating the experiment

sampler = Sampler(backend)

total_jobs = len(circuit_types) * RUNS_PER_CIRCUIT
completed = 0
failed = 0

print(f"Plan: {len(circuit_types)} circuit types x {RUNS_PER_CIRCUIT} runs = {total_jobs} total jobs")
print("optimization_level=0 used to prevent X-X pairs being optimized away\n")

for circuit_name, config in circuit_types.items():
    folder = f"quantumbridge_data/{circuit_name}"
    os.makedirs(folder, exist_ok=True)

    circuit = config["builder"]()
    isa_circuit = pm.run(circuit)
    single_qubit_gates = config["single_qubit_gates"]

    print(f"=== {circuit_name} ({single_qubit_gates} extra single-qubit gates, 1 CNOT) ===")

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
                "num_qubits": 2,
                "cnot_count": 1,
                "single_qubit_gates": single_qubit_gates,
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
            time.sleep(10)
            continue

    print()

print(f"\n{'='*50}")
print(f"COLLECTION COMPLETE")
print(f"Successful runs: {completed}/{total_jobs}")
print(f"Failed runs: {failed}/{total_jobs}")
print(f"{'='*50}")
