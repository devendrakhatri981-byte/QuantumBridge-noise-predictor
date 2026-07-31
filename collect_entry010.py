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

# Let IBM pick whichever chip is actually available right now —
# ibm_fez appears stuck in maintenance, so forcing it caused a hang overnight
backend = service.least_busy(operational=True, simulator=False)
print(f"Connected. Running on: {backend.name}")
print("(All 4 conditions will use THIS SAME chip in one session — that's what matters)\n")

def build_gate_stress(extra_x_pairs):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    for _ in range(extra_x_pairs):
        qc.x(0)
        qc.x(0)
    qc.measure([0, 1], [0, 1])
    return qc

circuit_types = {
    "gate_stress_0_v2":  {"builder": lambda: build_gate_stress(0), "single_qubit_gates": 1},
    "gate_stress_4_v2":  {"builder": lambda: build_gate_stress(2), "single_qubit_gates": 5},
    "gate_stress_8_v2":  {"builder": lambda: build_gate_stress(4), "single_qubit_gates": 9},
    "gate_stress_12_v2": {"builder": lambda: build_gate_stress(6), "single_qubit_gates": 13},
}

pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
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
    sq_gates = config["single_qubit_gates"]

    print(f"=== {circuit_name} ({sq_gates} single-qubit gates, 1 CNOT) ===")

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
                "single_qubit_gates": sq_gates,
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
print(f"Backend used throughout: {backend.name}")
print(f"{'='*50}")
