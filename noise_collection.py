from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import json
import os
from datetime import datetime

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token="whudFfVHj_V1izGAIXKVulrRhub49tQc-nrr8prWqVSA",
    instance="open-instance"
)

backend = service.least_busy(operational=True, simulator=False)
print(f"Running on: {backend.name}")

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(circuit)

os.makedirs("quantumbridge_data", exist_ok=True)
sampler = Sampler(backend)

for run_num in range(1, 21):
    job = sampler.run([isa_circuit], shots=1024)
    result = job.result()
    pub_result = result[0]
    data = pub_result.data
    key = list(data.__dict__.keys())[0]
    bitarray = getattr(data, key)
    counts = bitarray.get_counts()

    entry = {
        "run_number": run_num,
        "timestamp": datetime.now().isoformat(),
        "backend": backend.name,
        "job_id": job.job_id(),
        "shots": 1024,
        "counts": counts
    }

    filepath = f"quantumbridge_data/run_{run_num:02d}.json"
    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2)

    error_rate = (counts.get('01', 0) + counts.get('10', 0)) / 1024 * 100
    print(f"Run {run_num}/20 — {backend.name} — error rate: {error_rate:.2f}% — saved to {filepath}")

print("\nAll 20 runs complete. Dataset saved in quantumbridge_data/")
