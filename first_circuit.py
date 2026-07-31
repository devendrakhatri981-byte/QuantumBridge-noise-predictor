from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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
print("Circuit transpiled successfully")

sampler = Sampler(backend)
job = sampler.run([isa_circuit], shots=1024)
print(f"Job ID: {job.job_id()}")
print("Waiting for results (1-10 minutes)...")

result = job.result()
pub_result = result[0]

# Find the correct data key
print("Data keys available:", pub_result.data.__dict__.keys())
# Get the bitarray and convert to counts
data = pub_result.data
key = list(data.__dict__.keys())[0]
bitarray = getattr(data, key)
counts = bitarray.get_counts()

print(f"\nReal hardware results on {backend.name}:")
print(counts)
print("\nSimulator was: {'00': ~512, '11': ~512, '01': 0, '10': 0}")
print("\nDifference = NOISE. This is what QuantumBridge learns.")
