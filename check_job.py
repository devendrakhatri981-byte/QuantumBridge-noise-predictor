from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token="PASTE_YOUR_ACTUAL_TOKEN_HERE",
    instance="open-instance"
)
job = service.job("PASTE_YOUR_ACTUAL_JOB_ID_HERE")
print(job.status())
