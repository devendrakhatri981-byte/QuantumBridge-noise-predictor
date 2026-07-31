"""
QuantumBridge — Explore Offline Real-Chip Calibration Snapshots

Qiskit ships real (historical) calibration data from actual IBM chips,
bundled offline. No account, no quota, no internet needed.
"""

from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

provider = FakeProviderForBackendV2()
backends = provider.backends()

print(f"Found {len(backends)} offline fake backends (real historical chip data)\n")
for b in backends[:15]:
    print(f"  {b.name}  —  {b.num_qubits} qubits")

print(f"\n... and {max(0, len(backends)-15)} more" if len(backends) > 15 else "")
