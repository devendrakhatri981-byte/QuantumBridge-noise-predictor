"""
QuantumBridge — Entry 049a: submit the real-hardware validation batch.

WHY THIS EXISTS
---------------
Three independent AI reviews of the public demo converged on one critique:
this project has only ever been validated against Qiskit's Aer simulator
(fed by real IBM calibration snapshots), never against an actual queued
job on physical hardware. Every number in the demo -- v4.1, the GNN, and
the "real simulated result" -- all ultimately come from the same
simulator. This script closes that gap: it submits the 13 circuits
selected in entry049_realhw_candidates.json (a deliberate spread of
floor-collapse candidates, high-fidelity adjacent pairs, medium-distance
pairs, and one small GHZ circuit per chip) to REAL IBM Quantum hardware.

HOW TO RUN THIS
----------------
This must be run on YOUR OWN machine with YOUR OWN IBM Quantum account --
it cannot be run from here, since it needs your credentials and this
sandbox has no reason to ever see your API token.

1. Create a free account at https://quantum.ibm.com (or cloud.ibm.com IBM
   Quantum Platform) if you don't have one.
2. Copy your API token from the account dashboard.
3. Set it as an environment variable (never paste it directly into this
   file or into chat with anyone, including me):
       export IBM_QUANTUM_TOKEN="your_token_here"
4. Run: python3 entry049_submit_real_jobs.py
5. It prints job IDs and saves them to quantumbridge_data/entry049_job_ids.json.
   Real hardware jobs can take minutes to hours to complete depending on
   queue -- run entry049_collect_real_results.py later to fetch results.

This intentionally submits all 13 circuits in a SINGLE Sampler job per
backend (not 13 separate jobs) to use queue time efficiently -- IBM's
Runtime primitives accept a list (a "public circuit interface run") of
circuits (PUBs) per job.
"""

import json
import os

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

SHOTS = 4096
CANDIDATES_PATH = "quantumbridge_data/entry049_realhw_candidates.json"
JOB_IDS_PATH = "quantumbridge_data/entry049_job_ids.json"

# Real device names -- these are the actual physical chips the project's
# FakeKyiv/FakeSherbrooke snapshots are modeled on. Availability under the
# free/open plan can change; the script checks what you actually have
# access to and tells you if these exact names aren't available.
BACKEND_NAMES = {"kyiv": "ibm_kyiv", "sherbrooke": "ibm_sherbrooke"}


def build_circuit(chip_nq, cand):
    if cand["kind"] == "bell":
        a, b = cand["a"], cand["b"]
        qc = QuantumCircuit(chip_nq, 2)
        qc.h(a); qc.cx(a, b)
        qc.measure(a, 0); qc.measure(b, 1)
        return qc, [a, b]
    else:  # ghz (star topology, matches entry044/045's build_circuit convention)
        pairs = cand["logical_pairs"]
        qubits = cand["qubits"]
        qc = QuantumCircuit(chip_nq, len(qubits))
        qc.h(pairs[0][0])
        for a, b in pairs:
            qc.cx(a, b)
        for i, q in enumerate(qubits):
            qc.measure(q, i)
        return qc, qubits


def main():
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        print("ERROR: set the IBM_QUANTUM_TOKEN environment variable first "
              "(see the docstring at the top of this file). Not proceeding "
              "without it -- never hardcode your token into this file.")
        return

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token,
                                   instance="open-instance")
    available = {b.name for b in service.backends()}
    print(f"you have access to {len(available)} backends")

    candidates = json.load(open(CANDIDATES_PATH))
    by_chip = {}
    for c in candidates:
        by_chip.setdefault(c["chip"], []).append(c)

    job_ids = {}
    for chip, cands in by_chip.items():
        wanted = BACKEND_NAMES[chip]
        if wanted not in available:
            print(f"WARNING: {wanted} not available on your account -- "
                 f"skipping {len(cands)} {chip} circuits. Available backends: {sorted(available)}")
            continue
        backend = service.backend(wanted)
        nq = backend.num_qubits
        print(f"{chip}: submitting {len(cands)} circuits to {wanted} ({nq} qubits)")

        transpiled = []
        meta = []
        for cand in cands:
            qc, meas_qubits = build_circuit(nq, cand)
            t = transpile(qc, backend=backend, initial_layout=list(range(nq)),
                         optimization_level=3, seed_transpiler=1)
            transpiled.append(t)
            meta.append({**cand, "meas_qubits": meas_qubits})

        sampler = SamplerV2(mode=backend)
        job = sampler.run([(t,) for t in transpiled], shots=SHOTS)
        print(f"  submitted job {job.job_id()} ({len(transpiled)} circuits, {SHOTS} shots each)")
        job_ids[chip] = {"job_id": job.job_id(), "backend": wanted, "candidates": meta}

    json.dump(job_ids, open(JOB_IDS_PATH, "w"), indent=2)
    print(f"\nsaved job IDs -> {JOB_IDS_PATH}")
    print("Run entry049_collect_real_results.py later (jobs can take a while) to fetch results.")


if __name__ == "__main__":
    main()
