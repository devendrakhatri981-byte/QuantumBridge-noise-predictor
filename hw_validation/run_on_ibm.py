"""
Hardware-validation kit, step 2: run the manifest circuits.
  python3 hw_validation/run_on_ibm.py MANIFEST --dry-run          # local noisy-sim smoke test
  python3 hw_validation/run_on_ibm.py MANIFEST --backend NAME     # real IBM hardware (needs account)
Real runs need: pip install qiskit-ibm-runtime; QiskitRuntimeService.save_account(...) done once.
IMPORTANT: predictions are for the *fake-backend snapshot* named in the manifest. They are only a fair test
on the physical device that snapshot came from (same qubit layout); on a different device the physical
qubit indices/errors differ, so treat the run as an out-of-distribution test and say so.
Circuits use the identity layout (logical q == physical q), as in the training simulations.
"""
import argparse, json, sys
sys.path.insert(0, ".")
from qiskit import QuantumCircuit, transpile

ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("--backend")
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
man = json.load(open(a.manifest)); shots = man["shots"]

if a.dry_run:
    import api.main as M
    import entry089_grow_multibell as g89
    backend, _, _, sim = g89.get_ctx(man["chip_model"])   # noisy MPS simulator (127q-safe)
else:
    from qiskit_ibm_runtime import QiskitRuntimeService
    backend = QiskitRuntimeService().backend(a.backend)
    if backend.num_qubits != man["num_qubits"]:
        sys.exit(f"{a.backend} has {backend.num_qubits} qubits, manifest expects {man['num_qubits']}")


def build(pairs):
    qc = QuantumCircuit(backend.num_qubits, 2 * len(pairs))
    for (x, y) in pairs:
        qc.h(x); qc.cx(x, y)
    for i, (x, y) in enumerate(pairs):
        qc.measure(x, 2 * i); qc.measure(y, 2 * i + 1)
    return qc


circs = man["circuits"][: a.limit or None]
tcs = [transpile(build(c["pairs"]), backend=backend, initial_layout=list(range(backend.num_qubits)),
                 optimization_level=3, seed_transpiler=1) for c in circs]
if a.dry_run:
    res = sim.run(tcs, shots=shots, seed_simulator=11).result()
    counts = [res.get_counts(i) for i in range(len(tcs))]
    tag = "dryrun"
else:
    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run(tcs, shots=shots)
    print("job id:", job.job_id(), "(results saved when it completes)", flush=True)
    r = job.result(); counts = [x.data.c.get_counts() if hasattr(x.data, "c") else
                                list(x.data.values())[0].get_counts() for x in r]
    tag = a.backend
out = {"manifest": a.manifest, "backend": tag, "shots": shots,
       "results": {c["id"]: cnt for c, cnt in zip(circs, counts)}}
fn = a.manifest.replace("manifest_", f"counts_{tag}_")
json.dump(out, open(fn, "w")); print("saved", fn)
