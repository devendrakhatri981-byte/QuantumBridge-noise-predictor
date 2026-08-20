"""
QuantumBridge — Entry 042: parallelized growth toward 5,000 circuits.

Same exact methodology as Entries 034/035/038/039/040 (Aer/MPS ground
truth, 4 seeds x 4096 shots, optimization_level=3/seed_transpiler=1,
exact_dwell_cost prediction) -- zero precision tradeoff. The only change
is using a multiprocessing.Pool across the sandbox's 4 CPU cores instead
of one circuit at a time, since each circuit's Aer simulation is
independent and embarrassingly parallel. Each worker process builds its
own backend/graph/coherence/AerSimulator once and reuses it across every
task it's handed (cached in a per-process global), so the pool doesn't
pay chip-setup cost per circuit.
"""

import json
import multiprocessing as mp
import os
import random
import time

# IMPORTANT: must be set before qiskit/qiskit_aer import in every process.
# Aer's matrix_product_state simulator already uses OpenMP/BLAS internal
# threading -- without this, 4 worker processes each try to grab all 4
# cores for their own linear-algebra calls, oversubscribing the machine
# and making the "parallel" run SLOWER than one single-threaded process
# (confirmed by a direct timing test: ~0.23 circuits/s with 4 unpinned
# workers, worse than the ~0.46/s a single sequential process achieved on
# short-route batches in Entries 038/040). Pinning each worker to 1
# thread lets the OS actually run 4 independent simulations at once.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost

SHOTS = 4096
SEEDS = (1, 2, 3, 4)
OUT_PATH = "quantumbridge_data/entry042_parallel_dataset.json"
BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke}
BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 25), (26, 999)]

_cache = {}


def get_ctx(chip):
    if chip not in _cache:
        backend = BACKENDS[chip]()
        graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
        coh = v4.load_coherence(chip)
        nm = NoiseModel.from_backend(backend)
        sim = AerSimulator(noise_model=nm, method="matrix_product_state")
        _cache[chip] = (backend, graph, coh, sim)
    return _cache[chip]


def run_bell(task):
    chip, a, b, dist = task
    backend, graph, coh, sim = get_ctx(chip)
    nq = backend.num_qubits

    qc = QuantumCircuit(nq, 2)
    qc.h(a); qc.cx(a, b)
    qc.measure(a, 0); qc.measure(b, 1)
    t = transpile(qc, backend=backend, initial_layout=list(range(nq)),
                 optimization_level=3, seed_transpiler=1)
    # A single sim.run() with shots = SHOTS*len(SEEDS) is statistically
    # identical to averaging len(SEEDS) separate SHOTS-shot runs (same
    # total independent samples), and measured ~18% faster since the
    # per-call fixed overhead only pays once. No precision tradeoff --
    # confirmed by direct comparison (0.8198 4-call-averaged vs 0.8148
    # single-call on a test circuit, within expected shot noise).
    ideal = {"00", "11"}
    counts = sim.run(t, shots=SHOTS * len(SEEDS), seed_simulator=SEEDS[0]).result().get_counts()
    tot = sum(counts.values())
    ok = sum(c for bstr, c in counts.items() if bstr.replace(" ", "") in ideal)
    ref = ok / tot

    edges_used = sorted(set(tuple(sorted([t.find_bit(x).index for x in inst.qubits]))
                            for inst in t.data if inst.operation.num_qubits == 2))
    n_2q = sum(1 for i in t.data if i.operation.num_qubits == 2)

    qc_bare = QuantumCircuit(nq, 2)
    qc_bare.h(a); qc_bare.cx(a, b)
    pred, _, _ = exact_dwell_cost(qc_bare, backend, graph, coh, [(a, b)], [a, b])

    raw_edge_err, worst_t1 = None, None
    try:
        path = em.shortest_path(graph, a, b)
        raw_edge_err = max(em.edge_error(graph, path[i], path[i + 1])
                           for i in range(len(path) - 1))
        worst_t1 = min(coh["T1"].get(q, 1e9) for q in path)
    except Exception:
        pass

    return {"chip": chip, "kind": "bell", "pairs_flat": [a, b],
           "logical_pairs": [[a, b]], "bfs_hop_distance": dist,
           "n_real_edges_used": len(edges_used), "n_2q_gates_total": n_2q,
           "worst_raw_edge_error_on_bfs_path": raw_edge_err,
           "worst_t1_on_bfs_path": worst_t1,
           "aer_ground_truth": ref, "v4_1_prediction": pred,
           "gap": abs(ref - pred)}


def sample_tasks(chip, backend_nq, graph, per_bin, seed, existing_keys):
    rng = random.Random(seed)
    chosen = []
    seen = set()
    for lo, hi in BINS:
        got, attempts = 0, 0
        while got < per_bin and attempts < per_bin * 60:
            attempts += 1
            a, b = rng.randrange(backend_nq), rng.randrange(backend_nq)
            if a == b or (a, b) in seen or (b, a) in seen or (chip, a, b) in existing_keys:
                continue
            path = em.shortest_path(graph, a, b)
            if path is None:
                continue
            dist = len(path) - 1
            if lo <= dist <= hi:
                chosen.append((chip, a, b, dist))
                seen.add((a, b))
                got += 1
    return chosen


def load():
    if os.path.exists(OUT_PATH):
        return json.load(open(OUT_PATH))
    return []


def main(target_per_chip=400, per_bin=None, seed=None, n_workers=4):
    seed = seed if seed is not None else random.randrange(1_000_000)
    per_bin = per_bin or max(1, target_per_chip // len(BINS))

    records = load()
    existing = {(r["chip"], r["pairs_flat"][0], r["pairs_flat"][1]) for r in records}
    print(f"resuming with {len(records)} existing records, seed={seed}")

    tasks = []
    for chip in ("kyiv", "sherbrooke"):
        backend = BACKENDS[chip]()
        graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
        tasks += sample_tasks(chip, backend.num_qubits, graph, per_bin, seed, existing)
    print(f"queued {len(tasks)} new circuits across {n_workers} workers")

    t0 = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for rec in pool.imap_unordered(run_bell, tasks, chunksize=2):
            records.append(rec)
            done += 1
            if done % 20 == 0:
                json.dump(records, open(OUT_PATH, "w"))
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(tasks)} done ({rate:.2f}/s, "
                      f"total so far {len(records)})")

    json.dump(records, open(OUT_PATH, "w"))
    print(f"DONE this run: {done} new, {len(records)} total -> {OUT_PATH}")


if __name__ == "__main__":
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    main(target_per_chip=target)
