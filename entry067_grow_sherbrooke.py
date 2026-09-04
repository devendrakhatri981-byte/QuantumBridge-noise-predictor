"""
QuantumBridge — Entry 067: bell-pair dataset growth for Sherbrooke, to bring
it up to parity with Kyiv (~1,510) and Brisbane (~1,621) -- Sherbrooke
currently has only 723 bell-pair circuits (778 total incl. GHZ), the
thinnest of the three chips.

Identical methodology to entry061_grow_brisbane.py (Aer/MPS ground truth,
4 seeds x 4096 shots folded into one call, optimization_level=3,
seed_transpiler=1, exact_dwell_cost prediction, same six hop-distance bins,
atomic checkpointed writes, shuffled task order to avoid starving long-hop
bins) -- only the chip and existing-pair exclusion set changed. Existing
pairs already in entry043_combined_dataset.json are excluded so we don't
waste compute re-simulating circuits we already have ground truth for.
"""

import json
import multiprocessing as mp
import os
import random
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost

SHOTS = 4096
SEEDS = (1, 2, 3, 4)
OUT_PATH = "quantumbridge_data/entry067_sherbrooke_bell_dataset.json"
EXISTING_PATH = "quantumbridge_data/entry043_combined_dataset.json"
BACKENDS = {"sherbrooke": FakeSherbrooke}
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
            if a == b or (a, b) in seen or (b, a) in seen or (chip, a, b) in existing_keys or (chip, b, a) in existing_keys:
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


def load_existing_keys():
    keys = set()
    if os.path.exists(EXISTING_PATH):
        for r in json.load(open(EXISTING_PATH)):
            if r["chip"] == "sherbrooke" and r["kind"] == "bell":
                a, b = r["pairs_flat"]
                keys.add(("sherbrooke", a, b))
    return keys


def _atomic_save(records, path):
    # See entry061_grow_brisbane.py -- a SIGKILL mid-write to a non-atomic
    # save corrupted a checkpoint AND destroyed the prior good save. Always
    # write-then-replace.
    tmp = path + ".tmp"
    json.dump(records, open(tmp, "w"))
    os.replace(tmp, path)


def main(target=800, per_bin=None, seed=67, n_workers=4):
    per_bin = per_bin or max(1, target // len(BINS))

    records = load()
    existing = load_existing_keys()
    existing |= {("sherbrooke", r["pairs_flat"][0], r["pairs_flat"][1]) for r in records}
    print(f"resuming with {len(records)} new records this run, "
          f"{len(existing)} total existing sherbrooke bell pairs excluded, seed={seed}")

    tasks = []
    for chip in ("sherbrooke",):
        backend = BACKENDS[chip]()
        graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
        tasks += sample_tasks(chip, backend.num_qubits, graph, per_bin, seed, existing)
    random.Random(seed + 1).shuffle(tasks)
    print(f"queued {len(tasks)} new circuits across {n_workers} workers (shuffled)")

    t0 = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for rec in pool.imap_unordered(run_bell, tasks, chunksize=2):
            records.append(rec)
            done += 1
            if done % 2 == 0:
                _atomic_save(records, OUT_PATH)
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(tasks)} done ({rate:.2f}/s, "
                      f"total so far {len(records)})")

    _atomic_save(records, OUT_PATH)
    print(f"DONE this run: {done} new, {len(records)} total -> {OUT_PATH}")


if __name__ == "__main__":
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    main(target=target)
