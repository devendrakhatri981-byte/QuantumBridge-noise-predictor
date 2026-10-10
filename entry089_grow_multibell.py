"""
QuantumBridge — Entry 089: first step toward arbitrary-circuit support.
Parallel multi-bell-pair circuits: N independent bell pairs (H+CX) on
DISJOINT qubits, run simultaneously in one circuit. Unlike every prior
template (bell/GHZ/star/chain), this has more than one "thing that
matters" -- multiple entangling gates happening at once, which can
interact via crosstalk / shared control electronics / simultaneous
readout contention. It's the smallest possible step past "one dominant
gate" while keeping ground truth cheap and success unambiguous: success
= ALL N pairs measured correlated in the same shot.

This script ONLY generates circuits and computes ground truth (real Aer
simulation) plus two comparison baselines -- it does NOT touch the graph
feature pipeline yet. Purpose: confirm the ground truth looks sane and
that there's an actual crosstalk signal worth modeling before investing
in the feature-schema rework the plan calls for.

Two v4.1-style baselines are computed per circuit, since exact_dwell_cost
was built for CHAINED pairs (GHZ/star), not independent parallel ones,
and explicitly says so does not extend cleanly to that case:
  - product_of_marginals: call exact_dwell_cost separately on EACH pair
    in isolation (as if it were the only pair in the circuit) and
    multiply the results. This is what you'd expect the joint success
    to be if the pairs behaved completely independently (no crosstalk).
  - naive_chained: call exact_dwell_cost with all N pairs passed together,
    the way GHZ/star circuits are scored today. Included to see what the
    existing formula does with this input even though it's the wrong
    model for parallel execution.
A real gap between aer_ground_truth and product_of_marginals is the
signal this template exists to capture.
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
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke, FakeBrisbane, FakeOsaka, FakeQuebec
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import emulator_v3_routing as em
import emulator_v4 as v4
from exact_dwell_routing import exact_dwell_cost

SHOTS = 4096
SEEDS = (1, 2, 3, 4)
OUT_PATH = "quantumbridge_data/entry089_multibell_sanity.json"
BACKENDS = {"kyiv": FakeKyiv, "sherbrooke": FakeSherbrooke, "brisbane": FakeBrisbane,
            "osaka": FakeOsaka, "quebec": FakeQuebec}
N_PAIRS_CHOICES = (2, 3)

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


def run_multibell(task):
    chip, pairs, hops = task
    backend, graph, coh, sim = get_ctx(chip)
    nq = backend.num_qubits
    n = len(pairs)

    qc = QuantumCircuit(nq, 2 * n)
    for i, (a, b) in enumerate(pairs):
        qc.h(a); qc.cx(a, b)
    for i, (a, b) in enumerate(pairs):
        qc.measure(a, 2 * i); qc.measure(b, 2 * i + 1)

    t = transpile(qc, backend=backend, initial_layout=list(range(nq)),
                 optimization_level=3, seed_transpiler=1)
    counts = sim.run(t, shots=SHOTS * len(SEEDS), seed_simulator=SEEDS[0]).result().get_counts()
    tot = sum(counts.values())

    # bitstring is little-endian over the 2*n classical bits, MSB-first as
    # printed by qiskit (creg[2n-1] ... creg[0]); pair i's two bits are at
    # classical indices 2i, 2i+1, so in the printed string (reversed) they
    # sit at positions -(2i+1), -(2i+2) from the end.
    def pair_ok(bstr, i):
        clean = bstr.replace(" ", "")
        bit_a = clean[-(2 * i + 1)]
        bit_b = clean[-(2 * i + 2)]
        return bit_a == bit_b

    joint_ok = sum(c for bstr, c in counts.items() if all(pair_ok(bstr, i) for i in range(n)))
    joint_ref = joint_ok / tot

    marginal_refs = []
    for i in range(n):
        ok_i = sum(c for bstr, c in counts.items() if pair_ok(bstr, i))
        marginal_refs.append(ok_i / tot)

    # baseline 1: product of each pair's OWN isolated v4.1 prediction
    product_of_marginals = 1.0
    per_pair_v41 = []
    for (a, b) in pairs:
        qc_solo = QuantumCircuit(nq, 2)
        qc_solo.h(a); qc_solo.cx(a, b)
        pred_solo, _, _ = exact_dwell_cost(qc_solo, backend, graph, coh, [(a, b)], [a, b])
        per_pair_v41.append(pred_solo)
        product_of_marginals *= pred_solo

    # baseline 2: naive call with all pairs passed together (chain-style,
    # known-wrong model for this case, kept for comparison only)
    qc_bare = QuantumCircuit(nq, 2 * n)
    for (a, b) in pairs:
        qc_bare.h(a); qc_bare.cx(a, b)
    measured = [q for pr in pairs for q in pr]
    try:
        naive_chained, _, _ = exact_dwell_cost(qc_bare, backend, graph, coh, pairs, measured)
    except Exception as e:
        naive_chained = None

    return {"chip": chip, "kind": "multibell", "n_pairs": n,
           "pairs_flat": [q for pr in pairs for q in pr],
           "logical_pairs": [list(p) for p in pairs],
           "bfs_hop_distances": hops,
           "aer_ground_truth": joint_ref,
           "aer_marginal_per_pair": marginal_refs,
           "product_of_marginals_v41": product_of_marginals,
           "per_pair_v41": per_pair_v41,
           "naive_chained_v41": naive_chained,
           "gap_vs_product": abs(joint_ref - product_of_marginals)}


def sample_task(chip, backend_nq, graph, seed_rng):
    n = seed_rng.choice(N_PAIRS_CHOICES)
    pairs, hops = [], []
    used = set()
    attempts = 0
    while len(pairs) < n and attempts < 200:
        attempts += 1
        a, b = seed_rng.randrange(backend_nq), seed_rng.randrange(backend_nq)
        if a == b or a in used or b in used:
            continue
        path = em.shortest_path(graph, a, b)
        if path is None:
            continue
        dist = len(path) - 1
        # Multi-pair circuits don't stay cheap at long hops the way single
        # bell-pairs do: N simultaneous entangled clusters push MPS
        # simulation cost up sharply with route length (measured: hop~20
        # took 36s per circuit vs <1s at hop~1, vs single-pair growth where
        # even hop=26 stays fast). Capped at 8 to keep per-circuit cost in
        # the few-seconds range so growth throughput stays practical.
        if dist > 8:
            continue
        pairs.append((a, b))
        hops.append(dist)
        used.add(a); used.add(b)
    return (chip, pairs, hops) if len(pairs) == n else None


def _atomic_save(records, path):
    tmp = path + ".tmp"
    json.dump(records, open(tmp, "w"))
    os.replace(tmp, path)


def main(n_circuits=40, seed=89, n_workers=4, chip="kyiv"):
    records = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
    chip_records = [r for r in records if r["chip"] == chip]
    print(f"resuming with {len(records)} existing records total, {len(chip_records)} for {chip}")

    # advance the RNG state past what earlier runs already consumed, keyed
    # off how many records exist for THIS chip -- otherwise re-running this
    # script with the same seed regenerates the exact same circuits (bug
    # hit during the first sanity pass: runs 1 and 2 produced identical
    # tasks), and a per-chip seed avoids kyiv/sherbrooke/brisbane runs
    # colliding on the same random draws.
    rng = random.Random(seed + hash(chip) % 10000 + len(chip_records))
    existing_sigs = {(r["chip"], tuple(sorted(tuple(p) for p in r["logical_pairs"])))
                     for r in records}

    tasks = []
    backend = BACKENDS[chip]()
    graph = em.build_connectivity_graph(em.load_calibration(chip), chip)
    attempts = 0
    while len(tasks) < n_circuits and attempts < n_circuits * 100:
        attempts += 1
        t = sample_task(chip, backend.num_qubits, graph, rng)
        if t is None:
            continue
        sig = (chip, tuple(sorted(tuple(p) for p in t[1])))
        if sig in existing_sigs:
            continue
        existing_sigs.add(sig)
        tasks.append(t)
    print(f"queued {len(tasks)} multi-bell circuits ({sum(1 for t in tasks if len(t[1])==2)} with N=2, "
         f"{sum(1 for t in tasks if len(t[1])==3)} with N=3)")

    t0 = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for rec in pool.imap_unordered(run_multibell, tasks, chunksize=1):
            records.append(rec)
            done += 1
            if done % 5 == 0 or done == len(tasks):
                _atomic_save(records, OUT_PATH)
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(tasks)} done ({rate:.2f}/s)")

    _atomic_save(records, OUT_PATH)
    print(f"DONE: {len(records)} total -> {OUT_PATH}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    chip = sys.argv[2] if len(sys.argv) > 2 else "kyiv"
    main(n_circuits=n, chip=chip)
