# Entry 022 — v3's first honest validation: two failures pointing opposite ways

**Chip:** FakeKyiv (127 qubits, uniform `ecr`) · **Shots:** 4096 × 4 seeds · **Method:** MPS

**Scripts:** `diagnose_basis_mismatch.py`, `validate_v3_kyiv.py`
**Data:** `quantumbridge_data/entry022_kyiv_validation.json`

---

## Why the old validation was void

Two independent defects, both in the harness rather than the model.

**1. Basis mismatch.** `FakeCairoV2` is a mixed snapshot — 12 directed edges calibrated
as `cx`, 14 as `ecr`. `validate_v3_circuits.py` passed `basis_gates=backend.operation_names`,
handing the transpiler the union. It picked `cx` for the whole circuit and emitted it on
edges that only have `ecr` calibration. Qiskit's noise model keys errors by *(gate name,
qubit pair)*, so those gates matched nothing and ran with **no error applied at all**.

| Circuit | 2q gates | noisy | silent | % free |
|---|---|---|---|---|
| ghz_local_0_1_2 | 2 | 0 | 2 | **100%** |
| ghz_scattered_0_26_22 | 35 | 20 | 15 | 43% |
| bell_adjacent_24_25 | 1 | 1 | 0 | 0% |
| bell_scattered_0_26 | 34 | 13 | 21 | 62% |
| **Total** | **72** | 34 | **38** | **53%** |

**2. Calibration coverage.** Only 12 of Cairo's 28 topology edges carry real calibrated
errors — **42.9%**. The other 16 used v3's 1% fallback constant. Most of the "model" on
that chip was a guess.

Kyiv has neither problem: uniformly `ecr`, noise-model coverage on all 144 edges, and
139/144 (96.5%) real calibration. `validate_v3_kyiv.py` audits its own noise coverage on
every run and refuses to report numbers if any gate runs silent.

## Result

All 2q gates verified noisy. Zero silent.

| Circuit | hops | 2q gates | Aer ref | v3 | gap |
|---|---|---|---|---|---|
| bell_adjacent_77_78 | 1 | 1 | 96.07% | 99.25% | 3.18 |
| bell_near_77_82 | 5 | 13 | **48.74%** | 92.89% | **44.15** |
| bell_mid_77_100 | 6 | 16 | 83.69% | 90.27% | 6.58 |
| bell_far_0_126 | 26 | 76 | 69.90% | 60.51% | 9.39 |
| ghz_local_77_78_79 | 2 | 2 | 88.53% | 98.67% | 10.14 |

Mean absolute gap **14.69 points**. But the mean is misleading, because the errors point in
opposite directions.

*(`ghz_scattered_0_63_126` was killed mid-run — two 26-hop routes exceeded MPS memory. Not
included.)*

## The anomaly

**13 gates produce 48.74%. Seventy-six gates produce 69.90%.** Six times the gates, a
better outcome. Circuit depth does not explain this chip's behavior.

The 77→82 route passes through **qubit 80: T2 = 8.51 µs, T1 = 5.02 µs** — effectively dead
against a chip median in the hundreds. With a 560 ns `ecr`, that is ~6.6% dephasing per
gate, and the route crosses it six times.

Worse, edges (79,80) and (80,81) are among the five with **no calibration data**, so v3
applied its 1% fallback guess exactly where reality is catastrophic. Two failures stacked
on the same edges.

| Route | worst T2 on path | gates | Aer |
|---|---|---|---|
| 77→82 | **8.5 µs** (q80) | 13 | 48.74% |
| 77→100 | 32.7 µs | 16 | 83.69% |
| 0→126 | healthy throughout | 76 | 69.90% |

## Diagnosis

v3 models **two-qubit gate error only**. It has no decoherence term. Consequently:

- **Short routes through low-T2 qubits are catastrophically under-penalized** (44 points).
- **Long routes through healthy qubits are over-penalized** by SWAP compounding (9 points,
  v3 pessimistic).

This is Entry 021's finding resurfacing one level up. There, the domination factor
`D = gate_error / (gate_duration / T2_min)` was introduced as a *screening* criterion to
decide which edges were fit to measure. Entry 022 shows the same quantity belongs *inside
the model* as a cost term. Every edge on the 77→82 route through q80 has D ≈ 0.15 — the
regime Entry 021 identified as decoherence-dominated.

## A naive fix, and why it was rejected

Charging `gate_duration / T2_min` per gate alongside the gate-error term:

| Circuit | ref | v3 gate-only | +decoherence |
|---|---|---|---|
| bell_near_77_82 | 48.74% | 93.10% (44.36) | 58.83% (**10.09**) |
| bell_far_0_126 | 69.90% | 64.06% (5.84) | 30.67% (**39.23**) |
| **Mean gap** | | **14.10** | **13.85** |

It fixes the short bad route and destroys the long good one. Mean barely moves.

The reason is physical: in a SWAP chain the state *moves*. Each qubit holds it only
briefly. Charging every gate the full `duration / T2` of both its qubits double-counts
massively over 76 gates spread across 26 healthy qubits, while being roughly right over 6
gates parked on one dead qubit.

Decoherence must be **time-integrated along the carrier path** — sum, over each qubit the
state actually occupies, of (time spent there) / T2 of that qubit — not applied per gate to
both members of every pair. That is the v4 design.

## What this changes

1. **No v3 accuracy figure predates this entry.** Every circuit-level number from Entry 014
   onward was measured against a reference silencing half its noise. The forgiveness work
   (Entries 017–021) is unaffected — those used single-edge circuits with no routing, on
   properly covered edges.
2. **FakeCairoV2 is retired for validation.** Mixed basis and 42.9% calibration coverage.
   Kyiv or Sherbrooke only.
3. **v4 needs a decoherence term**, time-integrated along the route.
4. **The fallback constant is dangerous.** It fires on uncalibrated edges, which correlate
   with broken qubits — precisely where a wrong guess costs most. v4 should refuse to
   predict on uncalibrated edges rather than quietly assume 1%.
5. **`method="matrix_product_state"` is mandatory** for deep routed circuits. Aer's
   `automatic` returns `success=True`, `status=COMPLETED`, and then raises on
   `get_counts()`. A harness trusting it silently loses its hardest circuits.

## Next

- [ ] Implement time-integrated decoherence (v4) and re-validate
- [ ] Make v3 refuse uncalibrated edges instead of using the 1% fallback
- [ ] Route-quality awareness: v3's BFS minimizes hop count, ignoring T2 and edge error
- [ ] Re-run `ghz_scattered_0_63_126` with a memory-bounded MPS configuration
