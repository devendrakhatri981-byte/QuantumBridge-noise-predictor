# Entry 021 — Kyiv: the forgiveness-ratio law survives, the measurement protocol does not

**Chip:** `FakeKyiv` (127 qubits) · **Edges probed:** 6 · **Shots:** 4096 · **Seed:** 1729

**Scripts:** `kyiv_decay_curves.py`, `analyze_entry021.py`
**Data:** `quantumbridge_data/kyiv_decay_curves.json`, `quantumbridge_data/entry021_analysis.json`

---

## Why this entry exists

Entries 017 and 019 (Cairo) and Entry 020 (Sherbrooke) each measured exactly **two**
edges per chip and drew a power law through them. Two points define a power law
exactly, with zero residual — the fit could not fail, so it was never a test.

Those two chips produced an eye-catching coincidence: the high/low forgiveness ratio
was 0.6292 on Cairo and 0.6302 on Sherbrooke. That looks like a chip-invariant
constant. It isn't. The two chips were probed over different error spans (5.2x vs
8.8x), and under a genuine shared power law, a wider span must produce a *smaller*
high/low ratio. Matching ratios over different spans therefore imply *different*
exponents — which is exactly what the fits show: −0.280 vs −0.212.

Entry 021 probes six Kyiv edges spanning 0.31%–4.32% so the exponent is genuinely
fitted and the residuals become a real test.

## Result 1 — the six-point fit fails badly

```
ratio = 0.1200 * err^(-0.3303)     R² = 0.2956
```

Two edges are wildly off the curve:

| edge | raw error | measured ratio | fitted | residual |
|---|---|---|---|---|
| (119, 120) | 0.3113% | 0.4345 | 0.8076 | **−0.3731** |
| (28, 29) | 0.6013% | **1.5904** | 0.6497 | **+0.9406** |
| (77, 78) | 0.9860% | 0.5353 | 0.5518 | −0.0166 |
| (45, 46) | 1.6181% | 0.4308 | 0.4685 | −0.0377 |
| (66, 67) | 2.6181% | 0.3937 | 0.3997 | −0.0060 |
| (23, 24) | 4.3229% | 0.2927 | 0.3386 | −0.0460 |

A forgiveness ratio of **1.59** is not a small error — it is physically impossible
under the model. It says the pair decayed *faster* than its own gate error allows.
Something other than the two-qubit gate is driving that decay.

## Result 2 — the cause is decoherence, not gate error

The forgiveness ratio is defined as `measured_decay_constant / raw_gate_error`.
That definition assumes the decay is *caused* by gate error. But every CX also takes
finite time (562 ns on Kyiv), during which the qubits dephase. Define:

```
D = raw_gate_error / (gate_duration / T2_min)
```

`D > 1` means gate error dominates; `D < 1` means decoherence does.

| edge | raw error | T2_min | decoherence/gate | **D** | verdict |
|---|---|---|---|---|---|
| (119, 120) | 0.3113% | 202.7 µs | 0.277% | **1.12** | contaminated |
| (28, 29) | 0.6013% | **12.7 µs** | 4.430% | **0.14** | contaminated |
| (77, 78) | 0.9860% | 212.3 µs | 0.265% | 3.73 | usable |
| (45, 46) | 1.6181% | 57.2 µs | 0.983% | 1.65 | usable |
| (66, 67) | 2.6181% | 86.6 µs | 0.649% | 4.03 | usable |
| (23, 24) | 4.3229% | 65.1 µs | 0.863% | 5.01 | usable |

Qubit 28 has **T2 = 12.7 µs** against a chip median in the hundreds. Its decay is
~7x more dephasing than gate error. It was never measuring forgiveness.

Refitting on the four edges with `D ≥ 1.5`:

```
ratio = 0.0898 * err^(-0.3873)     R² = 0.9648
```

**R² goes from 0.296 to 0.965.** The power-law form was never the problem — the
edge-selection protocol was. Edges must be screened on `D`, not just on error
magnitude.

## Result 3 — Entry 019's anchor point is contaminated

Applying the same screen retroactively:

| chip | edge | raw error | T2_min | **D** | verdict |
|---|---|---|---|---|---|
| Cairo | (24, 25) | 0.5992% | 192.4 µs | 4.32 | usable |
| Cairo | **(19, 22)** | 3.1315% | **6.9 µs** | **0.31** | **contaminated** |
| Sherbrooke | (60, 61) | 0.3470% | 360.3 µs | 2.34 | usable |
| Sherbrooke | (66, 73) | 3.0667% | 170.0 µs | 9.78 | usable |

Cairo's high-error edge (19,22) has **T2 = 6.9 µs** and a 697 ns gate — 10.1%
decoherence per gate against 3.13% gate error. Entry 019's measured ratio of 0.334
at that edge is roughly three parts dephasing to one part gate error.

That number is one of the two points hardcoded in
`emulator_v3_routing.py :: variable_forgiveness_ratio()`. **The emulator's error
model is currently anchored on a broken qubit.**

## Result 4 — how much it matters

Old law (Cairo-anchored) vs new law (Kyiv, 4 clean edges):

| raw edge error | old ratio | new ratio | change |
|---|---|---|---|
| 0.300% | 0.6433 | 0.8518 | **+32.4%** |
| 0.500% | 0.5577 | 0.6989 | +25.3% |
| 1.000% | 0.4595 | 0.5343 | +16.3% |
| 2.000% | 0.3785 | 0.4085 | +7.9% |
| 3.000% | 0.3380 | 0.3491 | +3.3% |
| 4.500% | 0.3018 | 0.2984 | −1.1% |

The two laws agree where the contaminated point pinned them together (~3–4.5%) and
diverge sharply on low-error edges — which are the majority of every chip. On Kyiv,
75% of calibrated edges sit below 1.6%.

Circuit-level, on a 1.0% edge:

| CNOTs | old prediction | new prediction | gap |
|---|---|---|---|
| 1 | 99.54% | 99.47% | −0.07 pts |
| 10 | 95.50% | 94.78% | −0.72 pts |
| 20 | 91.20% | 89.84% | −1.36 pts |
| 40 | 83.18% | 80.71% | −2.46 pts |

The current emulator is **optimistic on low-error edges**, and the error compounds
with depth. At 40 CNOTs the correction is 2.5 points — comparable to the 1.87-point
validation gap that Entry 019 was celebrating.

## Robustness check

Ratios re-measured across three simulator seeds:

| edge | seeds 1729 / 42 / 20260804 | spread |
|---|---|---|
| (77, 78) | 0.5353, 0.5583, 0.5330 | 0.0253 |
| (23, 24) | 0.2927, 0.3076, 0.3143 | 0.0216 |

Seed spread is ~0.02. The two outlier residuals are 0.37 and 0.94 — 15x to 40x
larger. The outliers are real physics, not sampling noise.

## What this changes

1. **Edge screening is now part of the protocol.** Any edge used to fit a
   forgiveness ratio must satisfy `D = gate_error / (gate_duration / T2_min) ≥ 1.5`.
   Report `D` alongside every future measurement.
2. **Entry 019's Cairo anchor is retired.** It fails the screen it motivated.
3. **`variable_forgiveness_ratio()` needs re-anchoring** on the Kyiv 4-point fit
   (`0.0898 * err^-0.3873`, R² = 0.965) — the only fit in the project with enough
   points to have residuals at all.
4. **The exponent is not yet shown to be chip-invariant.** Kyiv's clean −0.387 still
   differs from Sherbrooke's 2-point −0.212. Sherbrooke needs the same six-edge
   treatment before any cross-chip claim is made.
5. **The "0.63 invariant" is withdrawn.** Kyiv gives 0.674 over a 13.9x span; the
   Cairo/Sherbrooke agreement was a coincidence between one contaminated point and
   one clean pair.

## Honest caveat

All of this is measured against Qiskit's `FakeKyiv` noise model, not live hardware.
The T2 = 12.7 µs on qubit 28 is a real recorded calibration value from a real device
snapshot, and the decoherence mechanism is real, but the *magnitude* of the
correction should be re-validated on hardware when IBM access is restored. The
protocol change (screen edges on `D`) holds regardless.

## Next

- [ ] Re-run the six-edge protocol on Sherbrooke and Cairo for a fair three-chip
      exponent comparison
- [ ] Re-anchor `variable_forgiveness_ratio()` and re-validate v3 against the Bell
      and GHZ circuits from Entry 019
- [ ] Check whether the residual exponent differences track chip *generation*
      (Cairo is an older Falcon-class device; Sherbrooke and Kyiv are Eagle-class)
