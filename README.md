# QuantumBridge

**Learning how real quantum hardware fails, so it can be simulated offline.**

An independent research project building a noise emulator for IBM quantum chips — starting
from 165 real hardware measurements, and documenting every wrong turn along the way.

---

## The Problem

Real quantum computers are noisy. Access to them is limited, metered, and requires cloud
credentials most students and small research teams don't have. Simulators exist, but they
simulate *ideal* qubits — not the noisy, imperfect ones that exist in the world.

**QuantumBridge asks:** can we learn a real chip's noise behavior well enough to predict it,
without running every new circuit on real hardware?

## Where the Project Is Now

The current emulator (**v3**) takes any Qiskit circuit and a target chip and returns a
predicted success probability, entirely offline. It models the chip's real coupling map,
per-edge calibrated error rates, BFS-routed SWAP costs for non-adjacent qubit pairs, and an
empirically fitted correction for how much of a gate's nominal error actually shows up in
measured outcomes.

Validated against Aer noise-model simulation on FakeCairoV2, averaged over 8 seeds:

| Circuit | Reference | v3 predicted | Gap |
|---|---|---|---|
| `ghz_local_0_1_2` | 96.87% | 98.71% | 1.85 pts |
| `ghz_scattered_0_26_22` | 80.38% | 79.95% | 0.43 pts |
| `bell_adjacent_24_25` | 96.22% | 99.39% | 3.16 pts |
| `bell_scattered_0_26` | 95.76% | 81.75% | **14.01 pts** |

**Mean gap 1.81 points on three of four circuits — and a known 14-point failure on the
fourth.** That last row is the honest headline. `bell_scattered_0_26` routes across the chip,
and v3's SWAP-cost compounding badly over-penalizes it. It is the single largest error in the
model and the current top priority. It is reported here rather than dropped from the average,
because an emulator with one catastrophic failure mode is not an emulator with a 1.81-point
average.

Reproduce with `python ab_test_forgiveness_law.py`.

## What the Research Found

**Gate error is structured, not random** (Entries 004–011, real hardware). Across 165
measurements on `ibm_fez` and `ibm_kingston`, error scales measurably with circuit
composition: each CNOT costs ~1.6–1.8 percentage points, each single-qubit gate ~0.22 points —
roughly 7–8x cheaper. A two-feature linear model on CNOT count and chip identity reaches
R² = 0.747 on held-out hardware data.

```
error_rate = 3.003 + (1.618 × cnot_count) + (−2.311 × is_kingston)
```

**Chips don't have universal noise constants** (Entry 020). The same measurement on
FakeSherbrooke returned forgiveness ratios ~26% above FakeCairoV2 at matched error magnitudes.
Any single hardcoded noise curve is a per-chip approximation, not a law.

**Nominal gate error over-predicts observed error** (Entries 017–019). When success is scored
against a multi-outcome ideal set, part of each gate's error is absorbed rather than observed.
The size of that absorption depends on the edge's error magnitude — high-error edges are
forgiven less — and follows a power law.

**But that power law was measured wrong for four entries** (Entry 021). Entries 017–020 each
fitted the curve through exactly two points per chip. Two points define a power law exactly,
with zero residual, so the fit could never fail and never tested anything. Widening the
measurement to six edges on FakeKyiv produced R² = 0.296 and one forgiveness ratio of **1.59** —
physically impossible under the model.

The cause was a confound the two-point method could not detect. Every CX gate occupies finite
time, during which qubits dephase. Where T2 is short relative to gate duration, the measured
decay is driven by decoherence, not gate error. Screening edges on

```
D = gate_error / (gate_duration / T2_min)     require D ≥ 1.5
```

and refitting the four qualifying edges restores **R² = 0.9648**. Applying the same screen
retroactively showed that Entry 019's high-error anchor — a qubit pair with **T2 = 6.9 µs** —
failed it, and that anchor was hardcoded into the shipped emulator. It has been replaced.

## Honest Status

- Entries 001–012 used **real IBM hardware**. Entries 013–021 use Qiskit fake providers, which
  carry real recorded calibration snapshots but are not live devices. IBM Quantum Platform
  access lapsed with a trial account on 4 Aug 2026; the Entry 021 correction should be
  re-validated on hardware when access is restored.
- The forgiveness law currently ships fitted on **one chip** (Kyiv, 4 edges, 0.99%–4.32% error
  range). Entry 020 already established this will carry a systematic offset on other chips.
  Below 0.99% it extrapolates.
- `bell_scattered_0_26` remains unexplained and is the dominant source of model error.
- The v1 emulator numbers previously headlined in this README (0.70-point average deviation)
  came from a narrower test set and are superseded by the v3 figures above.

## Repository Structure

```
QuantumBridge/
├── emulator_v3_routing.py          # current emulator: topology, SWAP routing, forgiveness law
├── validate_v3_circuits.py         # circuit-level validation harness
├── ab_test_forgiveness_law.py      # seeded A/B comparison of forgiveness laws
├── kyiv_decay_curves.py            # Entry 021: six-edge decay-curve protocol
├── analyze_entry021.py             # Entry 021: decoherence screen and cross-chip comparison
├── setup_*.py                      # per-chip topology and calibration export
├── quantumbridge_data/             # measurements, calibrations, fitted models (46 files)
│   └── forgiveness_law.json        # the fitted law, with its provenance and known limits
├── docs/
│   ├── research_log.pdf            # the full log, Entries 001–021
│   └── entry021_findings.md
└── data/full_project_dataset.csv   # 165 real hardware measurements
```

## Research Log

The [research log](docs/research_log.pdf) is the primary artifact of this project — 21 entries
across 70 pages, covering every experiment, every confound, and every correction. Each entry
records what was done, why, the theory, the results, what it means, and what it triggered next.

It is not a highlight reel. Entry 009 documents a confound that dropped the model's R². Entry
015 records a fix that made things worse. Entry 021 retracts a finding celebrated in Entry 020
and invalidates an anchor point used by the shipped emulator. That is what the log is for.

## Getting Started

```bash
pip install -r requirements.txt

python validate_v3_circuits.py       # run the emulator against a noise-model reference
python ab_test_forgiveness_law.py    # seeded comparison of the old and new forgiveness laws
python kyiv_decay_curves.py          # re-measure the six-edge protocol (slow, ~54 simulations)
python analyze_entry021.py           # decoherence screen across all three chips
```

Everything above runs fully offline against Qiskit fake providers. No IBM Quantum account is
required. Python 3.11 recommended.

## Roadmap

- [ ] Diagnose `bell_scattered_0_26` — the 14-point routing failure
- [ ] Run the six-edge D-screened protocol on Sherbrooke and Cairo for a fair three-chip
      exponent comparison
- [ ] Turn the protocol into `calibrate_chip(backend)` — one routine any new device runs once
- [ ] Re-validate the Entry 021 correction on real hardware
- [ ] Scale the hardware dataset toward 500+ samples with full factorial coverage
- [ ] Package as an installable offline quantum noise emulator

## Tech Stack

Qiskit · Qiskit Aer · Qiskit IBM Runtime · scikit-learn · pandas / numpy · matplotlib

## Author

Built by an undergraduate BTech AI/ML student at Lachoo Memorial College, Jodhpur, as an
independent research project — from first quantum circuit to a working predictive model, with
the mistakes left in.

---

*Research-oriented and open for collaboration. If you work on quantum computing, noise
modeling, or ML applied to physical systems, get in touch.*
