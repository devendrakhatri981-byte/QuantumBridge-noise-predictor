# QuantumBridge

**Learning how real quantum hardware fails, so it can be predicted offline — with the model
telling you when it isn't sure.**

An independent research project building a noise emulator for IBM quantum chips — starting
from 165 real hardware measurements, growing into a 3,221-circuit graph neural network with
calibrated uncertainty, and documenting every wrong turn along the way.

**[Try the live demo](https://devendrakhatri981-byte.github.io/QuantumBridge-noise-predictor/quantumbridge_live_demo.html)** ·
**[See the 3D chip visualization](https://devendrakhatri981-byte.github.io/QuantumBridge-noise-predictor/quantumbridge_3d_star_vs_chain.html)** ·
**[Read the research log](docs/research_log.pdf)** (134 pages, 59 entries)

---

## The Problem

Real quantum computers are noisy. Access to them is limited, metered, and requires cloud
credentials most students and small research teams don't have. Simulators exist, but they
simulate *ideal* qubits — not the noisy, imperfect ones that exist in the world.

**QuantumBridge asks:** can we learn a real chip's noise behavior well enough to predict it,
without running every new circuit on real hardware — and can the model be honest about how
confident it is?

## Where the Project Is Now

The project has two generations of predictor, both live in the demo:

**v4.1 — closed-form emulator.** Takes any Qiskit circuit and a target chip and returns a
predicted success probability from the chip's real coupling map, per-edge calibrated error
rates, and BFS-routed SWAP costs. Fast, interpretable, physics-first. Its blind spot: certain
circuits collapse to a near-50% "coin flip" floor that the closed-form formula doesn't see
coming, even on short routes.

**GNN — learned predictor with uncertainty.** A graph neural network (JAX + Optax, built from
scratch, no PyTorch) that represents each circuit as the real physical route it takes across
the chip and is trained directly against simulated hardware outcomes. It now:

- Predicts on circuits up to 90 physical qubits / 130 routing edges wide
- Reports a calibrated **uncertainty band** alongside every prediction (Monte Carlo Dropout,
  20 stochastic passes) — bands widen 1.6x–2.8x on unfamiliar chip territory, and that
  widening measurably tracks real error, not just noise
- Generalizes across chips it wasn't trained on (Kyiv ⇄ Sherbrooke), tested honestly: trained
  on one chip's circuits only, evaluated cold on the other's, never a mixed random split
- Was trained on **3,221 routed circuits** spanning two structurally different large-circuit
  topologies (hub-and-spoke "star" entanglement and sequential "chain" entanglement), which
  measurably closed a generalization gap that topological diversity alone was responsible for

Current same-chip performance: MAE 1.09 points, R² = 0.977. Cross-chip (the harder,
honest test): MAE 2.9–3.6 points depending on direction, R² 0.86–0.90 — one direction is
consistently easier than the other, and the demo says so rather than hiding it.

## What the Research Found

**Gate error is structured, not random** (Entries 004–011, real hardware). Across 165
measurements on `ibm_fez` and `ibm_kingston`, error scales measurably with circuit
composition: each CNOT costs ~1.6–1.8 percentage points, each single-qubit gate ~0.22 points —
roughly 7–8x cheaper.

**Chips don't have universal noise constants** (Entry 020). The same measurement returned
forgiveness ratios ~26% apart across chips at matched error magnitudes. Any single hardcoded
noise curve is a per-chip approximation, not a law.

**A power law was measured wrong for four entries in a row** (Entry 021). Two-point fits can't
fail — they have zero residual by construction. Widening to six edges surfaced a confound
(decoherence during gate time, not gate error, driving the signal on short-T2 edges) and a
retraction of a previously shipped calibration anchor.

**Closed-form models have a structural blind spot** (Entries 036–037). Certain circuit shapes
collapse toward a 50% floor that no amount of per-edge error tuning predicts, because the
failure mode isn't additive gate error — it's compounding decoherence across long routes. This
motivated the move to a learned model.

**A learned model closes that gap, but needs to prove it generalizes** (Entries 044–056).
Naive per-chip training overfits to one chip's calibration scale; per-chip *relative*
normalization (no raw error magnitudes, no chip-identity shortcut) was required before
cross-chip transfer became honest rather than illusory. Floor-collapse cases needed explicit
loss up-weighting (5x) to stop being systematically underpredicted.

**Uncertainty can be cheap and still be real** (Entry 057). Rather than training five separate
models for an ensemble, MC-Dropout gets a calibrated confidence signal from a single trained
model at inference time. Validated in both cross-chip directions against genuinely held-out
data — the widening isn't cosmetic, it tracks where the model is actually more wrong.

## Honest Status

- Entries 001–012 used **real IBM hardware**. Entries 013–059 use Qiskit fake providers, which
  carry real recorded calibration snapshots but are not live devices. IBM Quantum Platform
  access lapsed with a trial account on 4 Aug 2026.
- Cross-chip generalization is real but **asymmetric** — Sherbrooke→Kyiv transfers noticeably
  better than Kyiv→Sherbrooke. The demo discloses this on every result rather than presenting
  one blended reliability number.
- Chain-topology circuit generation only succeeded on Kyiv; Sherbrooke chain generation stalled
  repeatedly in the sandbox and was abandoned rather than forced. Logged honestly in Entry 058.
- `bell_scattered_0_26`-style long-route failures were the original motivation for the GNN and
  remain the class of circuit where the closed-form v4.1 model is least trustworthy — which is
  exactly where the GNN's uncertainty band tends to widen.

## Repository Structure

```
QuantumBridge/
├── quantumbridge_live_demo.html        # client-side demo: v4.1 vs GNN vs real, with uncertainty
├── quantumbridge_3d_star_vs_chain.html # 3D real-chip visualization, star vs chain routing
├── emulator_v3_routing.py              # closed-form emulator: topology, SWAP routing, forgiveness law
├── emulator_v4.py                      # v4.1: time-integrated decoherence closed-form model
├── entry057_mc_dropout.py              # MC-Dropout uncertainty quantification, cross-chip validated
├── entry058_chain_circuits.py          # chain-topology large-circuit generator
├── entry059_extract_routes.py          # real physical-qubit route extraction for the 3D viz
├── entry0NN_*.py                       # per-entry dataset growth / training / evaluation scripts
├── quantumbridge_data/                 # measurements, calibrations, fitted models, datasets
├── docs/
│   └── research_log.pdf                # the full log, Entries 001–059, 134 pages
└── data/full_project_dataset.csv       # 165 real hardware measurements
```

## Research Log

The [research log](docs/research_log.pdf) is the primary artifact of this project — 59 entries
across 134 pages, covering every experiment, every confound, and every correction. Each entry
records what was done, why, the theory, the results, what it means, and what it triggered next.

It is not a highlight reel. Entry 021 retracts a finding celebrated the entry before and
invalidates a calibration anchor already shipped in the emulator. Entry 058 documents a
generation run that stalled and was scoped down rather than forced to completion. That is what
the log is for.

## Try It Yourself

No install needed — both are static pages:

- **[Live prediction demo](https://devendrakhatri981-byte.github.io/QuantumBridge-noise-predictor/quantumbridge_live_demo.html)** —
  pick any two qubits on Kyiv or Sherbrooke, see the closed-form prediction, the GNN's
  prediction with its uncertainty band, and (where available) the real simulated result,
  side by side.
- **[3D chip visualization](https://devendrakhatri981-byte.github.io/QuantumBridge-noise-predictor/quantumbridge_3d_star_vs_chain.html)** —
  drag to rotate the real 127-qubit Kyiv coupling map with two actually-routed circuits
  highlighted on top of it.

## Getting Started (local / research code)

```bash
pip install -r requirements.txt

python validate_v3_circuits.py       # run the closed-form emulator against a noise-model reference
python ab_test_forgiveness_law.py    # seeded comparison of forgiveness laws
python entry057_mc_dropout.py train_cc kyiv_to_sherbrooke   # retrain + validate uncertainty, one direction
```

Everything above runs fully offline against Qiskit fake providers. No IBM Quantum account is
required. Python 3.11 recommended.

## Roadmap

- [ ] Re-validate the Entry 021 correction on real hardware when access is restored
- [ ] Generate chain-topology circuits on Sherbrooke (stalled in Entry 058, not yet resolved)
- [ ] Explore ensemble methods as a cross-check against MC-Dropout's uncertainty estimates
- [ ] Add a third chip architecture to stress-test generalization beyond Kyiv ⇄ Sherbrooke
- [ ] Scale the hardware dataset toward 500+ real-hardware samples with full factorial coverage
- [ ] Package as an installable offline quantum noise emulator

## Tech Stack

Qiskit · Qiskit Aer · Qiskit IBM Runtime · JAX / Optax · Three.js · scikit-learn ·
pandas / numpy · matplotlib

## Author

Built by an undergraduate BTech AI/ML student at Lachoo Memorial College, Jodhpur, as an
independent research project — from first quantum circuit to a working predictive model with
calibrated uncertainty, with the mistakes left in.

---

*Research-oriented and open for collaboration. If you work on quantum computing, noise
modeling, or ML applied to physical systems, get in touch.*
