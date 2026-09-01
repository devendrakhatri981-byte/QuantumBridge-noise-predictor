# QuantumBridge: Teaching a Model to Know How Quantum Hardware Fails — and When It Doesn't Know

*An independent research project by an undergraduate BTech AI/ML student, Lachoo Memorial College, Jodhpur.*

## The problem

Quantum computers are noisy, and real hardware access is limited and metered. Generic noise-model simulators exist, but they aren't tuned to how a *specific* chip actually fails. QuantumBridge set out to learn a real chip's noise behavior well enough to predict circuit success offline — and, eventually, to have the model flag when its own prediction shouldn't be trusted.

## Phase 1: What real hardware actually tells you

The project opened with 165 real measurements on IBM hardware. A simple linear model — CNOT count plus chip identity — explained 74.7% of held-out error variance. A follow-up "forgiveness law" (how much of a gate's nominal error actually shows up in measured outcomes) looked great through several fits, then broke completely once tested with more than two data points: two points can't fail a power-law fit, so nothing had actually been tested. Widening the measurement surfaced a real confound — decoherence during gate execution being mistaken for gate error on short-coherence-time qubits — and a previously shipped calibration anchor turned out to be invalid. It was corrected.

## Phase 2: Closed-form models hit a wall

A physics-based emulator (v3/v4) modeled real chip topology, SWAP routing, and decoherence. It matched reference simulation within 0.4–3.2 points on most circuits — and missed one, a long cross-chip route, by 14 points. Root cause: certain circuit shapes collapse toward a 50% "coin flip" floor that no closed-form correction can predict, because the failure is compounding decoherence, not additive gate error. That motivated a switch from fitting formulas to learning the function directly.

## Phase 3: A graph neural network, built from scratch

A GNN (JAX + Optax, no PyTorch) represents each circuit as a graph of physical qubits and routed connections, trained on simulated outcomes against real calibration snapshots. Growing the training set to 3,000+ circuits was mostly infrastructure work. The harder problem was honesty: early cross-chip tests (train on one chip, evaluate cold on a second, never mixed) failed because the model was quietly using each chip's absolute error scale as a shortcut. Fixing that required per-chip relative normalization with no chip-identity signal at all — the change that made cross-chip generalization real instead of illusory.

## Phase 4: Topological diversity closes a gap

All large circuits so far used one topology — a hub-and-spoke "star." Adding a second, structurally different "chain" topology and folding it into the dataset (now 3,221 circuits total) measurably improved the hardest cross-chip direction's floor-collapse error, from 8.6 points down to 7.0.

## Phase 5: Teaching the model to know what it doesn't know

Rather than training a five-model ensemble for uncertainty, the project used Monte Carlo Dropout — 20 stochastic forward passes per prediction from a single trained model, active at inference time. A first validation attempt used an invalid mixed-chip split and gave a misleadingly weak signal; caught and rebuilt against the project's proper cross-chip protocol, the real result held: uncertainty widened 2.77x and 1.61x (by direction) between familiar and unfamiliar chip territory, and that widening tracked real prediction error — not just noise.

## Phase 6: Making it real

A fully static live demo now shows, for any of 16,002 possible qubit pairs, the closed-form prediction, the GNN's prediction with its uncertainty band, and the real result where available — plus a flag for when the model disagrees sharply with the formula while being confident, and an honest disclosure that cross-chip reliability is asymmetric rather than uniform. A companion 3D visualization renders the real 127-qubit chip with two actually-routed circuits (star and chain) drawn across it, so the topological difference is something you can see, not just read in a table. Building it surfaced and fixed two real bugs along the way, caught by direct visual inspection.

## Where things stand

Dataset: 3,221 routed circuits across two topologies and two chips. Same-chip GNN accuracy: MAE 1.09 (R²=0.977). Cross-chip: MAE 2.9–3.6 depending on direction (R²=0.86–0.90), tested honestly and reported asymmetrically rather than averaged into one number. Uncertainty quantification: validated in both directions, live in the public demo. All of it — including the parts that didn't work the first time — is documented in a 134-page, 59-entry public research log.

## What's next

Re-validating on live hardware when access returns, finishing chain-topology data on the second chip, adding a third chip architecture to stress-test generalization further, and checking whether an ensemble would agree with the MC-Dropout uncertainty estimates.

---

*QuantumBridge is an independent, ongoing undergraduate research project. The live demo, 3D visualization, and full research log are public. If you work on quantum computing, noise modeling, or ML applied to physical systems, get in touch.*
