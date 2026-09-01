# QuantumBridge: Teaching a Model to Know How Quantum Hardware Fails — and When It Doesn't Know

*An independent research project by an undergraduate BTech AI/ML student, Lachoo Memorial College, Jodhpur.*

## The problem

Quantum computers are noisy. Every real gate you run on physical hardware degrades your result a little — sometimes a lot — and the amount depends on which specific qubits you used, how far apart they sit on the chip, how long your circuit takes to execute, and dozens of other physical details that don't show up if you only simulate an *ideal*, noiseless quantum computer.

If you want to know in advance whether a circuit you're about to design is going to survive real hardware, you have two options. Run it on real hardware — slow, metered, and increasingly out of reach for students and small teams without institutional access. Or use a noise-model simulator that approximates hardware behavior — fast, but only as good as the model behind it, and most available noise models are generic rather than tuned to how a *specific* chip actually fails.

QuantumBridge started as an attempt to close that gap: learn how a real chip fails, from real calibration data, well enough to predict success probability for any circuit, entirely offline. It grew into something with a second, less obvious goal — building a model that doesn't just predict, but tells you when its prediction shouldn't be trusted.

This is the story of both, including the parts that didn't work the first time.

## Phase 1: What does real hardware actually tell you? (Entries 001–021)

The project opened with real IBM hardware time — 165 measurements collected across `ibm_fez` and `ibm_kingston` before that access lapsed. The first finding was reassuring: gate error isn't noise, it's structure. A simple two-feature linear model — CNOT count, and which chip you're on — explained 74.7% of the variance in held-out error measurements:

```
error_rate = 3.003 + (1.618 × cnot_count) + (−2.311 × is_kingston)
```

Each CNOT cost roughly 1.6–1.8 percentage points of error; single-qubit gates were seven to eight times cheaper. That was Entries 004 through 011.

Then came the first real correction. A "forgiveness law" — a power-law correction describing how much of a gate's *nominal* calibrated error rate actually shows up in a measured, multi-outcome success score — had been fit and re-fit across Entries 017 through 020, and it looked great. Too great. Every fit used exactly two data points per chip, and two points define a power law exactly, with zero residual, every time. The fits had never actually been capable of failing, so they'd never tested anything.

Widening the measurement to six edges on FakeKyiv broke the illusion immediately: R² collapsed to 0.296, and one computed forgiveness ratio came out at 1.59 — a value the underlying model says is physically impossible. Digging into *why* surfaced a real confound: every two-qubit gate takes finite time to execute, and during that time qubits dephase. On edges where T2 (coherence time) is short relative to gate duration, what looks like "gate error" in the measurement is actually decoherence, not the gate itself. A screening ratio,

```
D = gate_error / (gate_duration / T2_min),  require D ≥ 1.5,
```

separated the two effects. Refitting on only the edges that passed the screen restored R² to 0.9648 — and retroactively showed that one of the anchor points already shipped in the emulator, from a qubit pair with T2 = 6.9 µs, had failed this exact screen the whole time. It was replaced.

This is Entry 021, and it's the entry the README still points to first, because it's the clearest example of the project's actual method: build something, believe it works, then go looking harder for the crack until you either can't find one or you do.

## Phase 2: Closed-form models hit a wall (Entries 022–040)

With the corrected forgiveness law in hand, the emulator (v3, then v4) got progressively more physical: real coupling-map topology, BFS-routed SWAP costs for non-adjacent qubit pairs, and eventually time-integrated decoherence accounting for how long a circuit actually spends executing, not just how many gates it has.

It closed most of the gap. Validated against Aer noise-model simulation on four representative circuits, three landed within 0.4–3.2 points of the reference. The fourth — `bell_scattered_0_26`, a Bell pair between two qubits routed across a long SWAP chain — missed by 14 points. That one circuit became the dominant source of model error and the motivating example for everything that followed.

The root cause, isolated across Entries 031–033, was structural: the emulator's routing logic entangled qubits *before* moving them into position rather than after, which meant the model was charging decoherence cost against the wrong portion of the circuit's timeline. Fixing the route-then-entangle order closed most of the gap — but a residual "floor-collapse" pattern remained. Certain circuit shapes, mostly long routes, would empirically collapse toward a 50% success rate — a coin flip — in a way that no amount of per-edge error tuning in a closed-form model could predict, because the failure isn't additive gate error accumulating linearly. It's compounding decoherence interacting with route length in a way a linear or power-law correction structurally can't capture.

That's where the project pivoted from "fit a better formula" to "learn the function directly."

## Phase 3: Building a graph neural network from scratch (Entries 041–056)

The chosen architecture: a graph neural network, implemented from scratch in JAX and Optax — no PyTorch, no pre-built GNN library. Each circuit becomes a graph: physical qubits as nodes, routed connections (including SWAP hops) as edges, with real per-edge calibration data as edge features. The model does a linear node embedding, two rounds of message passing between connected qubits, a masked mean-pool over the whole graph, concatenation with global circuit-level features, and a final MLP readout to a predicted success probability.

Training data came from simulating circuits against real chip calibration snapshots (Qiskit fake providers — recorded real hardware calibration, not live queued jobs) and recording actual outcomes. This required solving a genuinely tedious infrastructure problem: growing a labeled dataset large enough and diverse enough to generalize, one parallelized simulation batch at a time, with checkpointed resumability because no single sandbox session could run the whole thing in one pass. The dataset grew in stages — roughly 932, then 1,500, then 2,200, then past 3,000 circuits — with padding capacity (max nodes, max edges) bumped twice along the way as larger circuits started exceeding it.

The harder problem wasn't scale, it was honesty. A model trained and evaluated on the same chip's data will look great and tell you nothing about whether it actually learned physics versus memorized one chip's calibration numbers. The real test is cross-chip generalization: train exclusively on one chip's circuits, hold out a warm-calibration slice of that same chip, and evaluate cold on a *second* chip's circuits the model never saw a single example of. Early attempts at this failed in an instructive way — the model was implicitly using each chip's absolute error magnitude as a shortcut feature, so anything trained on one chip's error scale collapsed when moved to a chip with a different scale. The fix (Entry 048) was per-chip *relative* normalization: node and edge statistics computed separately per chip, with no chip-identity signal available to the model at all. That's what made cross-chip transfer honest rather than illusory.

Floor-collapse cases — the same long-route failures that broke the closed-form model — needed explicit handling even in the learned model: a 5x loss up-weighting on floor-collapse examples (Entry 051) to stop the training objective from treating them as rare outliers to be averaged away.

## Phase 4: Topological diversity and the cross-chip gap (Entries 052–058)

By Entry 056, the dataset's large-circuit tier had one structural limitation: every large circuit was a "star" — one control qubit at the hub, with all other qubits as spokes, entangled outward from the center. That's a real, useful topology, but it's one shape. The model had never seen a large circuit that entangles *sequentially* instead — qubit 0 to qubit 1 to qubit 2 and onward down a chain — which routes very differently across a real coupling map even when it produces the same target quantum state (a GHZ state, in both cases).

Adding a chain-topology circuit generator (Entry 058) and folding those circuits into the dataset — reaching 3,221 circuits total — closed a meaningful chunk of the remaining cross-chip gap. The harder direction, Kyiv-trained-evaluated-on-Sherbrooke, improved its floor-collapse error from 8.6 points to 7.0 points. Not a full fix — Sherbrooke chain-circuit generation itself stalled repeatedly during this phase and was honestly abandoned rather than forced through, leaving the chain-topology tier Kyiv-only for now — but a real, measured improvement traceable directly to topological diversity rather than just more of the same data.

## Phase 5: Teaching the model to know what it doesn't know (Entry 057)

A prediction without a confidence signal is only half a tool. The obvious approach — train an ensemble of five separate models and measure their disagreement — works, but costs five times the training compute for what is fundamentally one signal.

Instead, the project used Monte Carlo Dropout (Gal & Ghahramani, 2016): dropout layers stay active not just during training but during inference too. Running the same trained model 20 times on the same input, with different random dropout masks each time, produces 20 slightly different predictions. Their mean is the point prediction; their spread is a genuine, nearly-free uncertainty estimate.

The first validation attempt was wrong, and it's worth admitting why. It used a random 80/20 split of the *combined*, mixed-chip dataset, then labeled whichever chip happened to be first in the training fold as the "cold" chip for reporting purposes — except the model had already seen plenty of that chip's data through the random split. That produced a misleadingly weak result (a 1.08x uncertainty widening between "warm" and "cold" — barely a signal at all). The bug was caught before it reached any report, and the evaluation was rebuilt to match the project's own established protocol: train on one chip only, evaluate cold on the fully unseen other chip, in both directions.

With that fixed, the real result held up. Kyiv-trained, evaluated cold on Sherbrooke: uncertainty widened 2.77x between warm and cold predictions. Sherbrooke-trained, evaluated cold on Kyiv: 1.61x widening. In both directions, the widening correlated with actual prediction error — the model wasn't just reporting a bigger number on unfamiliar data by coincidence, it was reporting a bigger number *because* it was actually more wrong there. That's the difference between a decorative confidence interval and a calibrated one.

## Phase 6: Making it real — the live demo and the 3D chip

None of this is useful research if it stays in a JSON results file. Two things made it tangible.

The **live demo** is a fully static, self-contained web page with no backend: a precomputed lookup table covers all 16,002 possible qubit pairs across both chips, each with the closed-form prediction, the GNN's prediction, its MC-Dropout uncertainty band, and — where available — the real simulated result, all rendered client-side. It surfaces the model's own honesty in two extra ways: a "confident override" flag fires when the GNN disagrees sharply with the closed-form model *and* reports tight uncertainty at the same time — the case where you should probably trust the learned model over the formula — and a standing disclosure on every single result reminds the viewer that cross-chip transfer is asymmetric, rather than quietly presenting one blended reliability number.

The **3D visualization** renders the real 127-qubit Kyiv chip as a force-directed 3D layout computed from its actual coupling map — 144 real physical connections, not an artist's approximation — with two real, fully-routed circuits highlighted on top of it: the largest star-topology GHZ circuit and the largest chain-topology one, in different colors, so the topological difference that drove Phase 4's improvement is something you can see, not just read about in a results table. Building it surfaced two real bugs along the way, both caught by direct visual inspection rather than by any automated check: the first version reused node positions computed for the *wrong* chip, and never actually rendered the base connectivity mesh it claimed to show in its own legend; the second version's route highlights were attached to the wrong object in the 3D scene graph, so they stayed frozen in place while the rest of the chip rotated correctly during interaction. Both were fixed, verified live, and confirmed working before being called done.

## Where things stand

- **Dataset**: 3,221 routed circuits, both star and chain large-circuit topologies, spanning two real chip architectures.
- **Closed-form model (v4.1)**: fast, interpretable, and reliable except on long-route "floor-collapse" cases, which it structurally cannot see coming.
- **Learned model (GNN)**: same-chip MAE 1.09 points (R² = 0.977); cross-chip MAE 2.9–3.6 points depending on direction (R² 0.86–0.90) — genuinely tested, not just claimed, and openly asymmetric between directions rather than averaged into one misleadingly clean number.
- **Uncertainty quantification**: calibrated MC-Dropout, validated in both cross-chip directions, wired directly into the public demo.
- **Two public, interactive artifacts**: a live prediction tool and a 3D real-chip route visualization, both static pages requiring no backend and no IBM Quantum account.
- **A 134-page, 59-entry research log** documenting every one of the above steps, including the ones that didn't work the first time: the two-point power-law fits that couldn't fail, the route-then-entangle bug, the chip-identity shortcut the model was implicitly exploiting, the invalid random-split uncertainty evaluation, the stalled Sherbrooke chain generation, the frozen-route 3D bug. None of those are edited out.

## What's next

Real hardware validation is the clearest open item — the corrections made in Entries 021 through 058 were developed against Qiskit fake providers (real recorded calibration snapshots, not live queued devices), and re-running the core protocol on live hardware once access is available is the natural next check. Beyond that: finishing Sherbrooke chain-topology generation, adding a third chip architecture to stress-test generalization beyond a single pair of chips, and exploring whether an ensemble approach would agree with — or reveal blind spots in — the MC-Dropout uncertainty estimates.

---

*QuantumBridge is an independent, ongoing undergraduate research project. The live demo, 3D visualization, and full research log are public. If you work on quantum computing, noise modeling, or ML applied to physical systems, get in touch.*
