I tested my quantum-noise predictor on a real IBM quantum computer, and my own prediction failed. That turned out to be the most useful result so far.

QuantumBridge predicts how likely a circuit is to succeed on a specific IBM chip before you spend queue time on it. It pairs a closed-form physics formula with a neural network I built from scratch. Until recently, every test of the neural network was against simulated noise.

So I locked in predictions on GitHub before running anything, then ran them on the live ibm_fez.

What held up:
- With freshly pulled calibration, the physics formula predicted success within about 2.4 percentage points across 35 circuits, nearly unbiased. On multi-pair circuits it ranked outcomes with a 0.93 correlation.
- Old calibration snapshots were about 5 points too optimistic. Fresh calibration matters.

What didn't:
- A neural network trained on Heron-chip simulation only tied the physics formula on the real device (2.52 vs 2.49 points). On healthy routes the formula was already nearly exact in simulation, so there was little for a learned model to fix.
- My simulations said a coupler flagged "dead" in the calibration data should collapse success to about 50%. I committed that prediction, then ran circuits through the real coupler. It scored 86%. The flag was misleading, not the hardware. That means a chunk of my simulation results, where the neural network looked strongest, may be an artifact of how simulators read broken calibration entries.

One device, one day, a few dozen circuits. A real data point, not a proof. But a test you can fail is the only kind worth running.

Next: more hardware runs and a cleaner story about where learning actually helps.

Code, pre-registered predictions and raw counts: github.com/devendrakhatri981-byte/QuantumBridge-noise-predictor

#QuantumComputing #Qiskit #MachineLearning #IBMQuantum
