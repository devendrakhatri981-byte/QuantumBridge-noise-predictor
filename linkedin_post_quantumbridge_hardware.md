I finally tested my quantum-noise predictor on a real IBM quantum computer. Here's what held up and what didn't.

QuantumBridge predicts how likely a quantum circuit is to succeed on a specific IBM chip, before you spend queue time running it. It combines a closed-form physics formula with a graph neural network I built from scratch. Until now, every test of the neural network was against simulated noise built from real calibration snapshots.

So I locked in predictions for 35 circuits (committed to GitHub before any job ran), pulled the live calibration of ibm_fez, and ran them on the real device.

What held up:
- The physics formula, given fresh calibration, predicted success within about 2.4 percentage points on average, nearly unbiased.
- On circuits with several entangled pairs, where success ranged from 43% to 96%, it ranked them with a 0.93 correlation. A constant guess is off by 8.4 points there; the model by 2.2.
- Calibration freshness mattered a lot: predictions from an older snapshot were about 5 points too optimistic.

What didn't:
- My neural network, trained on a different chip family, did no better than the physics formula on this device. Its advantage is real in simulation. I have not shown it on real hardware yet.
- On single qubit pairs, success only varied by about 4 points, so there's no evidence of per-pair skill from this test.
- Along the way I found that a few dead couplers in the chip data silently break the formula. That's now a known blind spot I need to fix.

One device, one day, 35 circuits. A first real data point, not a proof. Next: more hardware runs and a model trained on this chip family.

Everything, including the pre-registered predictions and raw counts, is in the repo: github.com/devendrakhatri981-byte/QuantumBridge-noise-predictor

#QuantumComputing #Qiskit #MachineLearning #IBMQuantum
