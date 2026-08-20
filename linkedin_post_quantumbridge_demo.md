I just shipped the first public-facing piece of QuantumBridge, a project I've been building to predict how real IBM quantum chips actually fail.

The problem: every quantum circuit you run on real hardware degrades from noise, and the standard way to estimate how much is a closed-form physics formula. I built one of those first (v4.1) — then found it has a blind spot. Certain circuits crash straight to a 50% "coin flip" floor that the formula never sees coming, even on short, simple routes.

So I built a graph neural network from scratch (no PyTorch, just JAX) that represents each circuit as the real physical route it takes across the chip — real qubits, real SWAP chains, real calibration data — and trained it on 2,317 circuits simulated against real IBM Kyiv and Sherbrooke noise profiles.

Results: 91% lower error than the physics formula on exactly the failure cases it misses, and the model now handles routes up to 48 physical qubits wide.

Today I turned it into something anyone can try: pick any two qubits on a real 127-qubit chip layout, and see the physics model's prediction, the neural network's prediction, and (where we've actually run it) the real simulated result — side by side, instantly.

This is early-stage academic work, not a finished product — trained on simulated noise from real hardware calibration snapshots, not live queued jobs. But it's the first version I'm comfortable putting in front of people outside the project.

[image 1: the full real qubit-connection map behind the model]
[image 2: an illustrative look at the network itself]

#QuantumComputing #MachineLearning #GraphNeuralNetworks #Qiskit #IBMQuantum #UndergraduateResearch
