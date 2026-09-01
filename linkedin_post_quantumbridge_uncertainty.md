Update on QuantumBridge, the project predicting how real IBM quantum chips actually fail.

Since my last post, three things changed.

**1. The model now knows when it doesn't know.**
A neural network's prediction is only useful if you know how much to trust it. I added Monte Carlo Dropout — the model runs 20 slightly-different stochastic passes over the same circuit and reports both a mean prediction and a spread. Tested honestly in both directions across two real chip architectures (train on one, test cold on the other, never mixed): predictions on unfamiliar territory come back with 1.6x to 2.8x wider uncertainty bands than predictions on familiar ground, and that widening actually tracks real error. The model isn't just guessing with more confidence than it's earned.

**2. The dataset grew to 3,221 real-routed circuits**, now covering two structurally different large-circuit shapes — hub-and-spoke "star" entanglement and sequential "chain" entanglement — instead of one. Adding that topological diversity measurably closed a generalization gap I'd flagged as unresolved: cross-chip error on the hardest direction dropped from 8.6 points to 7.0.

**3. You can now see it, not just query it.**
Built a 3D view of the real 127-qubit chip — actual coupling map, actual routed circuits drawn as physical paths across real qubits, rotate and zoom freely — so the difference between a "star" circuit and a "chain" circuit is something you can look at, not just a table of numbers.

Try it yourself:
Live prediction demo: https://devendrakhatri981-byte.github.io/QuantumBridge-noise-predictor/quantumbridge_live_demo.html
3D chip + routing viz: https://devendrakhatri981-byte.github.io/QuantumBridge-noise-predictor/quantumbridge_3d_star_vs_chain.html

Still undergraduate, still independent, still built from scratch in JAX with no PyTorch — and still logging every wrong turn in a public research log, now past 130 pages.

#QuantumComputing #MachineLearning #GraphNeuralNetworks #UncertaintyQuantification #Qiskit #IBMQuantum #UndergraduateResearch
