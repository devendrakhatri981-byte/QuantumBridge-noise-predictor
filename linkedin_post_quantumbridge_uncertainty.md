Update on QuantumBridge — my project predicting how real IBM quantum chips fail.

Three additions since my last post: the model now reports a calibrated uncertainty band with every prediction (Monte Carlo Dropout), not just a number — and tested honestly across two chip architectures, that uncertainty reliably widens 1.6x–2.8x exactly where the model is more likely to be wrong. The training set grew to 3,221 real-routed circuits with a second large-circuit topology added, which measurably improved cross-chip generalization. And there's now a 3D view of the real 127-qubit chip with actual routed circuits drawn across it, so you can see the difference instead of just reading a table.

Try it:
Demo: https://devendrakhatri981-byte.github.io/QuantumBridge-noise-predictor/quantumbridge_live_demo.html
3D viz: https://devendrakhatri981-byte.github.io/QuantumBridge-noise-predictor/quantumbridge_3d_star_vs_chain.html

Still undergrad, still independent, still logging every wrong turn in a public research log (130+ pages now).

#QuantumComputing #MachineLearning #GraphNeuralNetworks #Qiskit #IBMQuantum #UndergraduateResearch
