# Hardware-validation kit (pre-registered)

Goal: test whether QuantumBridge predictions hold on REAL IBM hardware, not just the Aer simulation they were trained on.

1. `python3 hw_validation/make_manifest.py kyiv` -> `manifest_kyiv.json`: 41 circuits (20 single bell pairs at 1-5 hops, 21 multi-bell with 2-4 pairs), with v4.1 and GNN predictions frozen BEFORE any hardware run. Commit the manifest before running so the predictions are timestamped.
2. `python3 hw_validation/run_on_ibm.py hw_validation/manifest_kyiv.json --backend <ibm_device>` (needs `pip install qiskit-ibm-runtime` and a saved IBM account). `--dry-run` runs a local noisy simulation as a smoke test only (circular, proves nothing); delete `counts_dryrun_kyiv.json` after testing.
3. `python3 hw_validation/score.py hw_validation/manifest_kyiv.json hw_validation/counts_<device>_kyiv.json` -> MAE, bias, rank correlation for v4.1 vs GNN, split single/multi.

Caveats to state in any write-up: (a) the fake-backend snapshots are of specific devices (Kyiv, Sherbrooke, ...) that may be retired or unavailable on a free account; run on the matching device if available, otherwise report it as an out-of-distribution test (the runner only checks qubit count); (b) hardware calibration drifts, so measured success may differ from the snapshot even on the same device; (c) 41 circuits x 8192 shots is roughly minutes of QPU time, check your plan's allowance; (d) results, good or bad, go in the log.

Manifests for other chips: `make_manifest.py sherbrooke` etc. (all 5 trained chips supported).
