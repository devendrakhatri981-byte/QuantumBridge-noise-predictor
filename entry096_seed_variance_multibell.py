"""
QuantumBridge — Entry 096: multi-seed variance check on the Entry 095
leave-one-chip-out zero-shot test, run after growing the multi-bell
dataset from 650 to 985 circuits (Entry 118/119 growth + rebuild).

Entry 095's original single-seed (seed=0) result on the expanded dataset
showed Brisbane's holdout FLIP from +23.5% (at 650 circuits) to -10.1%
(at 985 circuits) -- the opposite of the hoped-for "more data closes the
gap" outcome. Before writing that up as a real finding, this script
checks whether it's genuine or just single-seed training variance, by
re-running each holdout fold at 3 different seeds (0, 1, 2) and looking
at the spread.

This also became the proper test of whether Entry 095's Sherbrooke
regression was a data-scale artifact (expected to shrink with more
data) or a stable, reproducible chip-specific generalization failure
(won't shrink just by adding seeds) -- single-seed evidence alone can't
distinguish these.
"""

import json

import entry095_zero_shot_multibell as m

CHIPS = ("kyiv", "sherbrooke", "brisbane")
SEEDS = (0, 1, 2)
OUT_PATH = "quantumbridge_data/entry096_seed_variance_results.json"

if __name__ == "__main__":
    results = {}
    print("=== Entry 096: multi-seed variance check (985-circuit dataset) ===")
    for chip in CHIPS:
        results[chip] = []
        for seed in SEEDS:
            r = m.run_fold(chip, seed=seed)
            better = (1 - r["gnn_mae"] / r["baseline_mae"]) * 100
            results[chip].append({"seed": seed, "gnn_mae": r["gnn_mae"], "gnn_r2": r["gnn_r2"],
                                  "baseline_mae": r["baseline_mae"], "pct_better": better})
            print(f"  {chip:12s} seed={seed}  GNN MAE={r['gnn_mae']*100:5.2f}  "
                 f"R2={r['gnn_r2']:6.3f}  vs baseline: {better:+.1f}%")
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nsaved -> {OUT_PATH}")
    print("\nsummary (mean +/- range across 3 seeds):")
    for chip in CHIPS:
        vals = [r["pct_better"] for r in results[chip]]
        print(f"  {chip:12s} mean={sum(vals)/len(vals):+.1f}%  range=[{min(vals):+.1f}%, {max(vals):+.1f}%]")
