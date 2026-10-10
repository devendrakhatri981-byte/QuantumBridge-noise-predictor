"""Entry 100: 10-seed zero-shot, own-stats vs avg-stats normalization (JIT training).
usage: python3 entry100_run.py MODE CHIP   (MODE: own|avg; results merged into json, preds saved)"""
import json, os, sys, numpy as np
import entry100_ensemble_zero_shot as m

P = "quantumbridge_data/entry100_10seed_results.json"
mode, chip = sys.argv[1], sys.argv[2]
per = []
for s in range(10):
    r = m.run_fold(chip, seed=s, mode=mode, save=True)
    per.append((1 - r["gnn_mae"] / r["baseline_mae"]) * 100)
out = json.load(open(P)) if os.path.exists(P) else {}
out[f"{mode}/{chip}"] = per
json.dump(out, open(P, "w"), indent=2)
print(f"{mode:3s} {chip:11s} mean {np.mean(per):+6.1f}%  min {min(per):+6.1f}  max {max(per):+6.1f}  pos {sum(p>0 for p in per)}/10")
