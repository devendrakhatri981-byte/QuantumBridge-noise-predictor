"""Entry 100: ensemble (mean) of per-seed zero-shot predictions, own-stats mode."""
import glob, json, numpy as np
res = {}
for chip in ("kyiv", "sherbrooke", "brisbane"):
    d = [np.load(f) for f in sorted(glob.glob(f"quantumbridge_data/entry100_preds/own_{chip}_seed*.npz"))]
    y, b = d[0]["y"], d[0]["base"]
    bm = np.abs(b - y).mean() * 100
    singles = [np.abs(x["pred"] - y).mean() * 100 for x in d]
    row = {"n_seeds": len(d), "baseline_mae": bm, "single_mean_mae": float(np.mean(singles))}
    for k in (3, 5, 10):
        e = np.abs(np.mean([x["pred"] for x in d[:k]], axis=0) - y).mean() * 100
        row[f"ens{k}_mae"] = float(e); row[f"ens{k}_pct"] = float((1 - e / bm) * 100)
    row["single_pct"] = float((1 - np.mean(singles) / bm) * 100)
    res[chip] = row
    print(f"{chip:11s} base {bm:.2f} | single {np.mean(singles):.2f} ({row['single_pct']:+.1f}%) | "
          f"ens3 {row['ens3_mae']:.2f} ({row['ens3_pct']:+.1f}%) ens5 {row['ens5_mae']:.2f} ({row['ens5_pct']:+.1f}%) "
          f"ens10 {row['ens10_mae']:.2f} ({row['ens10_pct']:+.1f}%)")
json.dump(json.loads(json.dumps(res, default=float)), open("quantumbridge_data/entry100_ensemble_results.json", "w"), indent=2)
