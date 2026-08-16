"""
QuantumBridge — Entry 037: first ML baseline.

WHY THIS EXISTS
---------------
Entries 034-036 built and validated a 238-circuit dataset (Kyiv +
Sherbrooke, Entries 034 + 035) and showed the closed-form v4.1 model has a
real, structural ceiling on "floor-collapse" routes where noise channels
correlate in a way no multiplicative formula captures. This entry asks the
obvious next question: can a learned model, given only the same circuit
and calibration features v4.1 uses, do better?

WHY A TABULAR MODEL, NOT A GNN, FIRST
--------------------------------------
The original plan (see Entry 033's close) was a GNN over the chip's
connectivity graph. 238 labeled circuits is not enough data to train a
graph neural network without either overfitting badly or leaning entirely
on heavy regularization that makes the comparison meaningless -- GNNs
typically need thousands of labeled graphs. A gradient-boosted tree over
the same scalar features v4.1's formulas already use (hop distance, real
edge/gate counts, worst edge error and worst T1 on the route) is a fair,
honest first baseline: if it can't beat v4.1 with today's dataset size,
a bigger architecture won't either. The GNN stays the long-term plan once
the dataset (already resumable/appendable via Entries 034/035's scripts)
grows past a few thousand circuits.
"""

import json

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

RECORDS = (json.load(open("quantumbridge_data/entry034_batch_dataset.json"))
          + json.load(open("quantumbridge_data/entry035_replication_dataset.json")))

CHIPS = ["kyiv", "sherbrooke"]
KINDS = ["bell", "ghz"]


def featurize(r):
    edge_err = r.get("worst_raw_edge_error_on_bfs_path")
    t1 = r.get("worst_t1_on_bfs_path")
    return {
        "chip_kyiv": 1.0 if r["chip"] == "kyiv" else 0.0,
        "chip_sherbrooke": 1.0 if r["chip"] == "sherbrooke" else 0.0,
        "kind_bell": 1.0 if r["kind"] == "bell" else 0.0,
        "kind_ghz": 1.0 if r["kind"] == "ghz" else 0.0,
        "bfs_hop_distance": float(r["bfs_hop_distance"]),
        "n_real_edges_used": float(r["n_real_edges_used"]),
        "n_2q_gates_total": float(r["n_2q_gates_total"]),
        "worst_edge_error": float(edge_err) if edge_err is not None else -1.0,
        "has_edge_error_feature": 1.0 if edge_err is not None else 0.0,
        "worst_t1_us": float(t1) * 1e6 if t1 is not None else -1.0,
        "has_t1_feature": 1.0 if t1 is not None else 0.0,
        "v4_1_prediction": float(r["v4_1_prediction"]),
    }


X_dicts = [featurize(r) for r in RECORDS]
feature_names = list(X_dicts[0].keys())
X = np.array([[d[k] for k in feature_names] for d in X_dicts])
y = np.array([r["aer_ground_truth"] for r in RECORDS])
v41 = np.array([r["v4_1_prediction"] for r in RECORDS])
is_floor_collapse = np.array([abs(r["aer_ground_truth"] - 0.5) < 0.02
                              and r["bfs_hop_distance"] <= 10 for r in RECORDS])

print(f"dataset: {len(RECORDS)} circuits, {sum(is_floor_collapse)} floor-collapse cases")
print(f"features: {feature_names}\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=5, random_state=42),
}

results = {}
for name, make_model in models.items():
    fold_mae, fold_r2, fold_mae_fc = [], [], []
    for train_idx, test_idx in kf.split(X):
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_mae.append(mean_absolute_error(y[test_idx], pred))
        fold_r2.append(r2_score(y[test_idx], pred))
        fc_mask = is_floor_collapse[test_idx]
        if fc_mask.sum() > 0:
            fold_mae_fc.append(mean_absolute_error(y[test_idx][fc_mask], pred[fc_mask]))
    results[name] = {
        "cv_mae": float(np.mean(fold_mae)), "cv_mae_std": float(np.std(fold_mae)),
        "cv_r2": float(np.mean(fold_r2)),
        "cv_mae_floor_collapse_only": float(np.mean(fold_mae_fc)) if fold_mae_fc else None,
    }
    print(f"{name:<20} CV MAE={np.mean(fold_mae)*100:.2f}pts (+/-{np.std(fold_mae)*100:.2f})  "
          f"CV R2={np.mean(fold_r2):.3f}  "
          f"floor-collapse-only MAE={np.mean(fold_mae_fc)*100:.2f}pts" if fold_mae_fc else "")

v41_mae = mean_absolute_error(y, v41)
v41_r2 = r2_score(y, v41)
v41_mae_fc = mean_absolute_error(y[is_floor_collapse], v41[is_floor_collapse])
print(f"\n{'v4.1 (closed-form)':<20} MAE={v41_mae*100:.2f}pts  R2={v41_r2:.3f}  "
      f"floor-collapse-only MAE={v41_mae_fc*100:.2f}pts")

# fit best model on full data for feature importances
best_name = min(results, key=lambda k: results[k]["cv_mae"])
best_model = models[best_name]()
best_model.fit(X, y)
importances = sorted(zip(feature_names, best_model.feature_importances_),
                     key=lambda t: -t[1])
print(f"\nBest model: {best_name}")
print("Feature importances:")
for fn, imp in importances:
    print(f"  {fn:<24}{imp:.3f}")

summary = {
    "n_circuits": len(RECORDS), "n_floor_collapse": int(sum(is_floor_collapse)),
    "features": feature_names,
    "model_results": results,
    "v4_1_baseline": {"mae": v41_mae, "r2": v41_r2, "mae_floor_collapse_only": v41_mae_fc},
    "best_model": best_name,
    "feature_importances": {fn: float(imp) for fn, imp in importances},
}
json.dump(summary, open("quantumbridge_data/entry037_ml_baseline.json", "w"),
          indent=2, default=str)
print("\nSaved to quantumbridge_data/entry037_ml_baseline.json")
