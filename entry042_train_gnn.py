"""Entry 042: retrain the Entry 041 GNN on the larger 2,198-circuit dataset."""

import json
import os

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

import entry041_train_gnn as m

FOLD_CKPT = "quantumbridge_data/entry042_gnn_folds.json"
RESULTS_PATH = "quantumbridge_data/entry042_gnn_results.json"
EPOCHS = 180


def load_ckpt():
    if os.path.exists(FOLD_CKPT):
        return json.load(open(FOLD_CKPT))
    return {}


if __name__ == "__main__":
    raw, y_all, v41_all = m.raw, m.y_all, m.v41_all
    n = len(raw)
    idx_all = np.arange(n)
    is_fc = np.array([abs(g["aer_ground_truth"] - 0.5) < 0.02 and g["bfs_hop_distance"] <= 10
                      for g in raw])
    print(f"n={n}, floor-collapse n={is_fc.sum()}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(idx_all))

    done = load_ckpt()
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        if str(fold_i) in done:
            print(f"fold {fold_i}: already done, skipping")
            continue
        test_loss, test_pred = m.train_fold(train_idx, test_idx, seed=fold_i, epochs=EPOCHS)
        mae = mean_absolute_error(y_all[test_idx], test_pred)
        r2 = r2_score(y_all[test_idx], test_pred)
        mask = is_fc[test_idx]
        mae_fc = (mean_absolute_error(y_all[test_idx][mask], test_pred[mask])
                  if mask.sum() > 0 else None)
        done[str(fold_i)] = {"test_idx": test_idx.tolist(), "pred": test_pred.tolist(),
                             "mae": mae, "r2": r2, "mae_fc": mae_fc, "n_fc": int(mask.sum())}
        json.dump(done, open(FOLD_CKPT, "w"))
        print(f"fold {fold_i}: MAE={mae*100:.2f} R2={r2:.3f} "
              f"(n_test={len(test_idx)}, n_fc={mask.sum()}) -- saved checkpoint")
        break

    if len(done) == 5:
        fold_mae = [done[str(i)]["mae"] for i in range(5)]
        fold_r2 = [done[str(i)]["r2"] for i in range(5)]
        fold_mae_fc = [done[str(i)]["mae_fc"] for i in range(5) if done[str(i)]["mae_fc"] is not None]

        print(f"\nGNN CV MAE={np.mean(fold_mae)*100:.2f} R2={np.mean(fold_r2):.3f} "
              f"floor-collapse MAE={np.mean(fold_mae_fc)*100:.2f}")

        v41_mae = mean_absolute_error(y_all, v41_all)
        v41_fc_mae = mean_absolute_error(y_all[is_fc], v41_all[is_fc])
        print(f"v4.1: MAE={v41_mae*100:.2f} floor-collapse MAE={v41_fc_mae*100:.2f}")

        result = {
            "n": n, "n_floor_collapse": int(is_fc.sum()),
            "gnn_cv_mae": float(np.mean(fold_mae)), "gnn_cv_r2": float(np.mean(fold_r2)),
            "gnn_cv_mae_floor_collapse": float(np.mean(fold_mae_fc)),
            "v41_mae": float(v41_mae), "v41_mae_floor_collapse": float(v41_fc_mae),
        }
        json.dump(result, open(RESULTS_PATH, "w"), indent=2)
        print(f"\nSaved to {RESULTS_PATH}")
    else:
        print(f"\n{len(done)}/5 folds done so far -- rerun to continue")
