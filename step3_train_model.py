"""Step 3: train IsolationForest using normal baseline data."""
import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from .common import FEATURES, OUT_DIR, read_feature_csv

CONTAMINATION = 0.01
CLEAN = os.path.join(OUT_DIR, "baseline_clean.csv")
RANGE_FEATURES = ["cpu_usage", "memory_mb", "thread_count", "connections", "children"]

def main():
    if not os.path.exists(CLEAN):
        raise SystemExit(
            f"{CLEAN} not found. Add normal process data with columns: "
            + ", ".join(FEATURES)
        )
    X = read_feature_csv(CLEAN)
    if len(X) < 50:
        print(f"WARNING: only {len(X)} rows. More normal samples are recommended.")
    if len(X) < 2:
        raise SystemExit("At least 2 rows are required to train.")
    conn_col = FEATURES.index("connections")
    folder_col = FEATURES.index("normal_folder")
    conn_known = X[~np.isnan(X[:, conn_col]), conn_col]
    fills = {
        "connections": float(np.median(conn_known)) if len(conn_known) else 0.0,
        "normal_folder": 1.0,
    }
    X[np.isnan(X[:, conn_col]), conn_col] = fills["connections"]
    X[np.isnan(X[:, folder_col]), folder_col] = fills["normal_folder"]
    # Other missing values use each column's median.
    for i in range(X.shape[1]):
        missing = np.isnan(X[:, i])
        median = float(np.nanmedian(X[:, i])) if np.any(~missing) else 0.0
        X[missing, i] = median
    order = np.random.default_rng(42).permutation(len(X))
    cut = max(1, int(len(X) * 0.8))
    if cut >= len(X):
        cut = len(X) - 1
    train, test = X[order[:cut]], X[order[cut:]]
    np.save(os.path.join(OUT_DIR, "normal_test.npy"), test)
    model = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=42)
    model.fit(train)
    limits = {name: float(np.percentile(train[:, FEATURES.index(name)], 99))
              for name in RANGE_FEATURES}
    bundle = {"model": model, "features": FEATURES, "fills": fills,
              "limits": limits, "contamination": CONTAMINATION,
              "trained_rows": len(train)}
    with open(os.path.join(OUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    print(f"Trained on {len(train)} rows; saved ai/output/model.pkl")
    print("Normal 99% limits:", {k: round(v, 2) for k, v in limits.items()})

if __name__ == "__main__":
    main()
