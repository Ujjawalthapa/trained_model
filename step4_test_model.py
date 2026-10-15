"""Step 4: evaluate model against held-out normal rows and optional suspicious samples."""
import argparse
import os
import numpy as np
from . import step5_predict as ai
from .common import DATA_DIR, OUT_DIR, feature_rows, read_pratham_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suspicious", default=os.path.join(DATA_DIR, "suspicious_samples.csv"))
    args = parser.parse_args()
    bundle = ai.load_bundle()
    normal_path = os.path.join(OUT_DIR, "normal_test.npy")
    if not os.path.exists(normal_path):
        raise SystemExit("normal_test.npy not found. Run python -m ai.step3_train_model first.")
    normal = np.load(normal_path)
    if not os.path.exists(args.suspicious):
        raise SystemExit(f"{args.suspicious} not found. Add suspicious simulator data to evaluate.")
    rows, scenarios = [], []
    for snapshot in read_pratham_csv(args.suspicious):
        for event, row in zip(snapshot, feature_rows(snapshot, bundle["fills"])):
            rows.append(row)
            scenarios.append(event.get("scenario") or "unknown")
    if not rows:
        raise SystemExit("No suspicious samples found in CSV.")
    fake = np.asarray(rows, dtype=float)
    threshold = ai.ANOMALY_FROM
    fake_flagged = ai.risk_scores(fake) >= threshold
    normal_flagged = ai.risk_scores(normal) >= threshold
    caught, false_alarms = int(fake_flagged.sum()), int(normal_flagged.sum())
    precision = caught / (caught + false_alarms) if caught + false_alarms else 0
    print(f"Suspicious caught: {caught}/{len(fake)} (recall={caught/len(fake):.2f})")
    print(f"Normal flagged: {false_alarms}/{len(normal)} ({false_alarms/len(normal):.2%})")
    print(f"Precision (rough combined-set estimate): {precision:.2f}")
    for scenario in sorted(set(scenarios)):
        mask = np.array([s == scenario for s in scenarios])
        print(f"{scenario:<26} {int(fake_flagged[mask].sum())}/{int(mask.sum())} ({fake_flagged[mask].mean():.0%})")

if __name__ == "__main__":
    main()
