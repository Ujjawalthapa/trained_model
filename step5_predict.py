"""Step 5: predict process risk using IsolationForest plus range checks."""
import logging
import os
import pickle
import time
import numpy as np
from .common import FEATURES, OUT_DIR, feature_rows, from_ayush

logger = logging.getLogger("OS_Sentinel.AI")
SUSPICIOUS_FROM = 40
ANOMALY_FROM = 60
RANGE_SUSPICIOUS = 2.0
RANGE_ANOMALY = 3.0
MIN_LIMITS = {"cpu_usage": 10.0, "memory_mb": 50.0, "thread_count": 10.0,
              "connections": 10.0, "children": 5.0}
_bundle = None
_conn_cache = {"time": 0.0, "value": None}

def load_bundle():
    global _bundle
    if _bundle is None:
        with open(os.path.join(OUT_DIR, "model.pkl"), "rb") as f:
            _bundle = pickle.load(f)
    return _bundle

def forest_risk(X):
    decision = load_bundle()["model"].decision_function(np.nan_to_num(np.asarray(X, dtype=float)))
    return 100 / (1 + np.exp(10 * decision))

def range_risk(X):
    X = np.nan_to_num(np.asarray(X, dtype=float))
    limits = load_bundle().get("limits", {})
    risk = np.zeros(len(X))
    reasons = [""] * len(X)
    for name, limit in limits.items():
        limit = max(float(limit), MIN_LIMITS.get(name, 1.0))
        col = X[:, FEATURES.index(name)]
        # 0 at the baseline; rises as the value exceeds the limit.
        r = np.where(col > limit, np.minimum(100, 20 + 20 * ((col - limit) / limit)), 0.0)
        for i in np.where(r > risk)[0]:
            risk[i] = r[i]
            reasons[i] = f"{name}={col[i]:.1f}; normal 99% limit={limit:.1f}"
    return risk, reasons

def risk_scores(X):
    return np.maximum(forest_risk(X), range_risk(X)[0])

def label_for(score):
    if score >= ANOMALY_FROM:
        return "Anomaly"
    if score >= SUSPICIOUS_FROM:
        return "Suspicious"
    return "Benign"

def score_events(events, kids=None):
    if not events:
        return []
    bundle = load_bundle()
    X = np.asarray(feature_rows(events, bundle.get("fills"), kids), dtype=float)
    forest = forest_risk(X)
    ranged, reasons = range_risk(X)
    output = []
    for f, r, why in zip(forest, ranged, reasons):
        risk = max(float(f), float(r))
        output.append((round(risk, 1), label_for(risk), why if r >= f and r > 0 else ""))
    return output

def connection_counts():
    """Return per-PID internet socket counts; cached for five seconds."""
    now = time.time()
    if now - _conn_cache["time"] < 5:
        return _conn_cache["value"]
    try:
        import psutil
        counts = {}
        for c in psutil.net_connections(kind="inet"):
            if c.pid:
                counts[c.pid] = counts.get(c.pid, 0) + 1
    except Exception:
        counts = None
    _conn_cache.update(time=now, value=counts)
    return counts

def add_ai_scores(procs):
    if not procs:
        return procs
    try:
        connections = connection_counts()
        events = [from_ayush(p, connections) for p in procs]
        for p, (risk, label, reason) in zip(procs, score_events(events)):
            p["ai_risk"], p["ai_label"], p["ai_reason"] = risk, label, reason
    except FileNotFoundError:
        for p in procs:
            p["ai_risk"], p["ai_label"] = None, "Model not trained"
    except Exception as exc:
        logger.warning("AI scoring failed: %s", exc)
        for p in procs:
            p["ai_risk"], p["ai_label"] = None, "AI error"
    return procs

if __name__ == "__main__":
    try:
        import psutil
        procs = []
        for p in psutil.process_iter(["pid", "ppid", "name", "exe", "cpu_percent",
                                      "memory_info", "num_threads"]):
            try:
                info = p.info
                procs.append({"pid": info["pid"], "ppid": info["ppid"], "name": info["name"],
                              "exe": info["exe"], "cpu_percent": info["cpu_percent"],
                              "memory_rss_mb": info["memory_info"].rss / (1024*1024)
                                  if info["memory_info"] else 0,
                              "num_threads": info["num_threads"]})
            except (psutil.Error, OSError):
                continue
        add_ai_scores(procs)
        procs.sort(key=lambda p: p.get("ai_risk") or 0, reverse=True)
        print(f"{'PID':>7} {'RISK':>6} {'LABEL':<12} NAME")
        for p in procs[:10]:
            print(f"{p['pid']:>7} {str(p.get('ai_risk')):>6} {p.get('ai_label','?'):<12} {p.get('name')}")
    except ImportError:
        print("Install requirements first: pip install -r requirements.txt")
