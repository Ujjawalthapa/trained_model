"""Step 6: database-first threat detection pipeline, with AI fallback."""
import hashlib
import json
import logging
import os
import time
from . import step5_predict as ai
from .common import count_kids, from_ayush

logger = logging.getLogger("OS_Sentinel.Pipeline")
DB_CACHE_SECONDS = 300
DB_RETRY_SECONDS = 60
MAX_NEW_HASHES_PER_SCAN = 15
MAX_HASH_FILE_MB = 150
LOG_AI_FLAGS_TO_DB = True
_hash_cache = {}
_db_cache = {}
_db_state = {"handler": None, "checked": False, "down_until": 0.0}

def _hash_key(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    if st.st_size > MAX_HASH_FILE_MB * 1024 * 1024:
        return None
    return (path, st.st_mtime_ns, st.st_size)

def sha256_of(path):
    key = _hash_key(path)
    if key is None:
        return None
    if key in _hash_cache:
        return _hash_cache[key]
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    except OSError:
        _hash_cache[key] = None
    return _hash_cache[key]

def _get_handler():
    if not _db_state["checked"]:
        _db_state["checked"] = True
        try:
            from src.monitor_adapter import handle_process_event
            _db_state["handler"] = handle_process_event
        except Exception as exc:
            logger.info("Database module unavailable (%s). Using AI only.", exc)
    return _db_state["handler"]

def to_db_event(e, sha256=None):
    event = {"pid": int(e["pid"]), "process_name": e["process_name"]}
    if e.get("ppid") is not None:
        event["ppid"] = int(e["ppid"])
    if e.get("executable_path"):
        event["executable_path"] = e["executable_path"]
    if sha256:
        event["sha256_hash"] = sha256
    if e.get("cpu_usage") is not None and e["cpu_usage"] >= 0:
        event["cpu_usage"] = round(float(e["cpu_usage"]), 2)
    if e.get("memory_usage_bytes") is not None and e["memory_usage_bytes"] >= 0:
        event["memory_usage_bytes"] = int(e["memory_usage_bytes"])
    return event

def _db_check(e, budget):
    handler = _get_handler()
    now = time.time()
    if handler is None or now < _db_state["down_until"] or not e.get("process_name"):
        return None
    path = e.get("executable_path")
    key = (e["pid"], path or e["process_name"])
    hit = _db_cache.get(key)
    fresh = hit and now - hit["time"] < DB_CACHE_SECONDS
    if fresh and not hit["hash_pending"]:
        return hit["result"]
    sha, pending = None, False
    hash_key = _hash_key(path) if path else None
    if hash_key is not None:
        sha = _hash_cache.get(hash_key)
        if hash_key not in _hash_cache:
            if budget["hashes"] > 0:
                budget["hashes"] -= 1
                sha = sha256_of(path)
            else:
                pending = True
    if fresh and hit["hash_pending"] and pending:
        return hit["result"]
    try:
        response = handler(to_db_event(e, sha))
    except (ValueError, TypeError) as exc:
        logger.debug("Database rejected PID %s: %s", e["pid"], exc)
        result = {"matched": False, "process_id": None}
        _db_cache[key] = {"time": now, "result": result, "hash_pending": False, "ai_logged": False}
        return result
    except Exception as exc:
        logger.warning("Database unavailable: %s", exc)
        _db_state["down_until"] = now + DB_RETRY_SECONDS
        return None
    detections = response.get("detections") or []
    result = {"matched": bool(detections), "process_id": response.get("process_id")}
    if detections:
        top = max(detections, key=lambda d: d.get("risk_score", 0))
        result.update(threat_name=top.get("threat_name"), severity=top.get("severity"),
                      risk_score=top.get("risk_score"))
    _db_cache[key] = {"time": now, "result": result, "hash_pending": pending,
                      "ai_logged": hit["ai_logged"] if hit else False}
    return result

def _log_ai_flag(key, process_id, name, risk, label):
    entry = _db_cache.get(key)
    if not LOG_AI_FLAGS_TO_DB or not entry or entry["ai_logged"] or not process_id:
        return
    try:
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_logs (action, entity_type, entity_id, details) VALUES (%s, %s, %s, %s)",
                    ("ai_flagged", "process_logs", process_id,
                     json.dumps({"process_name": name, "ai_risk": risk, "ai_label": label}))
                )
        entry["ai_logged"] = True
    except Exception as exc:
        logger.debug("Could not log AI flag: %s", exc)

def add_detection(procs):
    """Annotate every process; database first, AI for unmatched/unavailable entries."""
    if not procs:
        return procs
    try:
        connections = ai.connection_counts()
        events = [from_ayush(p, connections) for p in procs]
        kids = count_kids(events)
        budget = {"hashes": MAX_NEW_HASHES_PER_SCAN}
        need_ai = []
        for process, event in zip(procs, events):
            result = _db_check(event, budget)
            process["db_checked"] = result is not None
            if result and result.get("matched"):
                process.update(detection="DB Match", detection_source="database",
                               db_threat=result.get("threat_name"),
                               db_severity=result.get("severity"), db_risk=result.get("risk_score"),
                               ai_risk=None, ai_label="Skipped (database match)")
            else:
                need_ai.append((process, event, result))
        if need_ai:
            try:
                scores = ai.score_events([event for _, event, _ in need_ai], kids)
            except FileNotFoundError:
                for process, _, _ in need_ai:
                    process.update(detection="Not analysed", detection_source=None,
                                   ai_risk=None, ai_label="Model not trained")
                return procs
            for (process, event, result), (risk, label, reason) in zip(need_ai, scores):
                process["ai_risk"], process["ai_label"], process["ai_reason"] = risk, label, reason
                process["detection"] = {"Anomaly": "AI Anomaly", "Suspicious": "AI Suspicious"}.get(label, "Benign")
                process["detection_source"] = "ai" if label != "Benign" else None
                if label != "Benign" and result and result.get("process_id"):
                    key = (event["pid"], event.get("executable_path") or event["process_name"])
                    _log_ai_flag(key, result["process_id"], event["process_name"], risk, label)
    except Exception as exc:
        logger.warning("Detection pipeline failed: %s", exc)
        for process in procs:
            process.setdefault("detection", "Not analysed")
            process.setdefault("ai_label", "AI error")
    return procs

if __name__ == "__main__":
    print("Import add_detection(procs) from your dashboard after collecting process data.")
