"""Shared data conversion and feature helpers for the OPTAD AI engine."""
import csv
import os
import math

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "ai", "data")
OUT_DIR = os.path.join(ROOT_DIR, "ai", "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "cpu_usage",
    "memory_mb",
    "thread_count",
    "connections",
    "children",
    "normal_folder",
]

def _number(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        n = float(value)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default

def running_from_normal_folder(path):
    """Return True for common OS/application folders, False for temp folders."""
    if not path:
        return None
    p = str(path).replace("/", "\\").lower()
    if any(x in p for x in (r"\appdata\local\temp", r"\windows\temp", r"\tmp\\")):
        return False
    if any(x in p for x in (r"\windows\system32", r"\windows\syswow64",
                             r"\program files", r"\program files (x86)")):
        return True
    # On non-Windows, common system locations count as normal.
    unix = str(path).replace("\\", "/").lower()
    if unix.startswith(("/usr/bin/", "/usr/sbin/", "/bin/", "/sbin/", "/usr/lib/")):
        return True
    if unix.startswith(("/tmp/", "/var/tmp/")):
        return False
    return None

def from_ayush(proc, connections=None):
    """Convert a process dictionary from the scanner into a shared event."""
    path = proc.get("exe") or proc.get("executable_path")
    try:
        pid = int(proc.get("pid"))
    except (TypeError, ValueError):
        pid = 0
    try:
        ppid = int(proc.get("ppid")) if proc.get("ppid") is not None else None
    except (TypeError, ValueError):
        ppid = None
    mem_mb = _number(proc.get("memory_rss_mb"))
    mem_bytes = _number(proc.get("memory_usage_bytes"))
    if mem_bytes is None and mem_mb is not None:
        mem_bytes = mem_mb * 1024 * 1024
    conn = None
    if isinstance(connections, dict):
        conn = _number(connections.get(pid))
    elif connections is not None:
        conn = _number(connections)
    normal = running_from_normal_folder(path)
    return {
        "pid": pid,
        "ppid": ppid,
        "process_name": proc.get("name") or proc.get("process_name") or "unknown",
        "executable_path": path if path and "access denied" not in str(path).lower()
                              and "protected" not in str(path).lower() else None,
        "cpu_usage": _number(proc.get("cpu_percent", proc.get("cpu_usage"))),
        "memory_usage_bytes": int(mem_bytes) if mem_bytes is not None else None,
        "memory_mb": mem_mb if mem_mb is not None else (mem_bytes / (1024*1024) if mem_bytes is not None else None),
        "thread_count": _number(proc.get("num_threads", proc.get("thread_count"))),
        "connections": conn,
        "normal_folder": normal,
    }

def from_pratham_row(row):
    """Convert a simulator CSV row into the shared event format."""
    path = row.get("executable_path") or row.get("exe") or row.get("path")
    try:
        pid = int(row.get("pid", 0))
    except (TypeError, ValueError):
        pid = 0
    e = {
        "pid": pid,
        "ppid": _number(row.get("ppid")),
        "process_name": row.get("process_name") or row.get("name") or "unknown",
        "executable_path": path,
        "cpu_usage": _number(row.get("cpu_usage", row.get("cpu_percent"))),
        "memory_usage_bytes": _number(row.get("memory_usage_bytes")),
        "memory_mb": _number(row.get("memory_mb", row.get("memory_rss_mb"))),
        "thread_count": _number(row.get("thread_count", row.get("num_threads"))),
        "connections": _number(row.get("internet_connection_count")),
        "normal_folder": _number(row.get("running_from_normal_folder")),
        "scenario": row.get("scenario"),
    }
    if e["memory_mb"] is None and e["memory_usage_bytes"] is not None:
        e["memory_mb"] = e["memory_usage_bytes"] / (1024 * 1024)
    if e["memory_usage_bytes"] is None and e["memory_mb"] is not None:
        e["memory_usage_bytes"] = int(e["memory_mb"] * 1024 * 1024)
    if e["normal_folder"] is None:
        e["normal_folder"] = 1.0 if running_from_normal_folder(path) is True else (
            0.0 if running_from_normal_folder(path) is False else None)
    return e

def count_kids(events):
    """Count child processes using each event's parent PID."""
    counts = {}
    for e in events:
        parent = e.get("ppid")
        if parent is not None:
            counts[parent] = counts.get(parent, 0) + 1
    return counts

def feature_rows(events, fills=None, kids=None):
    """Return rows ordered exactly like FEATURES; unknowns use saved fill values."""
    fills = fills or {"connections": 0.0, "normal_folder": 1.0}
    kids = kids or count_kids(events)
    rows = []
    for e in events:
        vals = {
            "cpu_usage": e.get("cpu_usage"),
            "memory_mb": e.get("memory_mb"),
            "thread_count": e.get("thread_count"),
            "connections": e.get("connections"),
            "children": kids.get(e.get("pid"), 0),
            "normal_folder": e.get("normal_folder"),
        }
        row = []
        for name in FEATURES:
            value = vals.get(name)
            if value is None:
                value = fills.get(name, 0.0)
            row.append(float(value))
        rows.append(row)
    return rows

def read_feature_csv(path):
    """Read a CSV containing the FEATURES columns."""
    import numpy as np
    data = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FEATURES if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("CSV is missing columns: " + ", ".join(missing))
        for row in reader:
            data.append([_number(row.get(c), float("nan")) for c in FEATURES])
    return np.asarray(data, dtype=float)

def read_pratham_csv(path):
    """Read simulator CSV; each row is treated as one process snapshot."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [from_pratham_row(r) for r in csv.DictReader(f)]
    # The simulator may include snapshot identifiers; group rows by snapshot if present.
    snapshots = {}
    for raw, event in zip(_read_dict_rows(path), rows):
        key = raw.get("snapshot") or raw.get("snapshot_id") or "all"
        snapshots.setdefault(key, []).append(event)
    return list(snapshots.values())

def _read_dict_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
