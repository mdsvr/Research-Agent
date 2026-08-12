import os
import re
import hashlib

from veritas.config import Config
from veritas.schemas import TraceLog


def save_trace(trace: TraceLog, cfg: Config) -> str:
    """Writes the pipeline trace to `cfg.trace_dir` for audit and debugging.

    Ad-hoc queries are named by a hash of the question so two different questions in the
    same session cannot overwrite each other's trace.
    """
    os.makedirs(cfg.trace_dir, exist_ok=True)

    if trace.qid:
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", trace.qid)
    else:
        stem = "query_" + hashlib.sha256(trace.query.encode("utf-8")).hexdigest()[:10]

    filepath = os.path.join(cfg.trace_dir, f"trace_{stem}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(trace.model_dump_json(indent=2))
    except OSError as e:
        # A trace is an audit artifact, not a result. Losing an entire paid evaluation run
        # to a transient file lock (seen: WinError 22 mid-run, writable again seconds later)
        # trades something expensive for something reproducible. Loud, not fatal.
        print(f"Warning: could not write trace {filepath}: {e}")
        return ""
    return filepath
