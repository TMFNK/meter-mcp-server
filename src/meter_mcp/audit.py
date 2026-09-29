#!/usr/bin/env python3
"""JSONL audit writer: one line per classify call. Offline, never committed.

Logs summaries only (counts and labels), never the full 96-value array,
so audit files stay small and carry no raw customer series.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _default_path() -> Path:
    override = os.environ.get("MCP_AUDIT_PATH")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    return repo_root / "audit" / "calls.jsonl"


def _summarize_values(values) -> dict:
    try:
        items = list(values) if values is not None else []
    except TypeError:
        return {"n": 0, "n_missing": 0}
    return {
        "n": len(items),
        "n_missing": sum(1 for v in items if v is None),
    }


def log_call(
    tool: str,
    inputs_summary: dict,
    output: dict,
    path: Path | str | None = None,
) -> Path:
    """Append one audit line and return the file path."""
    target = Path(path) if path else _default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "inputs": inputs_summary,
        "label": output.get("label") if isinstance(output, dict) else None,
        "abstained": output.get("abstained")
        if isinstance(output, dict)
        else None,
        "model_version": output.get("model_version")
        if isinstance(output, dict)
        else None,
    }
    with target.open("a") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    return target


def log_classify_day(
    values,
    meter_reference_kwh,
    output: dict,
    date=None,
    path: Path | str | None = None,
) -> Path:
    """Convenience wrapper for single-day calls."""
    summary = _summarize_values(values)
    summary["meter_reference_kwh"] = meter_reference_kwh
    if date is not None:
        summary["date"] = date
    return log_call("classify_day", summary, output, path)


def log_classify_batch(
    days, output: dict, path: Path | str | None = None
) -> Path:
    """Convenience wrapper for batch calls (counts only)."""
    items = list(days) if isinstance(days, (list, tuple)) else []
    summary = {
        "n_days": len(items),
        "n_results": len(output.get("results", []))
        if isinstance(output, dict)
        else 0,
    }
    return log_call("classify_batch", summary, output, path)
