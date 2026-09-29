#!/usr/bin/env python3
"""Frozen meter-day classifier: local-day values + meter reference -> label.

Step 1 core library. No fitting happens here. The model file and the
operating point are vendored byte-identical from the sibling repo
(TMFNK/meter-day-classifier); this module loads them read-only and
applies the frozen argmax-unsure rule at t=0.0.

Evidence math below is vendored from MbitAI's shared meter-day harness
(first shipped in the sibling repo). Logic is unchanged; only the input
wiring is new: instead of raw CSVs, the caller passes 92, 96, or 100
interval values (None means missing) plus the meter reference
(whole-period positive p95 kWh per interval). Timestamps are derived from
an optional ISO date in Europe/Berlin so weekday buckets and is_weekend
stay correct. Without a date the day is treated as a weekday
(Monday 2026-01-05); this fallback is stated in model_info limits.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, date as Date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from meter_mcp import LABELS, MODEL_EVAL_ACC, MODEL_VERSION, RULES_FLOOR_ACC

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent.parent / "artifacts"

# Vendored constants (same values as the sibling evidence.py).
INTERVAL = timedelta(minutes=15)
LOW_BAND_FRACTION = 0.35
BORDERLINE_FLOOR_LOW = 0.28
BORDERLINE_FLOOR_HIGH = 0.55

# Default date when the caller passes none: a Monday, so buckets behave
# as a plain weekday and is_weekend is 0.
DEFAULT_DATE = "2026-01-05"
LOCAL_TIMEZONE = "Europe/Berlin"
LOCAL_ZONE = ZoneInfo(LOCAL_TIMEZONE)

# These bounds describe the range represented by the frozen synthetic fit.
# They prevent obvious unit mistakes (for example, sending Wh as kWh) from
# being treated as model-backed predictions.
MIN_REFERENCE_KWH = 0.1
MAX_REFERENCE_KWH = 150.0
MAX_INTERVAL_KWH = 1000.0

# Hard cap on reason text. The template below is ~170 chars, so this is a
# guardrail, not a formatter: it only fires if the template ever changes.
REASON_MAX_LEN = 500

# Closed feature list. Order is the column order. Must match the frozen
# operating_point.json exactly (checked at load time).
FEATURES = [
    "missing_pct",
    "n_missing",
    "n_valid",
    "zero_pct_valid",
    "zero_count",
    "max_zero_run",
    "trailing_zero_run",
    "positive_p10_kwh",
    "positive_median_kwh",
    "positive_p95_kwh",
    "positive_max_kwh",
    "positive_min_kwh",
    "flatness_ratio",
    "floor_to_ref_ratio",
    "low_positive_share_valid",
    "low_weekday_night_share_d",
    "low_weekday_day_share_d",
    "low_weekend_share_d",
    "weekday_night_median_kwh",
    "weekday_day_median_kwh",
    "weekend_median_kwh",
    "night_to_day_ratio",
    "meter_reference_kwh",
    "low_threshold_kwh",
    "is_borderline_floor",
    "is_weekend",
]


@dataclass(frozen=True)
class BucketEvidence:
    """Low-positive counts, safe share, and median for one time bucket."""

    valid_count: int
    low_positive_count: int
    low_positive_share: float | None
    median_kwh: float | None = None


@dataclass(frozen=True)
class DayEvidence:
    """Structured evidence for one meter day (vendored shape)."""

    day_id: str
    meter_id: str
    date: str
    values: tuple[float | None, ...]
    timestamps: tuple[datetime, ...]
    n_intervals: int
    n_missing: int
    missing_pct: float
    n_valid: int
    zero_count: int
    zero_pct_valid: float
    positive_min_kwh: float | None
    positive_median_kwh: float | None
    positive_p95_kwh: float | None
    positive_max_kwh: float | None
    meter_reference_kwh: float
    low_threshold_kwh: float
    low_positive_count: int
    low_positive_share_valid: float
    weekday_night: BucketEvidence
    weekday_day: BucketEvidence
    weekend: BucketEvidence
    max_zero_run: int = 0
    trailing_zero_run: int = 0
    positive_p10_kwh: float | None = None
    floor_to_ref_ratio: float | None = None
    flatness_ratio: float | None = None
    weekday_night_median_kwh: float | None = None
    weekday_day_median_kwh: float | None = None
    weekend_median_kwh: float | None = None
    night_to_day_ratio: float | None = None

    @property
    def low_weekday_night_share_d(self) -> float | None:
        return self.weekday_night.low_positive_share

    @property
    def low_weekday_day_share_d(self) -> float | None:
        return self.weekday_day.low_positive_share

    @property
    def low_weekend_share_d(self) -> float | None:
        return self.weekend.low_positive_share

    @property
    def is_borderline_floor(self) -> bool:
        if self.floor_to_ref_ratio is None:
            return False
        return BORDERLINE_FLOOR_LOW <= self.floor_to_ref_ratio <= BORDERLINE_FLOOR_HIGH


def _compute_zero_runs(values: tuple[float | None, ...]) -> tuple[int, int]:
    max_run = 0
    current_run = 0
    for value in values:
        if value == 0.0:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    trailing_run = 0
    for value in reversed(values):
        if value == 0.0:
            trailing_run += 1
        else:
            break
    return max_run, trailing_run


def stats_for(values) -> dict:
    """Safe day statistics. None stays missing; missing never becomes zero."""
    values = list(values)
    valid = [v for v in values if v is not None]
    positive = [v for v in valid if v > 0]
    max_zero_run, trailing_zero_run = _compute_zero_runs(tuple(values))
    p10 = float(np.percentile(positive, 10)) if positive else None
    p50 = statistics.median(positive) if positive else None
    p95 = float(np.percentile(positive, 95)) if positive else None
    flatness = (p50 / p95) if (p50 is not None and p95 and p95 > 0) else None
    return {
        "n_intervals": len(values),
        "n_missing": len(values) - len(valid),
        "missing_pct": (
            100 * (len(values) - len(valid)) / len(values) if values else 0.0
        ),
        "zero_pct_valid": 100 * sum(v == 0 for v in valid) / len(valid)
        if valid
        else 0.0,
        "positive_min_kwh": min(positive) if positive else None,
        "positive_p10_kwh": p10,
        "positive_median_kwh": p50,
        "positive_p95_kwh": p95,
        "positive_max_kwh": max(positive) if positive else None,
        "max_zero_run": max_zero_run,
        "trailing_zero_run": trailing_zero_run,
        "flatness_ratio": flatness,
    }


def _local_day_interval_count(day: Date) -> int:
    start = datetime.combine(day, datetime.min.time(), tzinfo=LOCAL_ZONE)
    next_start = datetime.combine(
        day + timedelta(days=1), datetime.min.time(), tzinfo=LOCAL_ZONE
    )
    elapsed = next_start.astimezone(UTC) - start.astimezone(UTC)
    return int(elapsed / INTERVAL)


def _derived_interval_ends(date_text: str, count: int) -> tuple[datetime, ...]:
    day = Date.fromisoformat(date_text)
    start = datetime.combine(day, datetime.min.time(), tzinfo=LOCAL_ZONE)
    start_utc = start.astimezone(UTC)
    return tuple(
        (start_utc + INTERVAL * (i + 1)).astimezone(LOCAL_ZONE)
        for i in range(count)
    )


def _bucket_for_start(start: datetime) -> str:
    if start.weekday() >= 5:
        return "weekend"
    if 7 <= start.hour <= 18:
        return "weekday_day"
    return "weekday_night"


def _bucket_result(
    valid_count: int,
    low_count: int,
    median_kwh: float | None = None,
) -> BucketEvidence:
    share = low_count / valid_count if valid_count else None
    return BucketEvidence(
        valid_count=valid_count,
        low_positive_count=low_count,
        low_positive_share=share,
        median_kwh=median_kwh,
    )


def day_of_timestamp(stamp: datetime) -> str:
    return (stamp - INTERVAL).date().isoformat()


def encode_day_evidence(
    values,
    meter_reference_kwh: float,
    timestamps=None,
    *,
    day_id: str = "unknown_day",
    meter_id: str = "unknown_meter",
    date: str | None = None,
) -> DayEvidence:
    """Encode one day with the frozen 35% low-band contract."""
    value_tuple = tuple(values)
    count = len(value_tuple)
    if date is None:
        date = day_id.rsplit("_", 1)[-1] if "_" in day_id else DEFAULT_DATE
    timestamp_tuple = (
        tuple(timestamps)
        if timestamps is not None
        else _derived_interval_ends(date, count)
    )
    if len(timestamp_tuple) != count:
        raise ValueError(
            f"{day_id} has {count} values but {len(timestamp_tuple)} timestamps"
        )
    wrong_days = [
        s.isoformat() for s in timestamp_tuple if day_of_timestamp(s) != date
    ]
    if wrong_days:
        raise ValueError(
            f"{day_id} contains timestamps outside local day {date}: "
            f"{wrong_days[:3]}"
        )
    if meter_reference_kwh <= 0:
        raise ValueError("meter_reference_kwh must be strictly positive")

    stats = stats_for(value_tuple)
    low_threshold = LOW_BAND_FRACTION * meter_reference_kwh
    bucket_counts = {
        "weekday_night": [0, 0],
        "weekday_day": [0, 0],
        "weekend": [0, 0],
    }
    bucket_values: dict[str, list[float]] = {
        "weekday_night": [],
        "weekday_day": [],
        "weekend": [],
    }
    low_positive_count = 0
    for stamp, value in zip(timestamp_tuple, value_tuple):
        bucket = _bucket_for_start(stamp - INTERVAL)
        if value is None:
            continue
        bucket_counts[bucket][0] += 1
        bucket_values[bucket].append(value)
        if 0 < value <= low_threshold:
            bucket_counts[bucket][1] += 1
            low_positive_count += 1

    bucket_medians: dict[str, float | None] = {}
    for name, vals in bucket_values.items():
        bucket_medians[name] = float(statistics.median(vals)) if vals else None

    night_med = bucket_medians["weekday_night"]
    day_med = bucket_medians["weekday_day"]
    if night_med is None or day_med is None:
        night_to_day_ratio = None
    elif day_med > 0:
        night_to_day_ratio = round(night_med / day_med, 4)
    elif night_med == 0:
        night_to_day_ratio = 1.0
    else:
        night_to_day_ratio = None

    p10 = stats["positive_p10_kwh"]
    floor_to_ref_ratio = (
        round(float(p10) / meter_reference_kwh, 4) if p10 is not None else None
    )

    n_valid = count - int(stats["n_missing"])
    low_share = low_positive_count / n_valid if n_valid else 0.0
    return DayEvidence(
        day_id=day_id,
        meter_id=meter_id,
        date=date,
        values=value_tuple,
        timestamps=timestamp_tuple,
        n_intervals=count,
        n_missing=int(stats["n_missing"]),
        missing_pct=float(stats["missing_pct"]),
        n_valid=n_valid,
        zero_count=sum(v == 0 for v in value_tuple if v is not None),
        zero_pct_valid=float(stats["zero_pct_valid"]),
        positive_min_kwh=stats["positive_min_kwh"],
        positive_median_kwh=stats["positive_median_kwh"],
        positive_p95_kwh=stats["positive_p95_kwh"],
        positive_max_kwh=stats["positive_max_kwh"],
        meter_reference_kwh=meter_reference_kwh,
        low_threshold_kwh=low_threshold,
        low_positive_count=low_positive_count,
        low_positive_share_valid=low_share,
        weekday_night=_bucket_result(
            *bucket_counts["weekday_night"],
            median_kwh=bucket_medians["weekday_night"],
        ),
        weekday_day=_bucket_result(
            *bucket_counts["weekday_day"],
            median_kwh=bucket_medians["weekday_day"],
        ),
        weekend=_bucket_result(
            *bucket_counts["weekend"], median_kwh=bucket_medians["weekend"]
        ),
        max_zero_run=int(stats["max_zero_run"]),
        trailing_zero_run=int(stats["trailing_zero_run"]),
        positive_p10_kwh=stats["positive_p10_kwh"],
        floor_to_ref_ratio=floor_to_ref_ratio,
        flatness_ratio=stats["flatness_ratio"],
        weekday_night_median_kwh=bucket_medians["weekday_night"],
        weekday_day_median_kwh=bucket_medians["weekday_day"],
        weekend_median_kwh=bucket_medians["weekend"],
        night_to_day_ratio=night_to_day_ratio,
    )


def _num(value) -> float:
    return float("nan") if value is None else float(value)


def row_from_evidence(evidence: DayEvidence, is_weekend: float) -> list[float]:
    """Build the ordered 26-feature row from encoded evidence."""
    return [
        _num(evidence.missing_pct),
        _num(evidence.n_missing),
        _num(evidence.n_valid),
        _num(evidence.zero_pct_valid),
        _num(evidence.zero_count),
        _num(evidence.max_zero_run),
        _num(evidence.trailing_zero_run),
        _num(evidence.positive_p10_kwh),
        _num(evidence.positive_median_kwh),
        _num(evidence.positive_p95_kwh),
        _num(evidence.positive_max_kwh),
        _num(evidence.positive_min_kwh),
        _num(evidence.flatness_ratio),
        _num(evidence.floor_to_ref_ratio),
        _num(evidence.low_positive_share_valid),
        _num(evidence.low_weekday_night_share_d),
        _num(evidence.low_weekday_day_share_d),
        _num(evidence.low_weekend_share_d),
        _num(evidence.weekday_night_median_kwh),
        _num(evidence.weekday_day_median_kwh),
        _num(evidence.weekend_median_kwh),
        _num(evidence.night_to_day_ratio),
        _num(evidence.meter_reference_kwh),
        _num(evidence.low_threshold_kwh),
        1.0 if evidence.is_borderline_floor else 0.0,
        float(is_weekend),
    ]


def _fmt(value, decimals: int = 4) -> str:
    return "not available" if value is None else f"{value:.{decimals}f}"


def _fmt_share(value) -> str:
    return "not available" if value is None else f"{value * 100:.0f}%"


def build_reason(evidence: DayEvidence, label: str) -> str:
    """One-line display reason with printed numbers (same template as sibling)."""
    assert label in LABELS, f"bad label {label!r}"
    night_med = _fmt(evidence.weekday_night_median_kwh)
    ref = _fmt(evidence.meter_reference_kwh)
    low_share = _fmt_share(evidence.low_positive_share_valid)
    reason = (
        f"label is {label} because night median {night_med} kWh vs meter "
        f"reference {ref} kWh; low-band share {low_share} of valid intervals; "
        f"max zero run {evidence.max_zero_run}; baseline p95 {ref} kWh."
    )
    return reason[:REASON_MAX_LEN]


_model_cache: dict = {}


def load_artifacts():
    """Load the frozen model + operating point (read-only, cached)."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["threshold"]
    model_path = ARTIFACTS / "model.joblib"
    operating_path = ARTIFACTS / "operating_point.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"missing frozen model at {model_path}; vendor it from the "
            "sibling repo (see artifacts/README.md)"
        )
    model = joblib.load(model_path)
    operating = json.loads(operating_path.read_text())
    if operating["features"] != FEATURES:
        raise ValueError("artifact/model feature drift")
    if not operating["threshold_rule"].startswith("max decline-F1"):
        raise ValueError("operating point is not the frozen calibration")
    threshold = float(operating["threshold"])
    classes = list(model.classes_)
    if set(classes) != set(LABELS):
        raise ValueError(f"unexpected model classes {classes}")
    _model_cache["model"] = model
    _model_cache["threshold"] = threshold
    return model, threshold


def _validate_values(values, date: str) -> list[float | None]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("values must be a list of numbers or None")
    expected = _local_day_interval_count(Date.fromisoformat(date))
    if len(values) != expected:
        raise ValueError(
            f"need {expected} values for {date} in {LOCAL_TIMEZONE}, "
            f"got {len(values)}"
        )
    cleaned: list[float | None] = []
    for v in values:
        if v is None:
            cleaned.append(None)
        elif isinstance(v, bool):
            raise ValueError("values must be numbers or None, got boolean")
        elif isinstance(v, (int, float)):
            f = float(v)
            if not math.isfinite(f):
                raise ValueError("values must be finite numbers or None")
            if f < 0:
                raise ValueError("values must be >= 0 kWh (got negative)")
            if f > MAX_INTERVAL_KWH:
                raise ValueError(
                    f"values must be <= {MAX_INTERVAL_KWH:g} kWh per interval"
                )
            cleaned.append(f)
        else:
            raise ValueError("values must be numbers or None")
    return cleaned


def _validate_reference(meter_reference_kwh) -> float:
    if isinstance(meter_reference_kwh, bool) or not isinstance(
        meter_reference_kwh, (int, float)
    ):
        raise ValueError("meter_reference_kwh must be a number")
    ref = float(meter_reference_kwh)
    if not math.isfinite(ref) or ref <= 0:
        raise ValueError("meter_reference_kwh must be strictly positive")
    if not MIN_REFERENCE_KWH <= ref <= MAX_REFERENCE_KWH:
        raise ValueError(
            "meter_reference_kwh must be between "
            f"{MIN_REFERENCE_KWH:g} and {MAX_REFERENCE_KWH:g} kWh"
        )
    return ref


def _validate_date(date) -> str:
    if date is None:
        return DEFAULT_DATE
    if not isinstance(date, str):
        raise ValueError("date must be an ISO string YYYY-MM-DD or None")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"date must be ISO YYYY-MM-DD, got {date!r}")
    try:
        parsed = Date.fromisoformat(date)
    except ValueError:
        raise ValueError(f"date must be ISO YYYY-MM-DD, got {date!r}")
    if parsed.isoformat() != date:
        raise ValueError(f"date must be ISO YYYY-MM-DD, got {date!r}")
    if parsed == Date.max:
        raise ValueError(f"date {date!r} cannot represent a complete local day")
    return parsed.isoformat()


def classify_day(
    values,
    meter_reference_kwh: float,
    date: str | None = None,
) -> dict:
    """Classify one meter day with the frozen model.

    Args:
        values: 96 numbers or None per 15-min interval. None means missing.
        meter_reference_kwh: whole-period positive p95 per interval.
        date: optional ISO day (YYYY-MM-DD) for weekday buckets and
            is_weekend. Defaults to a Monday (weekday fallback).

    Returns a dict with label, reason, confidence, abstained, model_version,
    and the effective date used for bucket assignment.
    """
    day = _validate_date(date)
    cleaned = _validate_values(values, day)
    ref = _validate_reference(meter_reference_kwh)
    is_weekend = 1.0 if datetime.fromisoformat(day).weekday() >= 5 else 0.0

    timestamps = _derived_interval_ends(day, len(cleaned))
    evidence = encode_day_evidence(
        cleaned, ref, timestamps, date=day,
    )
    row = np.array(row_from_evidence(evidence, is_weekend), dtype=float).reshape(
        1, -1
    )
    model, threshold = load_artifacts()
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    top2 = np.sort(proba)[-2:]
    margin = float(top2[1] - top2[0])
    argmax = classes[int(np.argmax(proba))]
    # Frozen combined rule: at t=0.0 the margin clause is a no-op and
    # unsure is argmax-only (same as sibling train.decline_curve).
    label = "unsure" if (argmax == "unsure" or margin < threshold) else argmax
    confidence = float(np.max(proba))
    return {
        "label": label,
        "reason": build_reason(evidence, label),
        "confidence": confidence,
        "abstained": label == "unsure",
        "model_version": MODEL_VERSION,
        "date": day,
    }


def classify_batch(days) -> dict:
    """Classify up to 31 days. Each day needs values + meter_reference_kwh."""
    if not isinstance(days, (list, tuple)):
        raise ValueError("days must be a list of up to 31 day objects")
    if len(days) > 31:
        raise ValueError(f"max 31 days per batch, got {len(days)}")
    results = []
    for index, day in enumerate(days):
        if not isinstance(day, dict):
            raise ValueError(f"day at index {index}: each day must be an object")
        try:
            results.append(
                classify_day(
                    day.get("values"),
                    day.get("meter_reference_kwh"),
                    day.get("date"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"day at index {index}: {exc}") from None
    return {"results": results, "model_version": MODEL_VERSION}


def model_info() -> dict:
    """Library-level model card (server wires this as an MCP tool in Step 2)."""
    _, threshold = load_artifacts()
    return {
        "model_version": MODEL_VERSION,
        "eval_accuracy": MODEL_EVAL_ACC,
        "rules_floor_accuracy": RULES_FLOOR_ACC,
        "labels": list(LABELS),
        "threshold": threshold,
        "threshold_rule": "argmax-unsure at t=0.0 (margin clause is a no-op)",
        "features": list(FEATURES),
        "inputs": "92, 96, or 100 x 15-min kWh intervals per Europe/Berlin "
        "local day, None means missing, plus meter_reference_kwh (positive "
        "p95 in the same kWh unit); optional ISO date for weekday buckets "
        "(defaults to 2026-01-05)",
        "input_units": "kWh per 15-minute interval; do not send Wh",
        "supported_interval_counts": [92, 96, 100],
        "supported_reference_kwh": {
            "min": MIN_REFERENCE_KWH,
            "max": MAX_REFERENCE_KWH,
        },
        "max_interval_kwh": MAX_INTERVAL_KWH,
        "limits": "offline sklearn logreg, template reasons only, abstains "
        "as unsure, max 31 days per batch; interval count must match the "
        "Europe/Berlin local day, and values/reference must stay in the "
        "published kWh envelope",
    }
