#!/usr/bin/env python3
"""STDIO server: frozen meter-day classifier as 3 MCP tools.

Step 2: typed tools delegate to the frozen Step 1 classifier.
No fitting, no retraining, no raw CSV access. Audit logging is
best-effort (never breaks a classification call).

Note: this module must NOT use `from __future__ import annotations`.
FastMCP 1.9.4 inspects real annotation objects at decoration time;
string annotations break tool registration.
"""

import argparse
import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, BeforeValidator

from meter_mcp import classifier
from meter_mcp.audit import log_classify_batch, log_classify_day, log_rejection

mcp = FastMCP("meter-mcp-server")
LOGGER = logging.getLogger(__name__)


def _reject_lax_number(value):
    """Keep Pydantic from turning booleans and strings into numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a JSON number, not a boolean or string")
    return value


KwhNumber = Annotated[float, BeforeValidator(_reject_lax_number)]


class DayInput(BaseModel):
    """One local meter day plus its meter reference."""

    values: list[KwhNumber | None]
    meter_reference_kwh: KwhNumber
    date: str | None = None


def _audit_day(values, meter_reference_kwh, output, date=None) -> None:
    try:
        log_classify_day(
            values,
            meter_reference_kwh,
            output,
            date=output.get("date", date),
        )
    except Exception as exc:
        LOGGER.warning("audit write failed for classify_day: %s", exc)


def _audit_batch(days, output) -> None:
    try:
        log_classify_batch(days, output)
    except Exception as exc:
        LOGGER.warning("audit write failed for classify_batch: %s", exc)


def _audit_rejection(tool, inputs_summary, error) -> None:
    try:
        log_rejection(tool, inputs_summary, error)
    except Exception as exc:
        LOGGER.warning("audit write failed for rejected %s: %s", tool, exc)


def model_info_impl() -> dict:
    """Return frozen model version, eval numbers, labels, and limits."""
    return classifier.model_info()


def classify_day_impl(
    values: list, meter_reference_kwh: float, date: str | None = None
) -> dict:
    """Classify one meter day with the frozen model."""
    if isinstance(values, DayInput):
        values, meter_reference_kwh, date = (
            values.values,
            values.meter_reference_kwh,
            values.date,
        )
    try:
        output = classifier.classify_day(values, meter_reference_kwh, date)
    except Exception as exc:
        _audit_rejection(
            "classify_day",
            {
                "n": len(values) if isinstance(values, (list, tuple)) else 0,
                "meter_reference_kwh": meter_reference_kwh,
                **({"date": date} if date is not None else {}),
            },
            exc,
        )
        raise
    _audit_day(values, meter_reference_kwh, output, date)
    return output


def classify_batch_impl(days: list) -> dict:
    """Classify up to 31 days with the frozen model."""
    normalized = []
    try:
        if not isinstance(days, (list, tuple)):
            raise ValueError("days must be a list of up to 31 day objects")
        for index, day in enumerate(days):
            if isinstance(day, DayInput):
                normalized.append(day.model_dump())
            elif isinstance(day, dict):
                normalized.append(day)
            else:
                raise ValueError(f"day at index {index}: each day must be an object")
        output = classifier.classify_batch(normalized)
    except Exception as exc:
        _audit_rejection(
            "classify_batch",
            {"n_days": len(days) if isinstance(days, (list, tuple)) else 0},
            exc,
        )
        raise
    _audit_batch(normalized, output)
    return output


@mcp.tool()
def model_info() -> dict:
    """Return frozen model version, eval numbers, labels, and limits."""
    return model_info_impl()


@mcp.tool()
def classify_day(
    values: list[KwhNumber | None],
    meter_reference_kwh: KwhNumber,
    date: str | None = None,
) -> dict:
    """Classify one meter day.

    92, 96, or 100 x 15-min kWh values (None means missing) plus the meter reference
    (whole-period positive p95 per interval). Optional ISO date
    (YYYY-MM-DD) keeps weekday buckets correct; omit it and the day
    uses the documented fallback date. Returns label, reason,
    confidence, abstained flag, and model version.
    """
    return classify_day_impl(values, meter_reference_kwh, date)


@mcp.tool()
def classify_batch(days: list[DayInput]) -> dict:
    """Classify up to 31 meter days. Same output per day, in order."""
    return classify_batch_impl(days)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    mcp.run()


if __name__ == "__main__":
    main()
