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

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from meter_mcp import classifier
from meter_mcp.audit import log_classify_batch, log_classify_day

mcp = FastMCP("meter-mcp-server")


class DayInput(BaseModel):
    """One meter day: 96 interval values plus the meter reference."""

    values: list[float | None]
    meter_reference_kwh: float
    date: str | None = None


def _audit_day(values, meter_reference_kwh, output, date=None) -> None:
    try:
        log_classify_day(values, meter_reference_kwh, output, date=date)
    except Exception:
        pass


def _audit_batch(days, output) -> None:
    try:
        log_classify_batch(days, output)
    except Exception:
        pass


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
    output = classifier.classify_day(values, meter_reference_kwh, date)
    _audit_day(values, meter_reference_kwh, output, date)
    return output


def classify_batch_impl(days: list) -> dict:
    """Classify up to 31 days with the frozen model."""
    normalized = []
    for day in days:
        if isinstance(day, DayInput):
            normalized.append(day.model_dump())
        elif isinstance(day, dict):
            normalized.append(day)
        else:
            raise ValueError("each day must be an object with values")
    output = classifier.classify_batch(normalized)
    _audit_batch(normalized, output)
    return output


@mcp.tool()
def model_info() -> dict:
    """Return frozen model version, eval numbers, labels, and limits."""
    return model_info_impl()


@mcp.tool()
def classify_day(
    values: list[float | None],
    meter_reference_kwh: float,
    date: str = "",
) -> dict:
    """Classify one meter day.

    96 x 15-min kWh values (None means missing) plus the meter reference
    (whole-period positive p95 per interval). Optional ISO date
    (YYYY-MM-DD) keeps weekday buckets correct; omit it (empty string)
    and the day is treated as a weekday. Returns label, reason,
    confidence, abstained flag, and model version.
    """
    return classify_day_impl(values, meter_reference_kwh, date or None)


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
