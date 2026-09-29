#!/usr/bin/env python3
"""STDIO server: frozen meter-day classifier as 3 MCP tools.

Step 0: transports work, model_info is real, classify_* are stubs
that return unsure until Step 1 vendors the frozen model.
"""

from mcp.server.fastmcp import FastMCP

from meter_mcp import LABELS, MODEL_EVAL_ACC, MODEL_VERSION, RULES_FLOOR_ACC

mcp = FastMCP("meter-mcp-server")


def model_info_impl() -> dict:
    """Return frozen model version, eval numbers, labels, and limits."""
    return {
        "model_version": MODEL_VERSION,
        "eval_accuracy": MODEL_EVAL_ACC,
        "rules_floor_accuracy": RULES_FLOOR_ACC,
        "labels": list(LABELS),
        "inputs": "96 x 15-min kWh per day, None means missing, plus meter_reference_kwh",
        "limits": "offline sklearn logreg, template reasons only, abstains as unsure, max 31 days per batch",
    }


def classify_day_impl(values: list, meter_reference_kwh: float) -> dict:
    """Classify one meter day. Step 0 stub: always unsure, validates length."""
    if len(values) != 96:
        raise ValueError(f"need 96 values, got {len(values)}")
    return {
        "label": "unsure",
        "reason": "Step 0 stub: frozen model lands in Step 1.",
        "confidence": 0.0,
        "abstained": True,
        "model_version": MODEL_VERSION,
    }


def classify_batch_impl(days: list) -> dict:
    """Classify up to 31 days. Step 0 stub: validates shape only."""
    if len(days) > 31:
        raise ValueError(f"max 31 days per batch, got {len(days)}")
    results = []
    for day in days:
        results.append(classify_day_impl(day["values"], day["meter_reference_kwh"]))
    return {"results": results, "model_version": MODEL_VERSION}


@mcp.tool()
def model_info() -> dict:
    """Return frozen model version, eval numbers, labels, and limits."""
    return model_info_impl()


@mcp.tool()
def classify_day(values: list, meter_reference_kwh: float) -> dict:
    """Classify one meter day. Step 0 stub: always unsure, validates length."""
    return classify_day_impl(values, meter_reference_kwh)


@mcp.tool()
def classify_batch(days: list) -> dict:
    """Classify up to 31 days. Step 0 stub: validates shape only."""
    return classify_batch_impl(days)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
