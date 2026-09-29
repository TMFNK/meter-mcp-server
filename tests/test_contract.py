"""Step 1 contract tests: frozen classifier, offline, no private data."""

import json
from pathlib import Path

import pytest

from meter_mcp import classifier
from meter_mcp.audit import log_call


def _check_schema(out):
    assert out["label"] in ("active", "standby", "off", "unsure")
    assert isinstance(out["reason"], str) and out["reason"]
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["abstained"] == (out["label"] == "unsure")
    assert out["model_version"] == "classical_logreg_v1"


def test_model_info_shape():
    info = classifier.model_info()
    assert info["model_version"] == "classical_logreg_v1"
    assert info["eval_accuracy"] == 0.90
    assert set(info["labels"]) == {"active", "standby", "off", "unsure"}
    assert len(info["features"]) == 26
    assert info["threshold"] == 0.0


def test_threshold_frozen_zero():
    _, threshold = classifier.load_artifacts()
    assert threshold == 0.0


def test_off_day_confident():
    out = classifier.classify_day([0.0] * 96, 5.0, "2026-01-05")
    _check_schema(out)
    assert out["label"] == "off"
    assert out["abstained"] is False
    assert "max zero run 96" in out["reason"]


def test_borderline_abstains_like_sibling_failure_mode():
    # Low-reference meter, flat floor near 40% of reference: the frozen
    # model abstains, as on the 3 sibling eval failures (meter_7, ref 0.176).
    out = classifier.classify_day([0.07] * 96, 0.176, "2026-03-29")
    _check_schema(out)
    assert out["label"] == "unsure"
    assert out["abstained"] is True


def test_all_missing_abstains():
    out = classifier.classify_day([None] * 96, 5.0, "2026-01-05")
    _check_schema(out)
    assert out["label"] == "unsure"
    assert out["abstained"] is True


def test_missing_values_accepted():
    out = classifier.classify_day([8.0] * 48 + [None] * 48, 8.5, "2026-01-05")
    _check_schema(out)


def test_no_date_uses_weekday_fallback():
    out = classifier.classify_day([0.0] * 96, 5.0)
    _check_schema(out)
    assert out["label"] == "off"


def test_determinism():
    first = classifier.classify_day([0.0] * 96, 5.0, "2026-01-05")
    second = classifier.classify_day([0.0] * 96, 5.0, "2026-01-05")
    assert first == second


@pytest.mark.parametrize(
    "values,ref,date",
    [
        ([1.0] * 95, 1.0, "2026-01-05"),  # too few
        ([1.0] * 97, 1.0, "2026-01-05"),  # too many
        ([-1.0] * 96, 1.0, "2026-01-05"),  # negative kWh
        ([float("nan")] * 96, 1.0, "2026-01-05"),  # non-finite
        ([0.0] * 96, 0.0, "2026-01-05"),  # bad reference
        ([0.0] * 96, -2.0, "2026-01-05"),  # bad reference
        ([0.0] * 96, 5.0, "not-a-date"),  # bad date
        ([True] * 96, 5.0, "2026-01-05"),  # booleans rejected
    ],
)
def test_input_validation(values, ref, date):
    with pytest.raises(ValueError):
        classifier.classify_day(values, ref, date)


def test_classify_batch_happy_path():
    days = [
        {"values": [0.0] * 96, "meter_reference_kwh": 5.0, "date": "2026-01-05"},
        {"values": [None] * 96, "meter_reference_kwh": 5.0, "date": "2026-01-05"},
    ]
    out = classifier.classify_batch(days)
    assert out["model_version"] == "classical_logreg_v1"
    assert [r["label"] for r in out["results"]] == ["off", "unsure"]
    for result in out["results"]:
        _check_schema(result)


def test_classify_batch_cap():
    days = [{"values": [0.0] * 96, "meter_reference_kwh": 5.0}] * 32
    with pytest.raises(ValueError, match="31"):
        classifier.classify_batch(days)


def test_audit_writes_jsonl(tmp_path):
    target = tmp_path / "calls.jsonl"
    out = classifier.classify_day([0.0] * 96, 5.0, "2026-01-05")
    written = log_call(
        "classify_day", {"n": 96, "n_missing": 0}, out, path=target
    )
    assert written == target
    line = json.loads(target.read_text().splitlines()[0])
    assert line["tool"] == "classify_day"
    assert line["label"] == "off"
    assert line["model_version"] == "classical_logreg_v1"


def test_no_fitting_code_in_package():
    root = Path(classifier.__file__).resolve().parent
    hits = []
    for path in root.glob("*.py"):
        text = path.read_text()
        if "fit(" in text:
            hits.append(path.name)
    assert hits == [], f"fitting call found in {hits}"


# Step 2: STDIO tool wiring over the frozen classifier.

import asyncio

from meter_mcp.server_stdio import (
    DayInput,
    classify_batch_impl,
    classify_day_impl,
    mcp,
    model_info_impl,
)


def test_server_model_info_delegates_to_frozen_card():
    info = model_info_impl()
    assert info["model_version"] == "classical_logreg_v1"
    assert len(info["features"]) == 26
    assert info["threshold"] == 0.0


def test_server_classify_day_returns_real_label():
    out = classify_day_impl([0.0] * 96, 5.0, "2026-01-05")
    _check_schema(out)
    assert out["label"] == "off"
    # Output must be plain JSON data (no timestamps, no numpy types).
    json.dumps(out)


def test_server_classify_day_rejects_bad_length():
    with pytest.raises(ValueError, match="96"):
        classify_day_impl([1.0] * 95, 1.0)


def test_server_batch_accepts_models_and_dicts():
    days = [
        DayInput(values=[0.0] * 96, meter_reference_kwh=5.0, date="2026-01-05"),
        {"values": [None] * 96, "meter_reference_kwh": 5.0},
    ]
    out = classify_batch_impl(days)
    assert [r["label"] for r in out["results"]] == ["off", "unsure"]
    json.dumps(out)


def test_server_batch_cap():
    with pytest.raises(ValueError, match="31"):
        classify_batch_impl(
            [{"values": [0.0] * 96, "meter_reference_kwh": 5.0}] * 32
        )


# Step 3: HTTP health check and API key gate over Streamable HTTP.

import os
from unittest import mock

import pytest
from starlette.testclient import TestClient

from meter_mcp import server_http


@pytest.fixture(scope="module")
def http_client():
    # One lifespan per module: the MCP session manager can only run once
    # per FastMCP instance, so all Step 3 tests share a single client.
    # Auth reads MCP_API_KEY per request, so env can still vary per call.
    with TestClient(server_http.create_app()) as client:
        yield client


def test_healthz_is_public_and_shaped(http_client):
    with mock.patch.dict(os.environ, {"MCP_API_KEY": "test"}):
        response = http_client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "classical_logreg_v1"
    assert set(body["tools"]) == {"classify_day", "classify_batch", "model_info"}


def test_mcp_path_rejects_bad_key_with_401(http_client):
    with mock.patch.dict(os.environ, {"MCP_API_KEY": "test"}):
        missing = http_client.post("/mcp", json={})
        assert missing.status_code == 401
        assert missing.json() == {"error": "unauthorized"}
        wrong = http_client.post("/mcp", json={}, headers={"X-API-Key": "wrong"})
        assert wrong.status_code == 401


def test_mcp_path_accepts_bearer_and_x_api_key(http_client):
    with mock.patch.dict(os.environ, {"MCP_API_KEY": "test"}):
        bearer = http_client.post(
            "/mcp", json={}, headers={"Authorization": "Bearer test"}
        )
        assert bearer.status_code != 401
        header = http_client.post("/mcp", json={}, headers={"X-API-Key": "test"})
        assert header.status_code != 401


def test_mcp_path_open_without_key_for_local_run(http_client):
    with mock.patch.dict(os.environ, {}, clear=True):
        response = http_client.post("/mcp", json={})
        assert response.status_code != 401


def test_mcp_tool_list_and_end_to_end_call():
    async def run():
        tools = await mcp.list_tools()
        assert {t.name for t in tools} == {
            "classify_day",
            "classify_batch",
            "model_info",
        }
        contents = await mcp.call_tool(
            "classify_day",
            {
                "values": [0.0] * 96,
                "meter_reference_kwh": 5.0,
                "date": "2026-01-05",
            },
        )
        out = json.loads(contents[0].text)
        _check_schema(out)
        assert out["label"] == "off"

    asyncio.run(run())
