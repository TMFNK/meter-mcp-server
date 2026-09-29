"""Step 0 contract tests: input shape and output schema."""

from meter_mcp.server_stdio import (
    classify_batch_impl,
    classify_day_impl,
    model_info_impl,
)


def test_model_info_shape():
    info = model_info_impl()
    assert info["model_version"] == "classical_logreg_v1"
    assert info["eval_accuracy"] == 0.90
    assert set(info["labels"]) == {"active", "standby", "off", "unsure"}


def test_classify_day_stub_validates_length():
    try:
        classify_day_impl([1.0] * 95, 1.0)
    except ValueError as e:
        assert "96" in str(e)
    else:
        raise AssertionError("expected ValueError for 95 values")


def test_classify_day_stub_output_schema():
    out = classify_day_impl([1.0] * 96, 1.0)
    assert out["label"] in ("active", "standby", "off", "unsure")
    assert isinstance(out["reason"], str) and out["reason"]
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["abstained"] == (out["label"] == "unsure")


def test_classify_batch_cap():
    days = [{"values": [1.0] * 96, "meter_reference_kwh": 1.0}] * 32
    try:
        classify_batch_impl(days)
    except ValueError as e:
        assert "31" in str(e)
    else:
        raise AssertionError("expected ValueError for 32 days")
