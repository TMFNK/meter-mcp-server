# meter-mcp-server

A small server that answers one question about energy meters: **"What was this meter doing on a given day?"** It speaks a protocol called MCP so that AI assistants can ask it directly.

## What is MCP?

MCP (Model Context Protocol) is a standard way for AI assistants to call external tools. Think of it like a USB port for AI: any assistant that speaks MCP can plug into any server that speaks MCP, without custom glue code.

This project is one such server. It wraps a frozen machine-learning model behind three MCP tools, so an assistant like Claude can ask "was this meter active last Tuesday?" and get a structured answer with a reason and a confidence score.

## What the server does

Energy meters record consumption every 15 minutes. A full day is 96 numbers (or 92/100 on daylight-saving days). The server takes those numbers plus a meter reference value and classifies the day into one of four labels:

| Label     | What it means                              |
| --------- | ------------------------------------------ |
| `active`  | The meter was actively consuming energy    |
| `standby` | The meter was in a low-power standby state |
| `off`     | The meter was off (all zeros or near-zero) |
| `unsure`  | The model declines to make a forced call   |

`unsure` is a first-class answer, not a failure. When the evidence is thin, the server says so instead of guessing.

### A concrete example

You send 96 numbers representing a Monday's 15-minute kWh readings, plus the meter reference (the meter's typical p95 consumption). The server replies:

```json
{
  "label": "standby",
  "reason": "label is standby because night median 0.12 kWh vs meter reference 2.45 kWh; low-band share 78% of valid intervals; max zero run 12; baseline p95 2.45 kWh.",
  "confidence": 0.89,
  "abstained": false,
  "model_version": "classical_logreg_v1",
  "date": "2026-01-05"
}
```

The reason cites the actual numbers the model looked at, so a human can check the logic.

## The three tools

| Tool             | What you send                                     | What you get                                             |
| ---------------- | ------------------------------------------------- | -------------------------------------------------------- |
| `classify_day`   | One day of 92/96/100 kWh values + meter reference | Label, reason, confidence, abstained flag, model version |
| `classify_batch` | Up to 31 days in one call                         | Same output per day, in order                            |
| `model_info`     | Nothing                                           | Version, eval numbers, feature list, input limits        |

## How it was built

The server has four layers, each with one job:

```bash
src/meter_mcp/
  classifier.py     The brain: loads the frozen model, computes 26 evidence
                    features from raw values, applies the argmax-unsure rule.
                    No MCP dependency -- pure Python + sklearn.

  audit.py          The notebook: appends one JSONL line per call with
                    timestamps, labels, and abstention flags. Never stores
                    raw meter values, so audit files stay small and private.

  server_stdio.py   The STDIO wrapper: exposes the three tools over
                    standard input/output for Claude Desktop, Claude Code,
                    and MCP Inspector. Adds typed inputs and audit logging.

  server_http.py    The HTTP wrapper: same three tools over Streamable HTTP
                    for shared or remote use. Adds API key auth and a
                    public /healthz endpoint.
```

**Key design choice:** the classifier is a pure library with no MCP dependency. The servers are thin wrappers that add typed inputs, audit logging, and transport. This means the same model logic runs identically over STDIO and HTTP -- there is only one implementation.

### The 26 evidence features

The model does not look at raw numbers directly. The classifier first computes 26 summary features that capture the shape of the day:

- **Missing data** -- how many intervals are missing, what percentage
- **Zero analysis** -- how many zeros, the longest run of zeros, trailing zeros
- **Positive stats** -- p10, median, p95, max, min of non-zero values
- **Ratios** -- flatness (median/p95), floor-to-reference, low-positive share
- **Time buckets** -- weekday night, weekday day, weekend (each with low-positive share and median)
- **Context** -- meter reference, low threshold (35% of reference), borderline-floor flag, weekend flag

These features are the same ones used in the sibling model repo. They were chosen because they capture the difference between "active", "standby", and "off" without needing the model to learn from raw sequences.

### The abstention rule

The model outputs probabilities for all four labels. The server applies a simple rule:

1. If the top prediction is `unsure`, the answer is `unsure`.
2. If the margin between the top two predictions is below a threshold, the answer is `unsure`.
3. Otherwise, the answer is the top prediction.

The threshold is frozen at 0.0, meaning only the first clause fires. This was calibrated on a sealed 30-day eval where the model declined 36.7% of days at precision 0.727 -- it would rather say "unsure" than make a confident error.

## Setup

```bash
pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/src"
python3 -m pytest -q
bash reproduce.sh
```

`reproduce.sh` runs compile checks, contract tests, a STDIO tool list, an artifact freeze check, and an HTTP health/auth check. It exits 0 only if everything passes.

## Running the server

### STDIO (Claude Desktop, Claude Code, MCP Inspector)

```bash
export PYTHONPATH="$PWD/src"
python3 -m meter_mcp.server_stdio
```

The server reads JSON-RPC messages from stdin and writes responses to stdout. Claude Desktop and Claude Code discover it as an MCP server automatically.

### HTTP (shared or remote use)

```bash
export MCP_API_KEY='replace-with-a-long-random-key'
python3 -m meter_mcp.server_http --port 8000
curl -s localhost:8000/healthz
```

The HTTP server adds two things on top of STDIO:

- **`/healthz`** -- a public liveness probe that returns status, model version, and tool list. No auth needed.
- **`/mcp`** -- the MCP protocol endpoint. Requires `X-API-Key` or `Authorization: Bearer` header. Without a key, it returns 401.

The API key is a shared secret, not per-user identity. It keeps casual traffic out; it does not attribute calls to individual users.

### Docker / Apple containers

```bash
# Docker
docker build -t meter-mcp .
docker run -d --name meter-mcp -p 8000:8000 -e MCP_API_KEY="$MCP_API_KEY" meter-mcp

# Apple containers (Davit, no Docker)
container build -t meter-mcp .
container run -d --name meter-mcp -p 8000:8000 -e MCP_API_KEY="$MCP_API_KEY" meter-mcp
```

The Dockerfile uses `python:3.12-slim`, runs as non-root, and mounts a writable volume for audit logs. The health check hits `/healthz` every 30 seconds.

## Input contract

| Parameter             | Type                  | Constraint                                                     |
| --------------------- | --------------------- | -------------------------------------------------------------- |
| `values`              | list of float or null | 92, 96, or 100 values (must match the Europe/Berlin local day) |
| `meter_reference_kwh` | float                 | 0.1 to 150.0 (the meter's whole-period p95)                    |
| `date`                | string or null        | ISO `YYYY-MM-DD`; defaults to `2026-01-05` (a Monday)          |

Values are kWh per 15-minute interval. The server rejects inputs outside the published envelope instead of treating them as predictions. This prevents unit mistakes (sending Wh as kWh) from producing confident garbage.

## Eval

| System                                 | Accuracy          | Note                                                                                   |
| -------------------------------------- | ----------------- | -------------------------------------------------------------------------------------- |
| `rules_floor_v1` (deterministic rules) | 86.7%             | Pre-registered baseline from the sibling repo                                          |
| `classical_logreg_v1` (this server)    | **90.0%** (27/30) | Decline 36.7% at precision 0.727, validity 1.0, p50 0.77 ms/day, zero confident errors |

The eval is a sealed 30-day set from the sibling repo `TMFNK/meter-day-classifier`. The model was trained on synthetic data and frozen; no retraining happens in this repo.

**What the numbers do not show:** day labels are the benchmark, not a customer savings claim. The server states what a meter did; it does not calculate kWh savings. Segment triage and a real pilot come later.

## Limits, stated plainly

- **Day labels only.** This server classifies what a meter did. It does not recommend actions or calculate savings.
- **kWh, not Wh.** Inputs are kWh per 15-minute interval. Values outside 0.1-150 kWh reference or 0-1000 kWh per interval are rejected.
- **Europe/Berlin only.** The interval count must match the local day (92/96/100). Other timezones are not supported.
- **Frozen model.** The weights and threshold are vendored byte-identical from the sibling repo. No retraining, no online learning.
- **Template reasons.** The reason string cites printed numbers and nothing more. It is a communication artifact, not a second prediction.
- **Shared-key auth.** HTTP auth is a shared API key, not per-user identity. It keeps casual traffic out; it does not attribute calls.
- **Best-effort audit.** Audit write failures emit a warning and do not change the classification response. Deployments must mount the writable audit volume if they need persistence.

## Project layout

```text
src/meter_mcp/
  __init__.py        Version, model constants, label list
  classifier.py      Frozen model + 26-feature engineering + argmax-unsure rule
  audit.py           JSONL audit writer (summaries only, never raw values)
  server_stdio.py    FastMCP STDIO server (3 tools, typed inputs, audit)
  server_http.py     Streamable HTTP server (API key, /healthz, same tools)

artifacts/
  model.joblib           Frozen sklearn logreg (C=1.0), vendored from sibling
  operating_point.json   Frozen threshold, medians, feature list, CV tables

tests/
  test_contract.py       Contract tests + STDIO integration + HTTP auth tests

Dockerfile              python:3.12-slim, non-root, healthcheck, audit volume
docker-compose.yml      Local HTTP run with MCP_API_KEY
reproduce.sh            Compile + pytest + STDIO list + HTTP health/auth + artifact freeze
requirements.txt        Pinned runtime dependencies
requirements-dev.txt    Pinned test dependencies
docs/SOLUTION.md        Business framing, buyer personas, engagement shapes
```

## License

Apache-2.0, copyright 2026 MbitAI. See LICENSE and NOTICE.

Need this applied to your own meters? [MbitAI](https://www.mbitai.com)
