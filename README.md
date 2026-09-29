# meter-mcp-server

Frozen meter-day classifier behind MCP. 90.0% eval vs 86.7% rules floor. Runs offline.

## What it is

One sklearn logreg (C=1.0, 26 evidence features) exposed as 3 MCP tools:

- `classify_day` - 92, 96, or 100 x 15-min kWh plus meter reference returns label plus reason.
- `classify_batch` - up to 31 days.
- `model_info` - version, numbers, limits.

Labels: `active`, `standby`, `off`, `unsure`. `unsure` is a first-class abstention, never a forced call.

Status: hardened v1. All three tools run over STDIO and Streamable HTTP.
`/healthz` is public; `/mcp` fails closed unless `MCP_API_KEY` is set.

## Setup

```bash
pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/src"
python3 -m pytest -q
bash reproduce.sh
```

## STDIO (Claude Desktop, Code, Inspector)

```bash
export PYTHONPATH="$PWD/src"
python3 -m meter_mcp.server_stdio
```

## HTTP + Davit (Apple containers, no Docker)

```bash
export MCP_API_KEY='replace-with-a-long-random-key'
python3 -m meter_mcp.server_http --port 8000
curl -s localhost:8000/healthz
container build -t meter-mcp .
container run -d --name meter-mcp -p 8000:8000 \
  -e MCP_API_KEY="$MCP_API_KEY" meter-mcp
```

`container` is Apple's CLI (Davit installs it per-user, no admin needed;
Davit Settings can add it to your shell). Same checks: `curl
localhost:8000/healthz` is 200, `POST /mcp` without the key is 401, and
the authenticated MCP handshake returns the three tools. The container
writes JSONL audit records to its named audit volume. The `Dockerfile` is
the build input; `docker-compose.yml` requires `MCP_API_KEY` explicitly.

## Input limits

Values are kWh per 15-minute interval, and `meter_reference_kwh` is a
positive p95 in the same unit. The frozen model accepts local
Europe/Berlin days with 92 intervals (spring forward), 96 intervals, or
100 intervals (fall back). The published support envelope is
`0.1 <= meter_reference_kwh <= 150` and each interval is at most 1,000
kWh; inputs outside it are rejected instead of being treated as a
prediction. If `date` is omitted, the response includes the assumed date
`2026-01-05` (a Monday).

## Eval

| System | Accuracy | Note |
| --- | --- | --- |
| `rules_floor_v1` (deterministic floor) | 86.7% | pre-registered baseline from the sibling repo |
| `classical_logreg_v1` (this server) | **90.0%** (27/30) | decline 36.7% at precision 0.727, validity 1.0, p50 0.77 ms/day, zero confident errors |

Sealed 30 day eval from the sibling repo. Full scoreboard and per-day
predictions: https://github.com/TMFNK/meter-day-classifier

What the numbers do not show: day labels are the benchmark, not a customer savings claim. Segment triage plus a real pilot come later.

## Layout

```text
src/meter_mcp/     classifier.py (frozen model), audit.py, server_stdio.py, server_http.py
artifacts/         model.joblib + operating_point.json (vendored, frozen)
tests/             test_contract.py (contract + STDIO + HTTP tests)
Dockerfile         python:3.12-slim, non-root, writable audit volume, healthcheck
reproduce.sh       compile + pytest + STDIO list + HTTP health/auth
requirements-dev.txt  pinned test dependencies
docs/SOLUTION.md   business framing and limits
```

## License

Apache-2.0, copyright 2026 MbitAI. See LICENSE and NOTICE.

Need this applied to your own meters? [MbitAI](https://www.mbitai.com)
