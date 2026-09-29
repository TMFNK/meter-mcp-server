# meter-mcp-server

Frozen meter-day classifier behind MCP. 90.0% eval vs 86.7% rules floor. Offline, small, and fast.

## What it is

One sklearn logreg (C=1.0, 26 evidence features) exposed as 3 MCP tools:

- `classify_day` - 96 x 15-min kWh plus meter reference returns label plus reason.
- `classify_batch` - up to 31 days.
- `model_info` - version, numbers, limits.

Labels: `active`, `standby`, `off`, `unsure`. `unsure` is a first-class abstention, never a forced call.

Status: Step 0 scaffold. `classify_*` are stubs until Step 1 vendors the frozen model.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python3 -m pytest -q
bash reproduce.sh
```

## STDIO (Claude Desktop, Code, Inspector)

```bash
export PYTHONPATH="$PWD/src"
python3 -m meter_mcp.server_stdio
```

## HTTP + Docker

```bash
MCP_API_KEY=test python3 -m meter_mcp.server_http --port 8000
curl -s localhost:8000/healthz
docker build -t meter-mcp .
docker run -p 8000:8000 -e MCP_API_KEY=test meter-mcp
```

## Eval

Sealed 30 day eval from the sibling repo: 90.0% (27/30), decline 36.7% at precision 0.727, validity 1.0, p50 0.77 ms per day, zero confident errors. Sibling: https://github.com/TMFNK/meter-day-classifier

What the numbers do not show: day labels are the benchmark, not a customer savings claim. Segment triage plus a real pilot come later.

## Layout

See `2026-09-29-implementation-plan` in the vault (`20_Projects/40_mbitai-Meter-MCP`).

## License

Apache-2.0. See LICENSE and NOTICE. MbitAI, Munich.
