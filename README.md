# meter-mcp-server

Frozen meter-day classifier behind MCP. 90.0% eval vs 86.7% rules floor. Runs offline.

## What it is

One sklearn logreg (C=1.0, 26 evidence features) exposed as 3 MCP tools:

- `classify_day` - 96 x 15-min kWh plus meter reference returns label plus reason.
- `classify_batch` - up to 31 days.
- `model_info` - version, numbers, limits.

Labels: `active`, `standby`, `off`, `unsure`. `unsure` is a first-class abstention, never a forced call.

Status: Step 3 done. All three tools run over STDIO and Streamable HTTP.
`/healthz` is public; `/mcp` needs `MCP_API_KEY`. Verified with
`container build` + `container run` (Davit, Apple containers).

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

## HTTP + Davit (Apple containers, no Docker)

```bash
MCP_API_KEY=test python3 -m meter_mcp.server_http --port 8000
curl -s localhost:8000/healthz
container build -t meter-mcp .
container run -d --name meter-mcp -p 8000:8000 -e MCP_API_KEY=test meter-mcp
```

`container` is Apple's CLI (Davit installs it per-user, no admin needed;
Davit Settings can add it to your shell). Same checks: `curl
localhost:8000/healthz` is 200, `POST /mcp` without the key is 401. The
`Dockerfile` is the build input; `docker-compose.yml` can be opened with
Davit's compose import.

## Eval

Sealed 30 day eval from the sibling repo: 90.0% (27/30), decline 36.7% at precision 0.727, validity 1.0, p50 0.77 ms per day, zero confident errors. Sibling: https://github.com/TMFNK/meter-day-classifier

What the numbers do not show: day labels are the benchmark, not a customer savings claim. Segment triage plus a real pilot come later.

## Layout

```text
src/meter_mcp/     classifier.py (frozen model), audit.py, server_stdio.py, server_http.py
artifacts/         model.joblib + operating_point.json (vendored, frozen)
tests/             test_contract.py (contract + STDIO + HTTP tests)
Dockerfile         python:3.12-slim, non-root, serves HTTP on 8000
reproduce.sh       compile + pytest + STDIO list + HTTP health/auth
docs/SOLUTION.md   business framing (Step 4)
```

## License

Apache-2.0, copyright 2026 MbitAI. See LICENSE and NOTICE.

Need this applied to your own meters? [MbitAI](https://www.mbitai.com)
