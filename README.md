# meter-mcp-server

A small server that answers one question about energy meters: **"What was
this meter doing on a given day?"** An AI assistant such as Claude can ask
it directly, get back one of four labels with a reason, and show that
answer to a person instead of guessing.

Headline result: **90.0% correct on a sealed 30-day test, against 86.7%
for a hand-written rule**, with an explicit "unsure" whenever the data is
too thin to call. Section 6 explains how small that gap really is.

## 1. The problem

Anyone can now ask an AI assistant "was this machine running last
Tuesday?" and get a fluent, confident answer. The assistant has no
built-in way to read a meter, so it will still produce an answer. That
answer may be wrong and look exactly like a right one, and someone may act
on it: send an engineer to a machine that was off, or ignore a compressor
that ran all night.

The safer setup is to let the assistant hand the question to a small
program that was built and tested for this one job, and to pass its answer
back unchanged. This project is that program. It keeps the checking
outside the assistant, where it can be measured and audited.

## 2. What it does

Energy meters record consumption every 15 minutes. A full day is 96
numbers (or 92 or 100 on the days the clocks change). The server takes
those numbers plus a meter reference value (the meter's typical high
reading) and classifies the day into one of four labels:

| Label     | What it means                              |
| --------- | ------------------------------------------ |
| `active`  | The meter was actively consuming energy    |
| `standby` | The meter was in a low-power standby state |
| `off`     | The meter was off (all zeros or near-zero) |
| `unsure`  | The model declines to make a forced call   |

`unsure` is a first-class answer, not a failure. When the evidence is
thin, the server says so instead of guessing.

### A concrete example

You send 96 numbers representing a Monday's 15-minute kWh readings, plus
the meter reference (the meter's typical p95 consumption; p95 is the level
95% of readings stay under, a way to ignore brief spikes). The server
replies with a structured answer like this (illustrative values):

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

The reason cites the actual numbers the model looked at, so a human can
check the logic.

### The three tools

A "tool" here is one action the assistant is allowed to ask the server to
perform.

| Tool             | What you send                                     | What you get                                             |
| ---------------- | ------------------------------------------------- | -------------------------------------------------------- |
| `classify_day`   | One day of 92/96/100 kWh values + meter reference | Label, reason, confidence, abstained flag, model version |
| `classify_batch` | Up to 31 days in one call                         | Same output per day, in order                            |
| `model_info`     | Nothing                                           | Version, eval numbers, feature list, input limits        |

## 3. How it works

### What is MCP?

MCP (Model Context Protocol) is a standard way for AI assistants to call
external tools. Think of it like a USB port for AI: any assistant that
speaks MCP can plug into any server that speaks MCP, without custom glue
code.

When an assistant uses a tool this way, it is doing **tool calling**: it
writes down which tool it wants and what to send, a program runs it, and
the result comes back for the assistant to use. The assistant does not
compute the answer itself. This project is one such server. It wraps a
frozen machine-learning model behind three MCP tools, so an assistant can
ask "was this meter active last Tuesday?" and get a structured answer with
a reason and a confidence score.

### The four layers

The server has four layers, each with one job:

```bash
src/meter_mcp/
  classifier.py     The brain: loads the frozen model, computes 26 evidence
                    features from raw values, applies the argmax-unsure rule.
                    No MCP dependency, pure Python + sklearn.

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

Two plain-language notes on that list. STDIO means the assistant and the
server talk through a local pipe on one computer. Streamable HTTP means
they talk over a network connection, so several people can share one
server. `sklearn` is scikit-learn, a standard library for classic
statistical models.

**Key design choice:** the classifier is a pure library with no MCP
dependency. The servers are thin wrappers that add typed inputs, audit
logging, and transport. This means the same model logic runs identically
over STDIO and HTTP: there is only one implementation.

### The 26 evidence features

The model does not look at raw numbers directly. The classifier first
computes 26 summary features (numbers that describe the shape of the day)
that capture how the day looked:

- **Missing data**: how many intervals are missing, what percentage
- **Zero analysis**: how many zeros, the longest run of zeros, trailing zeros
- **Positive stats**: p10, median, p95, max, min of non-zero values
- **Ratios**: flatness (median/p95), floor-to-reference, low-positive share
- **Time buckets**: weekday night, weekday day, weekend (each with low-positive share and median)
- **Context**: meter reference, low threshold (35% of reference), borderline-floor flag, weekend flag

These features are the same ones used in the sibling model repo. They were
chosen because they capture the difference between "active", "standby",
and "off" without needing the model to learn from raw sequences.

### The abstention rule

Abstention means declining to answer. The model outputs a probability for
each of the four labels. The server applies a simple rule:

1. If the top prediction is `unsure`, the answer is `unsure`.
2. If the margin between the top two predictions is below a threshold, the answer is `unsure`.
3. Otherwise, the answer is the top prediction.

The threshold is frozen at 0.0, meaning only the first clause fires. It
was set in the sibling repo on 30 human-labeled training days, not on the
sealed eval days. On the separate sealed eval the model declined 36.7% of
days at precision 0.727 (see Section 5): it would rather say "unsure" than
make a confident error.

### Guardrails on the input

A guardrail is a check that stops bad input before it reaches the model.

| Parameter             | Type                  | Constraint                                                     |
| --------------------- | --------------------- | -------------------------------------------------------------- |
| `values`              | list of float or null | 92, 96, or 100 values (must match the Europe/Berlin local day) |
| `meter_reference_kwh` | float                 | 0.1 to 150.0 (the meter's whole-period p95)                    |
| `date`                | string or null        | ISO `YYYY-MM-DD`; defaults to `2026-01-05` (a Monday)          |

Values are kWh per 15-minute interval. The server rejects inputs outside
the published envelope instead of treating them as predictions. This
prevents unit mistakes (sending Wh as kWh) from producing confident
garbage.

## 4. What this teaches you

This repo demonstrates **tool calling**: how an AI assistant hands a
question to a checked program and why that is more trustworthy than
letting the assistant answer alone. It also shows the abstention lesson
from the sibling classifier.

- **The assistant is only as good as the tool behind it.** The assistant
  decides what to ask. The server decides the answer, and gives the same
  answer every time for the same input.
- **A tool can refuse.** The server rejects inputs outside its limits and
  answers `unsure` when the evidence is thin. A tool that can say no is
  easier to trust than one that always produces something.
- **You can check the answer.** Every reply carries the numbers it was
  based on, and every call leaves an audit line, so a person can review
  what the assistant was told.

To see how the underlying model compares with a hand-written rule and with
a general-purpose AI model, read the sibling repo, which holds the tests.

## 5. Results

The numbers below are measured in the sibling repo
[`TMFNK/meter-day-classifier`](https://github.com/TMFNK/meter-day-classifier),
in its `outputs/results.csv`, on a sealed set of 30 human-labeled days.
"Sealed" means the model never saw these days and they were used only for
this one measurement. The model was trained on synthetic data and frozen;
no retraining happens in this repo.

| System                                 | Accuracy          | Note                                                                                   |
| -------------------------------------- | ----------------- | -------------------------------------------------------------------------------------- |
| `rules_floor_v1` (deterministic rules) | 86.7% (26/30)     | Pre-registered baseline from the sibling repo                                          |
| `classical_logreg_v1` (this server)    | **90.0%** (27/30) | Decline 36.7% at precision 0.727, validity 1.0, p50 0.77 ms/day, zero confident errors |

**What the terms mean.**

- **Rules floor.** A hand-written set of if-then rules, fixed before the
  model was tested. It is the score any learned model has to beat to
  justify existing.
- **Accuracy.** Share of the 30 sealed days labeled correctly.
- **Decline rate.** Share of days the model answered `unsure` instead of
  committing: 36.7%, which is 11 of 30 days.
- **Precision (of declines).** Of the days it declined, how many were
  genuinely ambiguous by the human labels: 0.727, which is 8 of 11.
- **Validity.** Share of replies that are a well-formed answer: 1.0.
- **p50 ms/day.** Median time to classify one day inside the model: 0.77
  milliseconds. The server adds transport time on top, which is not
  measured here.
- **Zero confident errors.** No day got a wrong `active`, `standby`, or
  `off` label. All 3 wrong days were `unsure` calls.

**What the gap means.** 3.3 points is one day: 27 correct against 26 out
of 30. Read it as "at least as good as the rules, with a measured way to
say I do not know", not as "clearly better". The value of this server is
making that model reachable from an assistant, with limits and an audit
trail, not a new accuracy record.

## 6. Where it fails

**What the numbers do not show:** day labels are the benchmark, not a
customer savings claim. The server states what a meter did; it does not
calculate kWh savings. Segment triage and a real pilot come later.

Limits, stated plainly:

- **The win over the rules is one day.** With 30 test days, a 3.3-point
  gap is within what a single relabeled day could erase.
- **It declines a lot.** 11 of 30 sealed days came back `unsure`, and 3 of
  those were days a person could label. Over a third of days would go to
  human review.
- **This repo tests the plumbing, not the accuracy.** The tests here check
  that inputs, outputs, limits, and authentication behave as documented.
  The accuracy numbers are copied from the sibling repo and were not
  re-measured here.
- **Day labels only.** This server classifies what a meter did. It does not recommend actions or calculate savings.
- **kWh, not Wh.** Inputs are kWh per 15-minute interval. Values outside 0.1-150 kWh reference or 0-1000 kWh per interval are rejected. The server can catch a unit slip that lands outside those ranges, but cannot tell whether the assistant sent the right meter's data.
- **Europe/Berlin only.** The interval count must match the local day (92/96/100). Other timezones are not supported.
- **Frozen model.** The weights and threshold are vendored (copied in unchanged) byte-identical from the sibling repo. No retraining, no online learning.
- **Template reasons.** The reason string cites printed numbers and nothing more. It is a communication artifact, not a second prediction.
- **Shared-key auth.** HTTP auth is a shared API key, not per-user identity. It keeps casual traffic out; it does not attribute calls to individual users.
- **Best-effort audit.** Audit write failures emit a warning and do not change the classification response. Deployments must mount the writable audit volume if they need persistence.

## 7. Try it yourself

**What you need.** Python (the Docker image uses 3.12) and the packages in
`requirements-dev.txt`. No GPU. The model is a small classic statistical
model, so an ordinary laptop is enough.

```bash
pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/src"
python3 -m pytest -q
bash reproduce.sh
```

`reproduce.sh` runs compile checks, contract tests, a STDIO tool list, an
artifact freeze check (confirming the model files are unchanged), and an
HTTP health/auth check. It exits 0 only if everything passes.

### Run the server over STDIO (Claude Desktop, Claude Code, MCP Inspector)

```bash
export PYTHONPATH="$PWD/src"
python3 -m meter_mcp.server_stdio
```

The server reads JSON-RPC messages (a simple request and response format)
from stdin and writes responses to stdout. Claude Desktop and Claude Code
discover it as an MCP server automatically.

### Run the server over HTTP (shared or remote use)

```bash
export MCP_API_KEY='replace-with-a-long-random-key'
python3 -m meter_mcp.server_http --port 8000
curl -s localhost:8000/healthz
```

The HTTP server adds two things on top of STDIO:

- **`/healthz`**: a public liveness probe that returns status, model version, and tool list. No auth needed.
- **`/mcp`**: the MCP protocol endpoint. Requires `X-API-Key` or `Authorization: Bearer` header. Without a key, it returns 401.

The API key is a shared secret, not per-user identity. It keeps casual
traffic out; it does not attribute calls to individual users.

### Docker / Apple containers

```bash
# Docker
docker build -t meter-mcp .
docker run -d --name meter-mcp -p 8000:8000 -e MCP_API_KEY="$MCP_API_KEY" meter-mcp

# Apple containers (Davit, no Docker)
container build -t meter-mcp .
container run -d --name meter-mcp -p 8000:8000 -e MCP_API_KEY="$MCP_API_KEY" meter-mcp
```

The Dockerfile uses `python:3.12-slim`, runs as non-root, and mounts a
writable volume for audit logs. The health check hits `/healthz` every 30
seconds.

### Project layout

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

## 8. Privacy and cost

- **The server itself sends nothing anywhere.** The model runs locally and
  needs no network for the classification. Over STDIO everything stays on
  one computer.
- **The assistant is a separate party.** If you connect a cloud assistant
  such as Claude, the meter values you ask about pass through that
  assistant and its provider. Check that this fits your data rules before
  connecting real meters.
- **The audit log keeps summaries, not raw values.** One line per call
  with timestamp, label, and abstention flag. Audit files are excluded
  from git.
- **No per-call cost from this server.** No tokens are used by the model.
  Any usage charges from the assistant you connect are separate.

## 9. Links

- Repo: [TMFNK/meter-mcp-server](https://github.com/TMFNK/meter-mcp-server)
- Sibling project (the classifier and its tests): [TMFNK/meter-day-classifier](https://github.com/TMFNK/meter-day-classifier)
- Business framing: [`docs/SOLUTION.md`](docs/SOLUTION.md)
- License: Apache-2.0, copyright 2026 MbitAI. See LICENSE and NOTICE.

Need this applied to your own meters? Email info@mbitai.com.
