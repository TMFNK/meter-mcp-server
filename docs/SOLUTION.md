# Meter states inside agent workflows (MbitAI)

Who this is for, what it buys you, and how it is delivered. The technical
detail lives in `README.md` (tools, scores, layout) and `artifacts/README.md`
(model provenance). The classifier itself is documented in the sibling repo
(TMFNK/meter-day-classifier); this repo is only its deployment.

## The problem in one paragraph

Energy teams already have a classifier that turns 15-minute meter data into
reviewable day states. The next friction is plumbing: every agent, copilot,
or audit script that needs a state re-implements the feature math, the
threshold, and the reason format, and the copies drift. This server exists
so there is one place that answers "what was this meter doing that day",
over a protocol agents already speak, with the same frozen model and the
same audit line every time.

## What the server delivers

- **Three tools, one frozen model.** `classify_day` states a single day,
  `classify_batch` states up to 31, `model_info` reports the version, the
  numbers, and the limits. Every answer carries the model version, so a
  label from last quarter still traces to the weights that made it.
- **Abstention with a paper trail.** Thin-evidence days come back `unsure`
  with a one-line reason citing printed numbers, never a forced guess.
  Successful calls and classifier-level rejections append one JSONL audit
  line. Batch lines include per-day counts, dates, labels, and abstention
  flags, never the raw series, so a reviewer can replay the request shape
  and outcome.
- **Two transports, same answers.** STDIO for Claude Desktop, Code, and
  the Inspector; Streamable HTTP with a key for shared or remote use.
  `/healthz` stays public for probes; bad input returns a plain error,
  never a traceback.
- **Runs where the data already is.** Fully offline: scikit-learn on
  commodity hardware, no GPU, no tokens, no accounts. Ships as a small
  container (Apple containers via Davit, or Docker) with a writable audit
  volume and health check.

## Who buys it

| Buyer | Pain | Server answer |
| --- | --- | --- |
| Facility / energy manager | An agent drafts the weekly review but its meter claims cannot be checked | States + reasons + audit lines the reviewer opens next to the draft |
| Utility / metering MSP | Each customer agent re-implements scoring; versions drift across sites | One pinned endpoint per deployment; `model_info` names the version |
| ESCO / energy consultant | Audit needs proof of what the assistant saw, not just what it said | JSONL audit log with labels, abstentions, and model version per call |
| Compliance / CISO | Consumption data must not leave the building for a cloud model call | Offline inference; private labels and raw series never ship in the image |

## Engagement shapes (MbitAI)

1. **Assessment (fixed scope).** Your meter sample in, day states +
   reasons + scorecard out, under NDA, on your hardware or ours. Go/no-go
   on numbers, not slides.
2. **Pilot (your meters).** The server states a live feed next to your
   current rules; rescued days and decline-precision deltas are measured
   on your data with the same scorecard.
3. **Production + handover.** Pinned image for your meter park, API key
   and audit retention set from your policy, runbooks for re-state and
   review, team training on the reason format.

Contact: [https://www.mbitai.com](https://www.mbitai.com)

## Costs to budget (measured, not estimated)

| Item | Observed |
| --- | --- |
| Inference | 0.77 ms/day, $0 per 1,000 (no tokens, no server) |
| Image | python:3.12-slim + pinned deps, non-root, one port |
| Audit storage | One short JSON line per call, per-day batch outcomes, no raw series |
| License | Apache-2.0 (keeps your tree license-clean) |

## Limits, stated plainly

- Day labels only. This server states what a meter did; it makes no kWh
  savings claim. Segment triage and a real pilot come later.
- Inputs are kWh per 15-minute interval, not Wh. The frozen fit accepts
  `meter_reference_kwh` from 0.1 through 150 and interval values up to
  1,000 kWh. Values outside that envelope are rejected.
- Local Europe/Berlin days may contain 92, 96, or 100 intervals. The
  interval count must match the supplied date; omitting the date uses
  Monday `2026-01-05`, and the response states that effective date.
- The frozen model inherits the sibling's limits: 90.0% on a 30-day
  sealed eval, remaining errors are over-abstentions on near-noise-floor
  meters, and each meter needs its own whole-period p95 reference.
- Template reasons only. The reason cites the printed numbers and nothing
  more; it is a communication artefact, not a second prediction.
- HTTP auth is a shared key, not identity. It keeps casual traffic out;
  it does not attribute calls to users. The key is required for MCP
  traffic; `/healthz` remains public.
- Audit storage is best-effort for inference: write failures emit a
  warning and do not change the classification response. Deployments must
  mount the writable audit volume if they need persistence.

These are documented here because enterprise buyers should find limits
before the pilot does.
