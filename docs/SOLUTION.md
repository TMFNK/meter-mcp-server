# SOLUTION (Step 0 stub, full text in Step 4)

Problem: industrial submeter data is messy, and standby waste hides in night and weekend bands. Cloud LLMs fail the requirement: data protection, per-token cost, auditability.

Deliverable: one local MCP server with 3 tools that classifies meter days, abstains as unsure with a reason, and writes an audit line per call. The client owns the code, the model file, and the docs.

Buyer: ops and energy roles with submeter exports who need reviewable state evidence before any savings talk.

Limits, stated plainly: day labels only, no kWh savings claim, template reasons only, 90.0% on a 30 day sealed eval with 3 over-abstentions on a low-reference meter.
