#!/usr/bin/env bash
# One-command public run: compile + tests + STDIO list + artifacts freeze + HTTP health/auth.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$PWD/src"

md5_value() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$@" | awk '{print $1}'
  else
    md5 -q "$@"
  fi
}

ARTIFACTS=(artifacts/model.joblib artifacts/operating_point.json)
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "artifact freeze check requires a git checkout" >&2
  exit 1
fi
for artifact in "${ARTIFACTS[@]}"; do
  [ -f "$artifact" ] || {
    echo "missing frozen artifact: $artifact" >&2
    exit 1
  }
done

echo "== compile =="
python3 -m py_compile src/meter_mcp/*.py tests/*.py

echo "== contract tests =="
python3 -m pytest -q

echo "== STDIO tool list =="
python3 - <<'PY'
from meter_mcp.server_stdio import mcp
import asyncio
tools = asyncio.run(mcp.list_tools())
names = sorted(t.name for t in tools)
print(names)
assert set(names) == {"classify_day", "classify_batch", "model_info"}, names
print("tools: OK - 3 tools")
PY

echo "== artifacts freeze check =="
for artifact in "${ARTIFACTS[@]}"; do
  expected="$(git show "HEAD:$artifact" | md5_value)"
  actual="$(md5_value "$artifact")"
  printf '%s  %s\n' "$actual" "$artifact"
  [ "$actual" = "$expected" ] || {
    echo "artifact changed from HEAD: $artifact" >&2
    exit 1
  }
done

echo "== HTTP health + auth =="
export MCP_API_KEY=test
python3 -m meter_mcp.server_http --port 8137 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 30); do
  if curl -sf localhost:8137/healthz >/dev/null 2>&1; then break; fi
  sleep 0.5
done
echo "-- /healthz (public, expect 200) --"
curl -sf localhost:8137/healthz
echo ""
echo "-- /mcp without key (expect 401) --"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST localhost:8137/mcp -H 'Content-Type: application/json' -d '{}')
echo "status: $CODE"
[ "$CODE" = "401" ] || { echo "expected 401 on /mcp without key, got $CODE"; exit 1; }
echo "-- /mcp with key (expect not 401) --"
INIT_HEADERS="$(mktemp)"
INIT_BODY="$(mktemp)"
CODE=$(curl --max-redirs 0 -sS -D "$INIT_HEADERS" -o "$INIT_BODY" -w "%{http_code}" \
  -X POST localhost:8137/mcp/ \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: test' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"reproduce","version":"1"}}}')
echo "status: $CODE"
[ "$CODE" = "200" ] || { echo "expected 200 from MCP initialize, got $CODE"; exit 1; }
SESSION="$(tr -d '\r' < "$INIT_HEADERS" | awk 'tolower($1) == "mcp-session-id:" {print $2; exit}')"
[ -n "$SESSION" ] || { echo "MCP initialize did not return a session"; exit 1; }
curl --max-redirs 0 -sS -o /dev/null -w "%{http_code}" \
  -X POST localhost:8137/mcp/ \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: test' \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' | grep -qx 202
TOOLS_BODY="$(mktemp)"
CODE=$(curl --max-redirs 0 -sS -D /dev/null -o "$TOOLS_BODY" -w "%{http_code}" \
  -X POST localhost:8137/mcp/ \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: test' \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}')
echo "tools/list status: $CODE"
[ "$CODE" = "200" ] || exit 1
grep -q '"classify_day"' "$TOOLS_BODY" || {
  echo "authenticated tools/list did not expose classify_day" >&2
  exit 1
}
kill $SERVER_PID 2>/dev/null || true
trap - EXIT
wait $SERVER_PID 2>/dev/null || true
echo "http: OK - health 200, auth 401 on bad key"

echo "== artifacts unchanged =="
for artifact in "${ARTIFACTS[@]}"; do
  expected="$(git show "HEAD:$artifact" | md5_value)"
  actual="$(md5_value "$artifact")"
  [ "$actual" = "$expected" ] || exit 1
done
rm -f "$INIT_HEADERS" "$INIT_BODY" "$TOOLS_BODY"
echo "artifacts: unchanged from HEAD"

echo "reproduce: OK"
