#!/usr/bin/env bash
# One-command public run: compile + contract tests + STDIO tool list.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$PWD/src"

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
if [ -f artifacts/model.joblib ]; then
  md5sum artifacts/* || md5 artifacts/*
else
  echo "artifacts not vendored yet (Step 1) - skipped"
fi

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
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST localhost:8137/mcp -H 'Content-Type: application/json' -H 'X-API-Key: test' -d '{}')
echo "status: $CODE"
[ "$CODE" != "401" ] || { echo "expected non-401 on /mcp with key"; exit 1; }
kill $SERVER_PID 2>/dev/null || true
trap - EXIT
wait $SERVER_PID 2>/dev/null || true
echo "http: OK - health 200, auth 401 on bad key"

echo "reproduce: OK"
