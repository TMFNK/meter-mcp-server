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

echo "reproduce: OK"
