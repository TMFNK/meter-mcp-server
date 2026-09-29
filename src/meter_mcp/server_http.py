#!/usr/bin/env python3
"""HTTP server: same 3 tools over Streamable HTTP plus /healthz.

Step 0: same stubs as STDIO. Step 3 adds MCP_API_KEY check and audit log.
"""

import os

from meter_mcp.server_stdio import mcp

app = mcp.streamable_http_app()


def main(port: int = 8000) -> None:
    api_key = os.environ.get("MCP_API_KEY", "")
    if not api_key:
        print("warning: MCP_API_KEY is empty, running without auth (local only)")
    mcp.run(transport="streamable-http", port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(args.port)
