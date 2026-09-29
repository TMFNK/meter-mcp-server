#!/usr/bin/env python3
"""HTTP server: same 3 tools over Streamable HTTP plus /healthz.

Step 3: Streamable HTTP via FastMCP, public /healthz, MCP_API_KEY check
on the MCP endpoint, audit logging via the shared STDIO tool impls
(classify_day_impl / classify_batch_impl already log; no extra wiring).

Auth contract: every request under the MCP path
(default /mcp) needs either `X-API-Key: <key>` or
`Authorization: Bearer <key>` (constant-time compare). Anything else
gets a clean 401 JSON with no traceback. /healthz stays public so
Docker / CI can probe it without a key. MCP_API_KEY is required; an empty
key fails closed.
"""

import argparse
import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from meter_mcp import MODEL_VERSION
from meter_mcp.server_stdio import mcp

HEALTHZ_PATH = "/healthz"


def _expected_key() -> str:
    return os.environ.get("MCP_API_KEY", "")


def _provided_key(request) -> str:
    if request.headers.get("x-api-key"):
        return request.headers["x-api-key"]
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer":
        return token.strip()
    return ""


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Gate the MCP endpoint on MCP_API_KEY, leave the rest public."""

    async def dispatch(self, request, call_next):
        mcp_path = mcp.settings.streamable_http_path
        if not request.url.path.startswith(mcp_path):
            return await call_next(request)
        expected = _expected_key()
        if not expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        provided = _provided_key(request)
        if provided and hmac.compare_digest(
            provided.encode("utf-8"), expected.encode("utf-8")
        ):
            return await call_next(request)
        return JSONResponse({"error": "unauthorized"}, status_code=401)


@mcp.custom_route(HEALTHZ_PATH, methods=["GET"])
async def healthz(request):  # noqa: ARG001 - Starlette route signature
    """Public liveness probe (no auth, no model load)."""
    return JSONResponse(
        {
            "status": "ok",
            "model_version": MODEL_VERSION,
            "tools": ["classify_day", "classify_batch", "model_info"],
        }
    )


def create_app():
    """Build the Streamable HTTP app with healthz and API key check."""
    app = mcp.streamable_http_app()
    app.add_middleware(ApiKeyMiddleware)
    return app


# Import-time app for `uvicorn meter_mcp.server_http:app` and tests.
app = create_app()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if not _expected_key():
        raise SystemExit("MCP_API_KEY must be set before starting the HTTP server")

    import uvicorn

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
