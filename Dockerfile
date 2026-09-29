FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/README.md ./data/README.md

ENV PYTHONPATH=/app/src
ENV MCP_AUDIT_PATH=/var/lib/meter-mcp/audit/calls.jsonl
RUN mkdir -p /var/lib/meter-mcp/audit \
    && chown -R nobody:nogroup /var/lib/meter-mcp
VOLUME ["/var/lib/meter-mcp/audit"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
USER nobody
CMD ["python", "-m", "meter_mcp.server_http", "--host", "0.0.0.0", "--port", "8000"]
