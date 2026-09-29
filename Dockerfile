FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/README.md ./data/README.md

ENV PYTHONPATH=/app/src
EXPOSE 8000
USER nobody
CMD ["python", "-m", "meter_mcp.server_http", "--host", "0.0.0.0", "--port", "8000"]
