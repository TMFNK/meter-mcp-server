"""MbitAI meter-mcp-server: frozen meter-day classifier behind MCP."""

__version__ = "0.1.0"
MODEL_VERSION = "classical_logreg_v1"
MODEL_EVAL_ACC = 0.90
RULES_FLOOR_ACC = 0.867
LABELS = ("active", "standby", "off", "unsure")
