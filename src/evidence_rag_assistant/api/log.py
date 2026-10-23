"""Structured logging: one JSON object per line on standard output.

Cloud Logging parses such lines into searchable fields and reads the log level
from "severity". Locally, the same lines are readable in the terminal.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Extra fields that may be attached to a log record and should be emitted.
_FIELDS = ("request_id", "method", "path", "status", "duration_ms", "answer_status")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _FIELDS:
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
