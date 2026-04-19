from __future__ import annotations

import json
from typing import Any


def parse_tool_call_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Accept either a JSON string or a dict and return a kwargs dict."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
