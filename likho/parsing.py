"""Defensive parsing of Param-1 output (docs/MODELS.md §3.4).

Param-1 is an early SFT checkpoint: it pads with preamble, wraps JSON in code
fences, or ignores the format entirely. Every parser here returns None rather
than raising, and every caller has a fallback.
"""

import json
import re

_JSON_OBJECT = re.compile(r"\{[\s\S]*\}")


def extract_json_object(text: str) -> dict | None:
    """The first-to-last-brace span of `text`, if it parses as a JSON object."""
    match = _JSON_OBJECT.search(text or "")
    if not match:
        return None
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
