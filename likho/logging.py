"""Logging setup. Use `get_logger(__name__)` everywhere; never `print`.

NFR4: log lengths and timings, never user text.
"""

import logging

from likho.config import settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root = logging.getLogger("likho")
        root.addHandler(handler)
        root.setLevel(settings.log_level.upper())
        root.propagate = False
        _configured = True
    if not name.startswith("likho"):
        name = f"likho.{name}"
    return logging.getLogger(name)
