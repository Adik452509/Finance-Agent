"""Shared result contract for every MCP tool.

Every tool returns a dict:
    success: {"ok": True, ...data}
    failure: {"ok": False, "error_type": <one of ErrorType>, "error": "<message for the agent>"}

Agents (and tests) can then handle every server the same way. The market and
filings servers already follow this shape; new servers use these helpers.
"""

import logging
from enum import Enum
from typing import Callable


class ErrorType(str, Enum):
    INVALID_INPUT = "invalid_input"      # bad arguments from the caller (the LLM) - fixable by retrying
    NOT_FOUND = "not_found"              # valid request, no data
    NOT_CONFIGURED = "not_configured"    # missing/invalid API key or setting
    RATE_LIMITED = "rate_limited"        # retry later (seconds/minutes)
    QUOTA_EXCEEDED = "quota_exceeded"    # retry much later (daily/plan limit)
    NOT_READY = "not_ready"              # e.g. empty index
    UPSTREAM_ERROR = "upstream_error"    # network / provider failure


ERROR_TYPES = {e.value for e in ErrorType}


class ToolInputError(ValueError):
    """Raise inside a tool for bad arguments; _safe() turns it into invalid_input."""


def ok(**data) -> dict:
    return {"ok": True, **data}


def fail(error_type: ErrorType | str, message: str, **extra) -> dict:
    return {"ok": False, "error_type": ErrorType(error_type).value, "error": message, **extra}


def safe_call(tool_name: str, compute: Callable[[], dict], logger: logging.Logger) -> dict:
    """Run a tool body; never let an exception escape to the MCP layer."""
    try:
        return compute()
    except ToolInputError as e:
        return fail(ErrorType.INVALID_INPUT, str(e))
    except Exception as e:  # unexpected: log with traceback (stderr), report briefly
        logger.exception("%s failed", tool_name)
        return fail(ErrorType.UPSTREAM_ERROR, f"{type(e).__name__}: {str(e)[:200]}")
