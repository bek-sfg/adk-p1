# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured JSON logging, PII redaction, and intent/outcome lifecycle logs.

- ``JsonFormatter`` emits one JSON object per line using Cloud Logging's
  structured-logging fields (``severity``, ``message``,
  ``logging.googleapis.com/trace``/``spanId``) so logs are correlated with the
  OpenTelemetry traces ADK already exports.
- ``redact_pii`` scrubs emails, phone numbers, SSNs, card numbers, IPs, and
  secrets from every log message and structured field before it is written.
- ``StructuredLoggingPlugin`` is an ADK plugin that logs an ``*.intent`` event
  before each run/agent/model/tool step and a matching ``*.outcome`` event
  (status, duration, token usage, errors) after it.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import TYPE_CHECKING, Any

from google.adk.plugins.base_plugin import BasePlugin
from opentelemetry import trace

if TYPE_CHECKING:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext
    from google.genai import types

LOGGER_NAME = "stock_research"
logger = logging.getLogger(LOGGER_NAME)

# Max characters of free text (prompts, responses) kept in a single log field.
MAX_TEXT_CHARS = 500

# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

# Matched against whole ``_``/``-``-separated key segments, so ``access_token``
# and ``api_key`` are redacted but metrics like ``prompt_tokens`` are not.
_SENSITIVE_KEYS = re.compile(
    r"(?:^|[_-])(?:pass(?:word)?|passwd|secret|token|api[_-]?key|authorization|"
    r"credentials?|ssn|social[_-]?security|credit[_-]?card|card[_-]?number|cvv|"
    r"iban|e?mail|email[_-]?address|phone(?:[_-]?number)?|address|dob|"
    r"date[_-]?of[_-]?birth)(?:$|[_-])",
    re.IGNORECASE,
)

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Secrets first, so their contents are not partially matched below.
    (re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED_API_KEY]"),
    (
        re.compile(
            r"(?i)\b(password|passwd|secret|token|api[_-]?key)\b(\s*[:=]\s*)\S+"
        ),
        r"\1\2[REDACTED_SECRET]",
    ),
    (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "[REDACTED_EMAIL]",
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (
        re.compile(
            r"(?<![\w.])(?:\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
        "[REDACTED_IP]",
    ),
]

# 13-19 digits, optionally grouped with spaces/dashes; validated with Luhn so
# large numeric market data (e.g. trading volumes) is not redacted.
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _redact_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_valid(digits):
        return "[REDACTED_CARD]"
    return match.group(0)


def redact_text(text: str) -> str:
    """Returns ``text`` with PII and secrets replaced by placeholders."""
    text = _CARD_CANDIDATE.sub(_redact_card, text)
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_pii(value: Any, _depth: int = 0) -> Any:
    """Recursively redacts PII from strings, dicts, lists, and tuples.

    Values stored under sensitive-looking keys (``password``, ``email``, ...)
    are replaced entirely, regardless of their content.
    """
    if _depth > 10:
        return "[TRUNCATED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if isinstance(k, str) and _SENSITIVE_KEYS.search(k)
            else redact_pii(v, _depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_pii(v, _depth + 1) for v in value]
    return value


def pseudonymize(value: str | None) -> str | None:
    """Stable, non-reversible identifier for correlating a user across logs."""
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _truncate(text: str | None, limit: int = MAX_TEXT_CHARS) -> str | None:
    if text is None:
        return None
    return (
        text if len(text) <= limit else f"{text[:limit]}...[+{len(text) - limit} chars]"
    )


def _content_text(content: types.Content | None) -> str | None:
    if not content or not content.parts:
        return None
    text = "".join(p.text for p in content.parts if getattr(p, "text", None))
    return text or None


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

_RESERVED_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
    | {"message", "asctime", "taskName"}
)


class JsonFormatter(logging.Formatter):
    """Formats records as single-line JSON compatible with Cloud Logging."""

    def __init__(self, project_id: str | None = None) -> None:
        super().__init__()
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
        }

        # Structured fields passed via ``extra=``.
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_ATTRS}
        payload.update(redact_pii(extras))

        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))

        # Correlate with the active OpenTelemetry span (Cloud Trace).
        span_ctx = trace.get_current_span().get_span_context()
        if span_ctx.is_valid:
            trace_id = format(span_ctx.trace_id, "032x")
            payload["logging.googleapis.com/trace"] = (
                f"projects/{self._project_id}/traces/{trace_id}"
                if self._project_id
                else trace_id
            )
            payload["logging.googleapis.com/spanId"] = format(span_ctx.span_id, "016x")
            payload["logging.googleapis.com/trace_sampled"] = (
                span_ctx.trace_flags.sampled
            )

        return json.dumps(payload, default=str, ensure_ascii=False)


_configured = False


def setup_logging(level: str | int | None = None) -> None:
    """Routes all Python logging to stdout as redacted, structured JSON.

    Idempotent. Level defaults to the ``LOG_LEVEL`` env var (``INFO``).
    """
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        if isinstance(existing.formatter, JsonFormatter):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level or os.environ.get("LOG_LEVEL", "INFO").upper())
    _configured = True


# ---------------------------------------------------------------------------
# Intent / outcome lifecycle plugin
# ---------------------------------------------------------------------------


def _log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Writes a lifecycle event; logging failures never break the agent."""
    try:
        phase = event.rsplit(".", 1)[-1]
        logger.log(
            level,
            event,
            extra={
                "event": event,
                "phase": phase,
                **{k: v for k, v in fields.items() if v is not None},
            },
        )
    except Exception:  # pragma: no cover - defensive
        pass


class StructuredLoggingPlugin(BasePlugin):
    """Emits paired ``*.intent`` / ``*.outcome`` logs for every ADK step.

    Intent logs record *what is about to happen* (user request, agent, model,
    tool + arguments). Outcome logs record *what actually happened* (status,
    duration, token usage, result summary, or error). All payloads are
    redacted by ``JsonFormatter`` before they are written.
    """

    def __init__(self, name: str = "structured_logging") -> None:
        super().__init__(name=name)
        self._starts: dict[tuple[str, ...], float] = {}

    # -- helpers -----------------------------------------------------------

    def _start(self, *key: str) -> None:
        self._starts[key] = time.perf_counter()

    def _elapsed_ms(self, *key: str) -> float | None:
        start = self._starts.pop(key, None)
        return None if start is None else round((time.perf_counter() - start) * 1000, 1)

    @staticmethod
    def _ctx_fields(ctx: CallbackContext | ToolContext) -> dict[str, Any]:
        session = getattr(ctx, "session", None)
        return {
            "invocation_id": ctx.invocation_id,
            "session_id": getattr(session, "id", None),
            "user_hash": pseudonymize(getattr(ctx, "user_id", None)),
            "agent": ctx.agent_name,
        }

    # -- run ---------------------------------------------------------------

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        self._start("run", invocation_context.invocation_id)
        _log_event(
            "run.intent",
            invocation_id=invocation_context.invocation_id,
            session_id=invocation_context.session.id,
            user_hash=pseudonymize(invocation_context.user_id),
            app_name=invocation_context.app_name,
            user_request=_truncate(_content_text(invocation_context.user_content)),
        )
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        _log_event(
            "run.outcome",
            invocation_id=invocation_context.invocation_id,
            session_id=invocation_context.session.id,
            status="success",
            duration_ms=self._elapsed_ms("run", invocation_context.invocation_id),
        )

    async def on_run_error_callback(
        self, *, invocation_context: InvocationContext, error: Exception
    ) -> None:
        _log_event(
            "run.outcome",
            logging.ERROR,
            invocation_id=invocation_context.invocation_id,
            session_id=invocation_context.session.id,
            status="error",
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=self._elapsed_ms("run", invocation_context.invocation_id),
        )

    # -- agent -------------------------------------------------------------

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        self._start("agent", callback_context.invocation_id, agent.name)
        _log_event(
            "agent.intent",
            **self._ctx_fields(callback_context),
            agent_type=type(agent).__name__,
            description=agent.description,
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        output_key = getattr(agent, "output_key", None)
        output = callback_context.state.get(output_key) if output_key else None
        _log_event(
            "agent.outcome",
            **self._ctx_fields(callback_context),
            status="success",
            duration_ms=self._elapsed_ms(
                "agent", callback_context.invocation_id, agent.name
            ),
            output_key=output_key,
            output_chars=len(output) if isinstance(output, str) else None,
            output_preview=_truncate(output) if isinstance(output, str) else None,
        )
        return None

    async def on_agent_error_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext, error: Exception
    ) -> None:
        _log_event(
            "agent.outcome",
            logging.ERROR,
            **self._ctx_fields(callback_context),
            status="error",
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=self._elapsed_ms(
                "agent", callback_context.invocation_id, agent.name
            ),
        )

    # -- model -------------------------------------------------------------

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        self._start(
            "model", callback_context.invocation_id, callback_context.agent_name
        )
        _log_event(
            "model.intent",
            logging.DEBUG,
            **self._ctx_fields(callback_context),
            model=llm_request.model,
            content_count=len(llm_request.contents or []),
            tools=sorted(llm_request.tools_dict.keys()) or None,
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.partial:
            return None
        usage = llm_response.usage_metadata
        function_calls = [
            p.function_call.name
            for p in (
                llm_response.content.parts
                if llm_response.content and llm_response.content.parts
                else []
            )
            if p.function_call
        ]
        failed = bool(llm_response.error_code)
        _log_event(
            "model.outcome",
            logging.WARNING if failed else logging.INFO,
            **self._ctx_fields(callback_context),
            status="error" if failed else "success",
            duration_ms=self._elapsed_ms(
                "model", callback_context.invocation_id, callback_context.agent_name
            ),
            model_version=llm_response.model_version,
            finish_reason=str(llm_response.finish_reason)
            if llm_response.finish_reason
            else None,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            function_calls=function_calls or None,
            error_code=llm_response.error_code,
            error=llm_response.error_message,
        )
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        _log_event(
            "model.outcome",
            logging.ERROR,
            **self._ctx_fields(callback_context),
            status="error",
            model=llm_request.model,
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=self._elapsed_ms(
                "model", callback_context.invocation_id, callback_context.agent_name
            ),
        )
        return None

    # -- tool --------------------------------------------------------------

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        self._start(
            "tool",
            tool_context.invocation_id,
            tool_context.function_call_id or tool.name,
        )
        _log_event(
            "tool.intent",
            **self._ctx_fields(tool_context),
            tool=tool.name,
            function_call_id=tool_context.function_call_id,
            tool_args=tool_args,
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        status = (
            result.get("status", "success") if isinstance(result, dict) else "success"
        )
        _log_event(
            "tool.outcome",
            logging.WARNING if status == "error" else logging.INFO,
            **self._ctx_fields(tool_context),
            tool=tool.name,
            function_call_id=tool_context.function_call_id,
            status=status,
            duration_ms=self._elapsed_ms(
                "tool",
                tool_context.invocation_id,
                tool_context.function_call_id or tool.name,
            ),
            result_message=result.get("message") if isinstance(result, dict) else None,
            result_records=len(result["data"])
            if isinstance(result, dict) and isinstance(result.get("data"), list)
            else None,
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        _log_event(
            "tool.outcome",
            logging.ERROR,
            **self._ctx_fields(tool_context),
            tool=tool.name,
            function_call_id=tool_context.function_call_id,
            status="error",
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=self._elapsed_ms(
                "tool",
                tool_context.invocation_id,
                tool_context.function_call_id or tool.name,
            ),
        )
        return None
