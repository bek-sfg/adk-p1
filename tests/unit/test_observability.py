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
"""Unit tests for structured logging, PII redaction, and lifecycle logs."""

import asyncio
import json
import logging
from types import SimpleNamespace

from app.agent import app as adk_app
from app.app_utils.observability import (
    JsonFormatter,
    StructuredLoggingPlugin,
    pseudonymize,
    redact_pii,
    redact_text,
)


def test_redact_text_scrubs_common_pii() -> None:
    text = (
        "Contact jane.doe@example.com or (555) 123-4567, SSN 123-45-6789, "
        "card 4111 1111 1111 1111, ip 10.0.0.1, Authorization: Bearer abc.def"
    )
    redacted = redact_text(text)
    for secret in [
        "jane.doe@example.com",
        "123-4567",
        "123-45-6789",
        "4111 1111 1111 1111",
        "10.0.0.1",
        "abc.def",
    ]:
        assert secret not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_CARD]" in redacted


def test_redact_text_keeps_market_data() -> None:
    text = "GOOGL closed at 154.25 on 2026-01-02 with volume 1234567890123"
    assert redact_text(text) == text


def test_redact_pii_sensitive_keys_and_nesting() -> None:
    data = {
        "ticker": "AAPL",
        "api_key": "xyz",
        "access_token": "abc",
        "prompt_tokens": 42,
        "nested": [{"email": "a@b.co"}],
    }
    assert redact_pii(data) == {
        "ticker": "AAPL",
        "api_key": "[REDACTED]",
        "access_token": "[REDACTED]",
        "prompt_tokens": 42,
        "nested": [{"email": "[REDACTED]"}],
    }


def test_pseudonymize_is_stable_and_opaque() -> None:
    assert pseudonymize("user-1") == pseudonymize("user-1")
    assert pseudonymize("user-1") != "user-1"
    assert pseudonymize(None) is None


def test_json_formatter_emits_redacted_structured_json() -> None:
    record = logging.LogRecord(
        "stock_research", logging.INFO, __file__, 1, "mail me at x@y.com", None, None
    )
    record.event = "tool.intent"
    record.tool_args = {"ticker": "GOOGL", "password": "hunter2"}

    payload = json.loads(JsonFormatter(project_id="p").format(record))

    assert payload["severity"] == "INFO"
    assert payload["message"] == "mail me at [REDACTED_EMAIL]"
    assert payload["event"] == "tool.intent"
    assert payload["tool_args"] == {"ticker": "GOOGL", "password": "[REDACTED]"}


def test_plugin_logs_tool_intent_and_outcome(caplog) -> None:
    plugin = StructuredLoggingPlugin()
    tool = SimpleNamespace(name="get_stock_history")
    ctx = SimpleNamespace(
        invocation_id="inv-1",
        function_call_id="fc-1",
        agent_name="stock_history_agent",
        user_id="user-1",
        session=SimpleNamespace(id="sess-1"),
    )

    with caplog.at_level(logging.INFO, logger="stock_research"):
        asyncio.run(
            plugin.before_tool_callback(
                tool=tool, tool_args={"ticker": "GOOGL"}, tool_context=ctx
            )
        )
        asyncio.run(
            plugin.after_tool_callback(
                tool=tool,
                tool_args={"ticker": "GOOGL"},
                tool_context=ctx,
                result={"status": "success", "data": [{}, {}]},
            )
        )

    intent, outcome = caplog.records
    assert intent.event == "tool.intent"
    assert intent.tool_args == {"ticker": "GOOGL"}
    assert intent.user_hash == pseudonymize("user-1")
    assert outcome.event == "tool.outcome"
    assert outcome.status == "success"
    assert outcome.result_records == 2
    assert outcome.duration_ms >= 0


def test_plugin_registered_on_app() -> None:
    assert any(isinstance(p, StructuredLoggingPlugin) for p in adk_app.plugins)
