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

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


def test_gatekeeper_rejects_off_topic() -> None:
    """
    Tests that the gatekeeper root_agent asks the user to stay on topic
    and does not invoke the research pipeline for off-topic queries.
    """
    session_service = InMemorySessionService()

    session = session_service.create_session_sync(
        user_id="test_user", app_name="test"
    )
    runner = Runner(
        agent=root_agent, session_service=session_service, app_name="test"
    )

    message = types.Content(
        role="user", parts=[types.Part.from_text(text="Why is the sky blue?")]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one message"

    has_text_content = any(
        event.content
        and event.content.parts
        and any(part.text for part in event.content.parts)
        for event in events
    )
    assert has_text_content, "Expected at least one message with text content"

    authors = {event.author for event in events if event.author}
    assert "web_research_agent" not in authors
    assert "stock_history_agent" not in authors
    assert "report_synthesizer_agent" not in authors


def test_agent_stream_stock_analysis() -> None:
    """
    Integration test for the full stock research pipeline.
    Tests that an on-topic company analysis request runs web_research_agent
    and stock_history_agent in parallel and synthesizes a final report.
    """
    session_service = InMemorySessionService()

    session = session_service.create_session_sync(
        user_id="test_user", app_name="test"
    )
    runner = Runner(
        agent=root_agent, session_service=session_service, app_name="test"
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="Analyze whether Apple (AAPL) is a good investment."
            )
        ],
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one message"

    authors = {event.author for event in events if event.author}
    assert "web_research_agent" in authors
    assert "stock_history_agent" in authors
    assert "report_synthesizer_agent" in authors

    synthesizer_has_text = any(
        event.author == "report_synthesizer_agent"
        and event.content
        and event.content.parts
        and any(part.text for part in event.content.parts)
        for event in events
    )
    assert synthesizer_has_text, (
        "Expected report_synthesizer_agent to produce a final report with text content"
    )

