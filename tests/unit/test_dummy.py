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
"""
You can add your unit tests here.
This is where you test your business logic, including agent functionality,
data processing, and other core components of your application.
"""


from unittest.mock import MagicMock, patch

import pandas as pd

from app.agent import (
    get_stock_history,
    parallel_research_agent,
    report_synthesizer_agent,
    root_agent,
    stock_history_agent,
    stock_research_pipeline,
    web_research_agent,
)


def test_get_stock_history_success() -> None:
    """Tests that get_stock_history formats OHLCV records properly."""
    mock_df = pd.DataFrame(
        {
            "Open": [150.0],
            "High": [155.0],
            "Low": [149.0],
            "Close": [154.0],
            "Volume": [1000000],
        },
        index=pd.Index(pd.to_datetime(["2026-01-02"]), name="Date"),
    )

    with patch("app.agent.yf.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_history("googl", period="5d", interval="1d")

    assert result["status"] == "success"
    assert result["ticker"] == "GOOGL"
    assert len(result["data"]) == 1
    assert result["data"][0] == {
        "Date": "2026-01-02",
        "Open": 150.0,
        "High": 155.0,
        "Low": 149.0,
        "Close": 154.0,
        "Volume": 1000000,
    }


def test_get_stock_history_empty() -> None:
    """Tests that get_stock_history returns an error dict when no data is found."""
    with patch("app.agent.yf.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_history("INVALID")

    assert result["status"] == "error"
    assert "INVALID" in result["message"]


def test_agent_hierarchy() -> None:
    """Tests that root_agent and sub-agents are wired as expected."""
    assert root_agent.name == "stock_reasearch"
    assert root_agent.sub_agents == [stock_research_pipeline]
    assert stock_research_pipeline.sub_agents == [
        parallel_research_agent,
        report_synthesizer_agent,
    ]
    assert parallel_research_agent.sub_agents == [
        web_research_agent,
        stock_history_agent,
    ]
    assert web_research_agent.output_key == "web_research"
    assert stock_history_agent.output_key == "stock_history"

