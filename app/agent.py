# ruff: noqa
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

import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import google_search
from google.genai import types

from app.app_utils.observability import StructuredLoggingPlugin, setup_logging

setup_logging()

MODEL = "gemini-3.8-flash"
SYNTHESIZER_MODEL = "gemini-3.1-pro-preview"


def get_stock_history(
    ticker: str, period: str = "1mo", interval: str = "1d"
) -> dict:
    """Fetches historical stock price data (OHLCV) for a given ticker symbol.

    Args:
        ticker: The stock ticker symbol (e.g., 'GOOGL', 'AAPL').
        period: Data period to download (e.g., '5d', '1mo', '6mo', '1y', '5y').
        interval: Data interval (e.g., '1d', '1wk', '1mo').

    Returns:
        A dictionary containing the historical price records or an error message.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        return {
            "status": "error",
            "message": f"No historical data found for {ticker}.",
        }

    hist = hist.reset_index()
    date_col = "Date" if "Date" in hist.columns else "Datetime"
    hist[date_col] = hist[date_col].dt.strftime("%Y-%m-%d")
    records = hist[
        [date_col, "Open", "High", "Low", "Close", "Volume"]
    ].to_dict(orient="records")
    return {"status": "success", "ticker": ticker.upper(), "data": records}


web_research_agent = Agent(
    name="web_research_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Searches the web using Google Search to find current information, news, and market context.",
    instruction=(
        "You are a financial researcher. Use Google Search to gather recent news, "
        "financial performance, market sentiment, and risks to analyze whether "
        "the requested company is a good investment."
    ),
    tools=[google_search],
    output_key="web_research",
)


stock_history_agent = Agent(
    name="stock_history_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Retrieves historical stock price and volume data (OHLCV) for ticker symbols.",
    instruction=(
        "You are a quantitative stock market analyst. Identify the ticker symbol "
        "for the requested company, use the get_stock_history tool to retrieve its "
        "historical price and volume data, and analyze price trends, volatility, "
        "and trading volume to evaluate its recent market performance."
    ),
    tools=[get_stock_history],
    output_key="stock_history",
)


parallel_research_agent = ParallelAgent(
    name="parallel_research_agent",
    description="Runs web_research_agent and stock_history_agent in parallel.",
    sub_agents=[web_research_agent, stock_history_agent],
)


report_synthesizer_agent = Agent(
    name="report_synthesizer_agent",
    model=Gemini(
        model=SYNTHESIZER_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Combines results from web_research_agent and stock_history_agent into a comprehensive stock research report.",
    instruction=(
        "You are a senior investment analyst. Synthesize the qualitative web research "
        "findings and the quantitative historical stock price analysis to determine "
        "whether the requested company is a good investment. Structure your report with:\n"
        "1. Executive Summary\n"
        "2. Recent News & Market Sentiment\n"
        "3. Historical Price & Volume Performance\n"
        "4. Key Opportunities & Risks\n"
        "5. Investment Outlook & Conclusion\n\n"
        "Web Research Findings:\n{web_research?}\n\n"
        "Historical Stock Data:\n{stock_history?}"
    ),
)


stock_research_pipeline = SequentialAgent(
    name="stock_research_pipeline",
    description="Researches a specific company's stock in parallel using web search and historical data, then synthesizes a final investment report.",
    sub_agents=[parallel_research_agent, report_synthesizer_agent],
)


root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name. Renaming the agent only here makes the two disagree,
    # and anything selecting traces by name stops finding this agent's.
    name="stock_reasearch",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Gatekeeper agent that checks whether the user request is about financial analysis of a company.",
    instruction=(
        "You are a gatekeeper for a financial stock research system. "
        "Determine whether the user's request is asking for financial or investment "
        "analysis of a specific company or stock.\n"
        "- If the request IS about financial or investment analysis of a company/stock, "
        "transfer the request to `stock_research_pipeline`.\n"
        "- If the request is NOT about financial analysis of a company, politely ask "
        "the user to stay on topic and provide a company or stock ticker they would "
        "like analyzed. Do NOT invoke or transfer to any other agent."
    ),
    sub_agents=[stock_research_pipeline],
)

app = App(
    root_agent=root_agent,
    name="app",
    plugins=[StructuredLoggingPlugin()],
)
