# stock-reasearch

Multi-agent financial stock research system built with the Google Agent Development Kit (ADK).
Agent generated with `agents-cli` version `1.8.0`

## Agent Architecture

Defined in `app/agent.py` using `gemini-3.8-flash`:

```
root_agent (stock_reasearch) — Gatekeeper
└── stock_research_pipeline (SequentialAgent)
    ├── parallel_research_agent (ParallelAgent)
    │   ├── web_research_agent (google_search -> output_key: "web_research")
    │   └── stock_history_agent (get_stock_history -> output_key: "stock_history")
    └── report_synthesizer_agent (Synthesizes final investment report)
```

### Agents & Tools

- **`root_agent` (`stock_reasearch`)**: Gatekeeper agent that checks whether the user request is asking for financial or investment analysis of a specific company or stock. Transfers valid requests to `stock_research_pipeline` and politely asks off-topic users to stay on topic.
- **`stock_research_pipeline` (`SequentialAgent`)**: Executes the parallel research phase followed by report synthesis.
- **`parallel_research_agent` (`ParallelAgent`)**: Runs `web_research_agent` and `stock_history_agent` concurrently:
  - **`web_research_agent`**: Financial researcher that uses `google_search` to gather recent news, financial performance, market sentiment, and risks (saved to `web_research`).
  - **`stock_history_agent`**: Quantitative analyst that uses the `get_stock_history` tool (`yfinance` OHLCV data) to analyze historical price trends, volatility, and trading volume (saved to `stock_history`).
- **`report_synthesizer_agent`**: Senior investment analyst that synthesizes `{web_research?}` and `{stock_history?}` into a structured 5-section report:
  1. Executive Summary
  2. Recent News & Market Sentiment
  3. Historical Price & Volume Performance
  4. Key Opportunities & Risks
  5. Investment Outlook & Conclusion

## Project Structure

```
stock-reasearch/
├── app/         # Core agent code
│   ├── agent.py               # Main multi-agent stock research pipeline
│   ├── fast_api_app.py        # FastAPI Backend server
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, eval, and load tests
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Antigravity CLI](https://antigravity.google/) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install `agents-cli` and its skills if not already installed:

```bash
uvx google-agents-cli setup
```

Install required packages:

```bash
agents-cli install
```

Test the agent with a local web server:

```bash
agents-cli playground
```

You can also use features from the [ADK](https://adk.dev/) CLI with `uv run adk`.

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        |
| [A2A Inspector](https://github.com/a2aproject/a2a-inspector) | Launch A2A Protocol Inspector                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use the [A2A Inspector](https://github.com/a2aproject/a2a-inspector) to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.
