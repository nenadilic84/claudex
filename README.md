# claudex

A CLI proxy to run Claude API requests (Anthropic-style) against OpenAI-compatible LLM providers (like OpenRouter), either for local development, automation, or as a bridge to OpenAI tooling.

## Features

- FastAPI-based proxy for low-latency, robust relaying.
- Converts Anthropic Claude v3-style and Claude tool-calls API to OpenAI-compatible requests.
- Flexible environment variable configuration for provider settings.

## Requirements

- Python 3.8+
- [uvicorn](https://www.uvicorn.org/) for ASGI server
- FastAPI, httpx, python-dotenv, pydantic (see `pyproject.toml`)

## Setup

```bash
git clone <your-repo-url>
cd claudex
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env  # edit .env to fill in your API settings
```

Fill in your `.env` like:
```
TARGET_API_BASE=https://api.openrouter.ai/v1
TARGET_API_KEY=<your_provider_key>
BIG_MODEL_TARGET=openai/gpt-4.1
SMALL_MODEL_TARGET=openai/gpt-4.1-mini
LOG_LEVEL=INFO
```

## Usage

After setup and installing dependencies, you can run the proxy in either of these ways:

### 1. Recommended: Run via the CLI/main entrypoint

```bash
# Run as module:
python -m claudex --host 0.0.0.0 --port 8082 --reload

# Or (if installed as a script):
claudex --host 0.0.0.0 --port 8082 --reload
```

### 2. Alternative: Run directly with Uvicorn

```bash
uvicorn claudex.proxy:app --host 0.0.0.0 --port 8082 --reload
```

---

In a second terminal, you can now use the Claude CLI tool with this:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 DISABLE_PROMPT_CACHING=1 claude
```

---
