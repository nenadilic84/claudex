# claudex

A CLI proxy to run Claude API requests (Anthropic-style) against OpenAI-compatible LLM providers (like OpenRouter), enabling Claude Code to work with any OpenAI-compatible endpoint.

## Requirements

- Python 3.8+
- [uvicorn](https://www.uvicorn.org/) for ASGI server
- FastAPI, httpx, python-dotenv, pydantic (see `pyproject.toml`)

## Setup

```bash
git clone https://github.com/nenadilic84/claudex.git
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
# Run as module with environment variables:
python -m claudex.main --host 0.0.0.0 --port 8082 --reload

# Or with command line arguments:
python -m claudex.main --target-api-base https://api.openrouter.ai/v1 --big-model openai/gpt-4.1 --small-model openai/gpt-4.1-mini --log-level INFO

# Or if installed as a script:
claudex --host 0.0.0.0 --port 8082 --target-api-base https://api.openrouter.ai/v1 --big-model openai/gpt-4.1 --small-model openai/gpt-4.1-mini
```

#### CLI Options

- `--host` - Host to bind (default: 0.0.0.0)
- `--port` - Port to bind (default: 8082)
- `--reload` - Enable live-reload for development
- `--target-api-base` - Base URL for target OpenAI-compatible API
- `--big-model` - Target model for Claude Sonnet/larger models
- `--small-model` - Target model for Claude Haiku/smaller models
- `--log-level` - Logging level (DEBUG, INFO, WARNING, ERROR)

Note: The API key must still be provided via the TARGET_API_KEY environment variable for security reasons.

### 2. Alternative: Run directly with Uvicorn

```bash
uvicorn claudex.proxy:app --host 0.0.0.0 --port 8082 --reload
```

### Using with Claude CLI

In a second terminal, you can now use the Claude CLI tool with this proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 DISABLE_PROMPT_CACHING=1 claude
```

This allows you to use Claude Code with any OpenAI-compatible LLM provider, such as:
- OpenRouter
- Together.ai
- Local LLM endpoints
- Any other OpenAI-compatible API

## API Endpoints

- `GET /` - Health check endpoint
- `POST /v1/messages` - Main endpoint that receives Anthropic API requests, converts them to OpenAI format, and returns converted responses

## Development

Run tests:
```bash
pytest
```

## License

MIT License - Copyright (c) 2025 nenadilic84