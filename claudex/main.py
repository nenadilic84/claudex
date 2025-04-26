import uvicorn
import os
from .proxy import app

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the claudex proxy (Anthropic-to-OpenAI or OpenRouter compatible proxy)")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8082, help='Port to bind (default: 8082)')
    parser.add_argument('--reload', action='store_true', help='Enable live-reload (for development)')
    parser.add_argument('--target-api-base', help='Base URL for target OpenAI-compatible API (default: from TARGET_API_BASE env)')
    parser.add_argument('--big-model', help='Target model for Claude Sonnet/larger models (default: from BIG_MODEL_TARGET env)')
    parser.add_argument('--small-model', help='Target model for Claude Haiku/smaller models (default: from SMALL_MODEL_TARGET env)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None, help='Logging level (default: from LOG_LEVEL env)')
    args = parser.parse_args()

    # Set environment variables from CLI arguments if provided
    if args.target_api_base:
        os.environ['TARGET_API_BASE'] = args.target_api_base
    if args.big_model:
        os.environ['BIG_MODEL_TARGET'] = args.big_model
    if args.small_model:
        os.environ['SMALL_MODEL_TARGET'] = args.small_model
    if args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level

    uvicorn.run("claudex.proxy:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
