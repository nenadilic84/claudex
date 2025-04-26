import uvicorn
from .proxy import app

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the claudex proxy (Anthropic-to-OpenAI or OpenRouter compatible proxy)")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8082, help='Port to bind (default: 8082)')
    parser.add_argument('--reload', action='store_true', help='Enable live-reload (for development)')
    args = parser.parse_args()

    uvicorn.run("claudex.proxy:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
