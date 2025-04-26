#!/usr/bin/env python
import os
import unittest.mock
import requests

# Mock environment variables to allow proxy to initialize
with unittest.mock.patch.dict(os.environ, {
    'TARGET_API_BASE': 'https://api.example.com',
    'TARGET_API_KEY': 'mock-api-key',
    'BIG_MODEL_TARGET': 'model-large',
    'SMALL_MODEL_TARGET': 'model-small'
}):
    # Import claudex with mocked environment
    import uvicorn
    import claudex.proxy
    
    if __name__ == "__main__":
        # Start server in the background
        import threading
        import time
        
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=("claudex.proxy:app",),
            kwargs={"host": "127.0.0.1", "port": 8082, "log_level": "error"},
            daemon=True
        )
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Check health endpoint
        try:
            response = requests.get("http://127.0.0.1:8082/")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")