#!/usr/bin/env python
import os
import sys

# Get directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get project root directory (parent of script directory)
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
# Add project directory to path to ensure module can be found
sys.path.insert(0, project_dir)

# Set environment variables for testing
os.environ.setdefault("TARGET_API_BASE", "https://api.example.com")
os.environ.setdefault("TARGET_API_KEY", "mock-api-key") 
os.environ.setdefault("BIG_MODEL_TARGET", "model-large")
os.environ.setdefault("SMALL_MODEL_TARGET", "model-small")

# Now import and run
from claudex.main import main

if __name__ == "__main__":
    main()