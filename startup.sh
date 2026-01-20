#!/bin/bash
# Start the Risk Engine with environment-based port (defaults to 8000 for Azure)
python -m uvicorn "BACKABLE NEW INFRASTRUCTURE RISK ENGINE:app" --host 0.0.0.0 --port ${PORT:-8000}
