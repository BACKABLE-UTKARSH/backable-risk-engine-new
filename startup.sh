#!/bin/bash

# Install dependencies if not already installed
if [ ! -d "antenv" ]; then
    echo "Creating virtual environment..."
    python -m venv antenv
    source antenv/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source antenv/bin/activate
fi

# Start the application
gunicorn -w 4 -k uvicorn.workers.UvicornWorker "BACKABLE NEW INFRASTRUCTURE RISK ENGINE:app" --bind 0.0.0.0:8000 --timeout 3600 --access-logfile '-' --error-logfile '-' --log-level info
