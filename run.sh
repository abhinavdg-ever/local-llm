#!/bin/bash

# Script to run the AI Sleep Coach Database Query Service

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the FastAPI application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

