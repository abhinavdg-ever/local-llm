#!/bin/bash

# Start FastAPI backend in background on port 8001 (nginx proxies from 8000)
cd /app
uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1 &

# Start nginx in foreground on port 8000 (matches docker-compose.yml 8015:8000)
nginx -g "daemon off;"

