# Port Configuration Reference

This document outlines all port configurations across the Sleep Coach LLM application.

## Backend (FastAPI)

- **Development:** Port `8000`
  - `app.py`: Runs on port 8000
  - `run.sh`: Runs on port 8000
  - `docker-entrypoint.sh`: Runs on port 8000 (inside container)

## Frontend (Vite/React)

- **Development Server:** Port `3000`
  - `vite.config.ts`: Dev server on port 3000
  - Proxy to backend: `http://localhost:8000`

- **Preview Server:** Port `5173`
  - `package.json`: Preview/start scripts use port 5173
  - Used for testing production builds locally

## Docker Configuration

- **Host Mapping:** `8000:80`
  - External access: `http://localhost:8000`
  - Container internal: Port 80 (nginx)

- **Nginx:** Port `80` (inside container)
  - Serves frontend static files
  - Proxies API requests to FastAPI on `localhost:8000`

- **FastAPI:** Port `8000` (inside container)
  - Runs behind nginx
  - Accessible via nginx proxy

## External Services

- **Llama API:** Port `11434` (default)
  - Configured via `LLAMA_API_URL` env var
  - Default: `http://34.131.37.125:11434/api/generate`

- **Qdrant Vector DB:** Port `6333` (default)
  - Configured via `QDRANT_URL` env var
  - Default: `http://34.131.37.125:6333`

- **Embedding Service:** Port `8000` (default)
  - Configured via `EMBEDDING_API_URL` env var
  - Default: `http://34.131.37.125:8000/embed`

## Port Summary

| Service | Port | Notes |
|---------|------|-------|
| FastAPI (dev) | 8000 | Backend API |
| FastAPI (docker) | 8000 | Inside container, behind nginx |
| Nginx (docker) | 80 | Container internal |
| Docker Host | 8000 | External access |
| Vite Dev Server | 3000 | Frontend development |
| Vite Preview | 5173 | Frontend preview |
| Llama API | 11434 | External service |
| Qdrant | 6333 | External service |
| Embedding API | 8000 | External service |

## Running Locally

1. **Backend only:**
   ```bash
   python app.py
   # or
   ./run.sh
   # Access: http://localhost:8000
   ```

2. **Frontend only:**
   ```bash
   cd frontend
   npm run dev
   # Access: http://localhost:3000
   ```

3. **Full stack (Docker):**
   ```bash
   docker-compose up
   # Access: http://localhost:8000
   ```

