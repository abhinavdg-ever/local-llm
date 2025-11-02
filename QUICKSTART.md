# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Create .env File

```bash
cp env.example .env
```

The `.env` file should contain:
```
MYSQL_HOST=62.72.57.99
MYSQL_USER=aabo
MYSQL_PASSWORD=3#hxFkBFKJ2Ph!$@
MYSQL_DATABASE=aaboRing10Jan
OLLAMA_API_URL=http://34.131.0.29:11434/api/generate
```

## Step 3: Test Connections (Optional)

```bash
python test_connection.py
```

This will verify that both MySQL and LLM services are accessible.

## Step 4: Run the Application

**Main file to run: `app.py`**

### Option 1: Run directly with Python
```bash
python app.py
```

### Option 2: Run with uvicorn (recommended)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Run with Docker
```bash
docker-compose up --build
```

## Step 5: Access the Service

Once running, the service will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Quick Test

Open a new terminal and run:
```bash
curl http://localhost:8000/health
```

Or visit http://localhost:8000/docs in your browser for interactive testing.

## Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers are in the database?"}'
```

## Summary

- **Main file**: `app.py`
- **Run command**: `python app.py` or `uvicorn app:app --host 0.0.0.0 --port 8000`
- **Port**: 8000 (default)
- **Test file**: `test_connection.py` (optional, for testing connections)

