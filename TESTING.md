# Testing Guide

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file from template
cp env.example .env

# Edit .env with your credentials (already configured with your values)
```

### 2. Test Connections

```bash
# Test database and LLM connections
python test_connection.py
```

### 3. Start the Service

**Option A: Run directly**
```bash
python app.py
```

**Option B: Using uvicorn**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Using Docker**
```bash
docker-compose up --build
```

The service will be available at `http://localhost:8000`

## API Testing

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database_connected": true,
  "llm_available": true
}
```

### 2. Get Database Schema

```bash
curl http://localhost:8000/schema
```

### 3. Get Database Statistics

```bash
curl http://localhost:8000/stats
```

### 4. Get Sample Data

```bash
curl http://localhost:8000/sample?limit=3
```

### 5. Ask a Question (Natural Language Query)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many customers have High Risk insomnia classification?",
    "table_name": "ai_coach_modules_summary"
  }'
```

More example questions:
```bash
# Question about average scores
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average insomnia score?"}'

# Question about chronotypes
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers are Evening Types?"}'

# Question about risk levels
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "List all customers with High Risk insomnia classification"}'
```

### 6. Execute Raw SQL (SELECT only)

```bash
curl -X POST "http://localhost:8000/query/sql?query=SELECT COUNT(*) as count FROM ai_coach_modules_summary WHERE insomniaClassification = 'High Risk'"
```

## Interactive Testing (Swagger UI)

Visit `http://localhost:8000/docs` in your browser for an interactive API documentation and testing interface.

## Python Testing Script

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Ask a question
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "How many customers are in the database?",
        "table_name": "ai_coach_modules_summary"
    }
)
print(response.json())
```

## Troubleshooting

### Database Connection Failed
- Verify MySQL credentials in `.env`
- Check network connectivity to MySQL server (62.72.57.99)
- Ensure MySQL user has proper permissions

### LLM API Failed
- Verify Ollama API URL in `.env` (http://34.131.0.29:11434/api/generate)
- Check if Ollama service is running
- Test direct API call:
  ```bash
  curl -X POST http://34.131.0.29:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{"model": "llama2", "prompt": "Hello"}'
  ```

### Port Already in Use
- Change port in `app.py` or use: `uvicorn app:app --port 8001`
- Check what's using port 8000: `lsof -i :8000`

