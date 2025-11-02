# AI Sleep Coach Database Query Service

An AI-powered service that answers natural language questions about sleep and health data stored in a MySQL database using a locally hosted LLM.

## Features

- 🤖 Natural language query interface using LLM
- 🗄️ MySQL database integration
- 📊 Schema introspection and sample data retrieval
- 🔍 Health check endpoints
- 🚀 FastAPI REST API
- 📝 Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- MySQL database with the sleep coach data
- Ollama LLM service running at the configured endpoint
- Network access to both MySQL and Ollama services

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd LocalLLM
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` with your actual credentials:
   ```
   MYSQL_HOST=your_mysql_host
   MYSQL_USER=your_mysql_user
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_DATABASE=your_database_name
   OLLAMA_API_URL=http://your_ollama_host:11434/api/generate
   ```

## Usage

### Start the Service

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
GET /health
```
Returns the health status of the service and its dependencies.

#### Root
```bash
GET /
```
Returns API information and available endpoints.

#### Get Database Schema
```bash
GET /schema?table_name=ai_coach_modules_summary
```
Returns the schema of the specified table.

#### Get Database Statistics
```bash
GET /stats?table_name=ai_coach_modules_summary
```
Returns statistics about the specified table (row count, columns, etc.).

#### Get Sample Data
```bash
GET /sample?table_name=ai_coach_modules_summary&limit=5
```
Returns sample rows from the specified table.

#### Natural Language Query
```bash
POST /query
Content-Type: application/json

{
  "question": "What is the average insomnia score for customers with High Risk classification?",
  "table_name": "ai_coach_modules_summary"
}
```

#### Execute SQL Query
```bash
POST /query/sql?query=SELECT * FROM ai_coach_modules_summary LIMIT 10
```
Executes a raw SQL SELECT query (for safety, only SELECT queries are allowed).

### Example Usage with curl

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers have High Risk insomnia classification?"}'

# Get schema
curl http://localhost:8000/schema

# Get statistics
curl http://localhost:8000/stats
```

### API Documentation

Once the service is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
LocalLLM/
├── app.py              # FastAPI application and routes
├── config.py           # Configuration management
├── database.py         # MySQL database connection and queries
├── llm_client.py       # Ollama LLM client
├── requirements.txt      # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md          # This file
```

## Configuration

All configuration is done through environment variables:

- `MYSQL_HOST`: MySQL server hostname or IP
- `MYSQL_USER`: MySQL username
- `MYSQL_PASSWORD`: MySQL password
- `MYSQL_DATABASE`: Database name
- `OLLAMA_API_URL`: Ollama API endpoint URL

## Security Notes

- Never commit `.env` file to version control
- Use strong passwords for database connections
- Consider implementing authentication for production use
- The SQL query endpoint only allows SELECT queries for safety
- For production, consider adding rate limiting and API authentication

## Troubleshooting

### Database Connection Issues
- Verify MySQL credentials in `.env`
- Check network connectivity to MySQL server
- Ensure MySQL user has proper permissions

### LLM API Issues
- Verify Ollama API URL is correct and accessible
- Check if Ollama service is running
- Verify network connectivity to Ollama host

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (requires 3.8+)

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if needed]

