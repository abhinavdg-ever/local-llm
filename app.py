"""
AI Sleep Coach Database Query Service
FastAPI application for AI-powered database query service
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import mysql.connector
from mysql.connector import Error
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration"""
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "")
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    
    @property
    def mysql_config(self):
        return {
            "host": self.MYSQL_HOST,
            "user": self.MYSQL_USER,
            "password": self.MYSQL_PASSWORD,
            "database": self.MYSQL_DATABASE,
            "charset": "utf8mb4",
            "collation": "utf8mb4_unicode_ci"
        }

config = Config()

# Database class
class Database:
    """MySQL database connection manager"""
    
    def __init__(self):
        self.connection: Optional[mysql.connector.MySQLConnection] = None
    
    def connect(self) -> bool:
        """Establish connection to MySQL database"""
        try:
            self.connection = mysql.connector.connect(**config.mysql_config)
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
        return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dictionaries"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                raise ConnectionError("Failed to connect to database")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_table_schema(self, table_name: str = "ai_coach_modules_summary") -> List[Dict[str, Any]]:
        """Get schema information for a table"""
        query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """
        return self.execute_query(query, (config.MYSQL_DATABASE, table_name))
    
    def get_table_sample(self, table_name: str = "ai_coach_modules_summary", limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows from a table"""
        query = f"SELECT * FROM {table_name} LIMIT %s"
        return self.execute_query(query, (limit,))
    
    def get_table_stats(self, table_name: str = "ai_coach_modules_summary") -> Dict[str, Any]:
        """Get basic statistics about a table"""
        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(count_query)
        return {
            "table_name": table_name,
            "row_count": result[0]["count"] if result else 0,
            "columns": [col["COLUMN_NAME"] for col in self.get_table_schema(table_name)]
        }

# LLM Client class
class LLMClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self):
        self.api_url = config.OLLAMA_API_URL
        self.default_model = "llama3"
    
    def generate(self, prompt: str, model: Optional[str] = None, stream: bool = False) -> Dict[str, Any]:
        """Generate response from LLM"""
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LLM API: {e}")
            raise
    
    def answer_question(self, question: str, context: str) -> str:
        """Generate an answer to a question based on context"""
        prompt = f"""You are a helpful AI assistant that answers questions about sleep and health data from a database.

Context information about the database:
{context}

User Question: {question}

Please provide a clear, accurate, and helpful answer based on the context provided. If the question cannot be answered with the given context, say so clearly.
"""
        
        try:
            result = self.generate(prompt)
            if "response" in result:
                return result["response"].strip()
            elif "text" in result:
                return result["text"].strip()
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

# Initialize FastAPI app
app = FastAPI(
    title="AI Sleep Coach Database Query Service",
    description="AI-powered service to answer questions about sleep and health data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and LLM client
db = Database()
llm_client = LLMClient()

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    table_name: Optional[str] = "ai_coach_modules_summary"

class QueryResponse(BaseModel):
    answer: str
    context_used: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    llm_available: bool

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    logger.info("Starting up application...")
    db.connect()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    logger.info("Shutting down application...")
    db.disconnect()

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Sleep Coach Database Query Service",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/query": "Query the database with natural language",
            "/schema": "Get database schema",
            "/stats": "Get database statistics"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    db_connected = db.connection and db.connection.is_connected()
    
    llm_available = False
    try:
        result = llm_client.generate("test", stream=False)
        llm_available = True
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")
    
    status = "healthy" if (db_connected and llm_available) else "degraded"
    
    return HealthResponse(
        status=status,
        database_connected=db_connected,
        llm_available=llm_available
    )

@app.get("/schema", tags=["Database"])
async def get_schema(table_name: str = "ai_coach_modules_summary"):
    """Get database schema for a table"""
    try:
        schema = db.get_table_schema(table_name)
        return {"table_name": table_name, "schema": schema}
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", tags=["Database"])
async def get_stats(table_name: str = "ai_coach_modules_summary"):
    """Get database statistics"""
    try:
        stats = db.get_table_stats(table_name)
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample", tags=["Database"])
async def get_sample(table_name: str = "ai_coach_modules_summary", limit: int = 5):
    """Get sample rows from a table"""
    try:
        sample = db.get_table_sample(table_name, limit)
        return {"table_name": table_name, "sample_data": sample, "limit": limit}
    except Exception as e:
        logger.error(f"Error getting sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_database(request: QueryRequest):
    """Answer questions about the database using natural language"""
    try:
        logger.info(f"Processing question: {request.question}")
        
        # Get context information
        schema = db.get_table_schema(request.table_name)
        sample = db.get_table_sample(request.table_name, limit=5)
        stats = db.get_table_stats(request.table_name)
        
        # Build context string
        context_parts = [
            f"Database: {config.MYSQL_DATABASE}",
            f"Table: {request.table_name}",
            f"Total Rows: {stats['row_count']}",
            "\nTable Schema:",
            ", ".join([col['COLUMN_NAME'] for col in schema]),
            "\nColumn Details:",
        ]
        
        for col in schema[:10]:
            col_info = f"- {col['COLUMN_NAME']}: {col['DATA_TYPE']}"
            if col.get('CHARACTER_MAXIMUM_LENGTH'):
                col_info += f" (max length: {col['CHARACTER_MAXIMUM_LENGTH']})"
            context_parts.append(col_info)
        
        context_parts.append("\nSample Data (first 5 rows):")
        for i, row in enumerate(sample, 1):
            context_parts.append(f"\nRow {i}:")
            for key, value in list(row.items())[:10]:
                context_parts.append(f"  {key}: {value}")
        
        context = "\n".join(context_parts)
        
        # Generate answer using LLM
        answer = llm_client.answer_question(request.question, context)
        
        return QueryResponse(
            answer=answer,
            context_used={
                "table_name": request.table_name,
                "row_count": stats['row_count'],
                "column_count": len(schema)
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/sql", tags=["Query"])
async def execute_sql_query(query: str):
    """Execute a raw SQL query (only SELECT queries are allowed for safety)"""
    if not query.strip().upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")
    
    try:
        results = db.execute_query(query)
        return {"query": query, "results": results, "row_count": len(results)}
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
