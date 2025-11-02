#!/usr/bin/env python3
"""
Test script to verify database and LLM connections
"""
import sys
import os
import mysql.connector
from mysql.connector import Error
import requests
from dotenv import load_dotenv

load_dotenv()

def test_database():
    """Test database connection"""
    print("Testing database connection...")
    try:
        config = {
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", ""),
            "database": os.getenv("MYSQL_DATABASE", "")
        }
        connection = mysql.connector.connect(**config)
        
        if connection.is_connected():
            print("✓ Database connection successful")
            
            # Try a simple query
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM ai_coach_modules_summary")
            count = cursor.fetchone()[0]
            print(f"✓ Table has {count} rows")
            cursor.close()
            connection.close()
            return True
        else:
            print("✗ Database connection failed")
            return False
    except Error as e:
        print(f"✗ Database error: {e}")
        return False

def test_llm():
    """Test LLM connection"""
    print("\nTesting LLM connection...")
    api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    try:
        payload = {
            "model": "llama3",
            "prompt": "Say 'test successful' if you can read this.",
            "stream": False
        }
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if result:
            print("✓ LLM connection successful")
            if "response" in result:
                print(f"  Response: {result['response'][:100]}...")
            return True
        else:
            print("✗ LLM returned empty response")
            return False
    except Exception as e:
        print(f"✗ LLM connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Connection Test Script")
    print("=" * 50)
    
    print(f"\nConfiguration:")
    print(f"  MySQL Host: {os.getenv('MYSQL_HOST', 'localhost')}")
    print(f"  MySQL Database: {os.getenv('MYSQL_DATABASE', '')}")
    print(f"  LLM API URL: {os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')}")
    print()
    
    db_ok = test_database()
    llm_ok = test_llm()
    
    print("\n" + "=" * 50)
    if db_ok and llm_ok:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

