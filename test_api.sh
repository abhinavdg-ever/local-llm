#!/bin/bash

# API Testing Script for Sleep Coach LLM Service
# API hosted at: http://72.60.96.212:8015

API_URL="http://72.60.96.212:8015"

echo "🧪 Testing Sleep Coach LLM API at ${API_URL}"
echo "=========================================="
echo ""

# Test 1: Health Check
echo "1️⃣  Health Check..."
curl -s "${API_URL}/health" | python3 -m json.tool
echo ""
echo ""

# Test 2: Root Endpoint
echo "2️⃣  Root Endpoint..."
curl -s "${API_URL}/" | python3 -m json.tool
echo ""
echo ""

# Test 3: Query Endpoint (Personal Data)
echo "3️⃣  Query - Personal Sleep Data (customer_id: 1290)..."
curl -s -X POST "${API_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "1290",
    "query": "What is my average sleep duration?"
  }' | python3 -m json.tool
echo ""
echo ""

# Test 4: Query Endpoint (Comparison)
echo "4️⃣  Query - Comparison..."
curl -s -X POST "${API_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "1290",
    "query": "How do I compare to others?"
  }' | python3 -m json.tool
echo ""
echo ""

# Test 5: Get Trends
echo "5️⃣  Get Trends (customer_id: 1290, last 30 days)..."
curl -s "${API_URL}/trends/1290?days=30" | python3 -m json.tool
echo ""
echo ""

# Test 6: Database Stats
echo "6️⃣  Database Statistics..."
curl -s "${API_URL}/stats" | python3 -m json.tool
echo ""
echo ""

echo "✅ All tests completed!"

